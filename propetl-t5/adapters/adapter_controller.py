"""Implements Adapter Controller, a module that keeps multiple
layers of Adapters, and controls which adapter layer to use."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation
import math
from .adapter_configuration import AdapterConfig
from .adapter_modeling import Adapter
import torch.autograd as autograd

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None
    
def create_score_for_adapter(adapter_model):
    ef_attn_down_scores = nn.Parameter(torch.Tensor(adapter_model.down_sampler.weight.size()))
    ef_attn_up_scores = nn.Parameter(torch.Tensor(adapter_model.up_sampler.weight.size()))
    nn.init.kaiming_uniform_(ef_attn_down_scores, a=math.sqrt(5))
    nn.init.kaiming_uniform_(ef_attn_up_scores, a=math.sqrt(5))
    return ef_attn_down_scores, ef_attn_up_scores

class ParamMask(nn.Module):
    def __init__(self, param, sparsity):
        super().__init__()
        self.param = param
        self.sparsity = sparsity
    def forward(self):
        mask = GetSubnet.apply(self.param.abs(), self.sparsity)
        return mask
    
class AdapterController(nn.Module):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.adapters = nn.ModuleDict(dict())
        self.tasks = config.tasks
        self.mask_extreme_mode = self.config.mask_extreme_mode
        self.mask_extreme_mode_combine_method = self.config.mask_extreme_mode_combine_method
        self.use_multilingual = config.use_multilingual
        # If a dictionary from task to adapter is given, the task is over-written by the given adapters.
        if config.task_to_adapter is not None:
            self.task_to_adapter_name = config.task_to_adapter
        else:
            self.task_to_adapter_name = {task: task for task in self.tasks}
        self.adapter_names = self.task_to_adapter_name.values()
        self.adapters = self.construct_adapters(self.adapter_names)
            
        self.mask_scores = {}
        
        if self.use_multilingual:
            real_tasks = list(set([ task_language.split("_")[0] for task_language in self.tasks]))
            languages = list(set([ task_language.split("_")[1] for task_language in self.tasks]))
        else:
            real_tasks = self.tasks
            languages = []
        
        for task in real_tasks:
            down_scores, up_scores = create_score_for_adapter(self.adapters["common_adapter"])
            self.mask_scores[task+ "_down_mask"] = ParamMask(down_scores, self.config.sparsity)
            self.mask_scores[task+ "_up_mask"] = ParamMask(up_scores, self.config.sparsity)
        
        for language in languages:
            down_scores, up_scores = create_score_for_adapter(self.adapters["common_adapter"])
            self.mask_scores[language+ "_down_mask"] = ParamMask(down_scores, self.config.sparsity)
            self.mask_scores[language+ "_up_mask"] = ParamMask(up_scores, self.config.sparsity)
        
        self.mask_scores = nn.ModuleDict(self.mask_scores)
        if self.mask_extreme_mode:
            self.mask_scores["layer"+ "_down_mask"] = ParamMask(down_scores, self.config.sparsity)
            self.mask_scores["layer"+ "_up_mask"] = ParamMask(up_scores, self.config.sparsity)
        

    def set_task_to_adapter_map(self, mapping):
        self.task_to_adapter_name = mapping

    def get_adapter_name(self, task):
        return self.task_to_adapter_name[task]

    def construct_adapters(self, adapter_names):
        """
        Constructs adapter layers and adds them to a dictionary for the given
        tasks.
        Args:
            tasks: A list of string containing the task names.
        """
        for name in adapter_names:
            self.adapters[name] = Adapter(self.config)
        return self.adapters

    def disable_adapters(self, adapter_names):
        """
        Given a list of tasks, it freezes their corresponding adapter layers'
        parameters.
        Args:
           tasks: List of tasks.
        """
        adapter_names = self.convert_to_list(adapter_names)
        for name in adapter_names:
            adapter = self.get_adapter(name)
            for param in adapter.parameters():
                param.requires_grad = False

    def convert_to_list(self, adapter_names):
        if isinstance(adapter_names, list):
            return adapter_names
        return [adapter_names]

    def enable_adapters(self, adapter_names):
        """
        Given a list of tasks, it unfreezes their corresponding adapter layers.
        Args:
            tasks: Given list of tasks.
        """
        adapter_names = self.convert_to_list(adapter_names)
        for name in adapter_names:
            adapter = self.get_adapter(name)
            for param in adapter.parameters():
                param.requires_grad = True

    def get_adapter(self, adapter_name):
        """Given a task returns its corresponding adapter layer.
        Args:
            task: Input task name.
        Returns:
            Adapter layer corresponding to the given task.
        """
        return self.adapters[adapter_name]

    def forward(self, task, inputs, hidden_states_before_ff=None):
        """Retrieves the adapter layer corresponding to the given
        task. It freezes the adapter layers for all the other tasks
        and call the selected adapter layer.
        Args:
            task: the name of the current task.
            inputs: the inputs to feed in in the adapter layer.
        Returns:
            outputs of the adapter layer."""
        if self.use_multilingual: 
            if self.mask_extreme_mode_combine_method == "add": 
                down_mask = self.mask_scores[task.split("_")[0]+ "_down_mask"]() + self.mask_scores[task.split("_")[1]+ "_down_mask"]() +  self.mask_scores["layer" + "_down_mask"]()  
                up_mask = self.mask_scores[task.split("_")[0]+ "_up_mask"]() + self.mask_scores[task.split("_")[1]+ "_up_mask"]() + self.mask_scores["layer" + "_up_mask"]() 
            else:
                down_mask = 1 - (1 - self.mask_scores[task.split("_")[0]+ "_down_mask"]()) * (1 - self.mask_scores[task.split("_")[1]+ "_down_mask"]()) * (1 - self.mask_scores["layer" + "_down_mask"]()) # Or operation written in this way for easier gradient calculation
                up_mask = 1 - (1 - self.mask_scores[task.split("_")[0]+ "_up_mask"]()) *  (1 - self.mask_scores[task.split("_")[1]+ "_up_mask"]()) * (1 - self.mask_scores["layer" + "_up_mask"]()) # OR operation written in this way for easier gradient calculation
        else:
            if not self.mask_extreme_mode:
                down_mask = self.mask_scores[task+ "_down_mask"]()
                up_mask = self.mask_scores[task+ "_up_mask"]()
            elif self.mask_extreme_mode_combine_method == "and": 
                down_mask = self.mask_scores[task+ "_down_mask"]() *  self.mask_scores["layer" + "_down_mask"]() # AND operation written in this way for easier gradient calculation
                up_mask = self.mask_scores[task+ "_up_mask"]() * self.mask_scores["layer" + "_up_mask"]() # AND operation written in this way for easier gradient calculation
            elif self.mask_extreme_mode_combine_method == "or": 
                down_mask = 1 - (1 - self.mask_scores[task+ "_down_mask"]()) * (1 - self.mask_scores["layer" + "_down_mask"]()) # OR  operation written in this way for easier gradient calculation
                up_mask = 1 - (1 - self.mask_scores[task+ "_up_mask"]()) * (1 - self.mask_scores["layer" + "_up_mask"]()) # OR operation written in this way for easier gradient calculation
            elif self.mask_extreme_mode_combine_method == "add": 
                down_mask = self.mask_scores[task+ "_down_mask"]() +  self.mask_scores["layer" + "_down_mask"]() 
                up_mask = self.mask_scores[task+ "_up_mask"]() + self.mask_scores["layer" + "_up_mask"]() 
        adapter_name = self.get_adapter_name(task)
        # Enables the adapter layer for the given task.
        self.enable_adapters(adapter_name)
        # Disable other adapters.
        other_adapter_names = [x for x in self.adapter_names if x != adapter_name]
        self.disable_adapters(other_adapter_names)
        adapter = self.get_adapter(adapter_name)
        z = inputs + hidden_states_before_ff if self.config.adapter_config_name == "preiffer" else inputs
        outputs = adapter(z, down_mask, up_mask)
        outputs = outputs + inputs
        return outputs



class AutoAdapterController(nn.Module):
    """Generic adapter controller class to instantiate different adapter
    controller classes."""

    @classmethod
    def get(cls, config):
        assert isinstance(config, AdapterConfig)
        return AdapterController(config)

