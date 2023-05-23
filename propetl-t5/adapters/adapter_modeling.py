"""Implements an Adapter and Hyper-adapter Layers."""
import torch
import torch.nn as nn

from .adapter_outputs import (SamplerOutput, LayerNormOutput,
                              AdapterT5BlockOutput, AdapterOutput)
from .adapter_utils import Activations, linear_layer
import math
import torch.nn.functional as F





class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sparsity = self.config.sparsity
        self.input_dim = config.input_dim
        self.weight_init_range = config.weight_init_range
        if config.reduction_factor > 1:
            self.down_sample_size = int(self.input_dim // config.reduction_factor)
        else:
            self.down_sample_size = int(self.input_dim / config.reduction_factor)
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = linear_layer(self.input_dim, self.down_sample_size, std=self.weight_init_range)
        self.up_sampler = linear_layer(self.down_sample_size, self.input_dim, std=self.weight_init_range)
        
    def forward(self, x, down_mask=None, up_mask=None):
        if down_mask != None:
            return self.forward_with_mask(x, down_mask=down_mask, up_mask=up_mask)
        else:
            z = self.down_sampler(x)
            z = self.activation(z)
            return self.up_sampler(z)
            
    def forward_with_mask(self, x, down_mask=None, up_mask=None):
        w1 = self.down_sampler.weight * down_mask
        down = F.linear(x, w1, self.down_sampler.bias)
        down = self.activation(down)
        w2 = self.up_sampler.weight * up_mask
        up = F.linear(down, w2, self.up_sampler.bias)
        return up


