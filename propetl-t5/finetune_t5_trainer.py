import sys
import torch
import datasets
import json
import logging
import os
from pathlib import Path

from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import EvaluationStrategy
from adapters.adapter_controller import GetSubnet
from third_party.models import T5Config, T5ForConditionalGeneration, MT5Config, MT5ForConditionalGeneration
from third_party.trainers import T5Trainer
from adapters import AdapterController, AutoAdapterConfig
from data import AutoTask
from third_party.utils import TaskCollator, check_output_dir
from metrics import build_compute_metrics_fn
from training_args import Seq2SeqTrainingArguments, ModelArguments, DataTrainingArguments, \
    AdapterTrainingArguments
from utils import freezing_params, get_last_checkpoint_path,\
    handle_metrics, get_training_args
import wandb
logger = logging.getLogger(__name__)


def remove_rank_info_from_argv(args):
    extra_parameters = {}
    if args[1].startswith("--local_rank"):
        extra_parameters.update({'local_rank': int(args[1].split('=')[-1])})
        del args[1]
    return extra_parameters

def main():
    # See all possible arguments in src/transformers/training_args.py or by passing
    # the --help flag to this script. We now keep distinct sets of args, for a cleaner
    # separation of concerns.
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, AdapterTrainingArguments))
    
    
    # For running on multiple gpus with torch.distributed.launch, it adds a local_rank paramter, to allow the parser
    # still use the config file, we add the local_rank to the config file.
    if len(sys.argv) > 3 and sys.argv[1].startswith("--local_rank") and (sys.argv[2].endswith(".json")):
        rank_info = remove_rank_info_from_argv(sys.argv)
        args_dict = json.loads(Path(sys.argv[1]).read_text())
        args_dict.update(rank_info)
        model_args, data_args, training_args, adapter_args = parser.parse_dict(args_dict)
    elif len(sys.argv) == 3 and sys.argv[1].endswith(".json"):
        logger.warning("config path: %s", sys.argv[1])
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
        training_args.seed =   int(sys.argv[2])
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    if "wmt16-ro-en" in  data_args.tasks or  "wmt16-en-ro" in  data_args.tasks:
        project = "mT5MT"
    elif "ud_en" in  data_args.tasks:
        project = "T5CrossLingual"
    elif "xnli_en" in  data_args.tasks:
        project = "T5MultiTask-Multilingual-final" 
    elif "xsum" in data_args.tasks:
        project = "T5XSum"
    else:
        project =  "T5_GLUE_xlarge"
    run = wandb.init(project=project)
    if run.name != None:
        training_args.output_dir += run.name 
    check_output_dir(training_args)
    
    if data_args.train_tasks == None or len(data_args.train_tasks) == 0:
        data_args.train_tasks = data_args.tasks
    # Setup logging
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(training_args.output_dir + "/terminal_logs.txt", "w") as f:
        f.write("start")
    logging.basicConfig(
        filename=training_args.output_dir + "/terminal_logs.txt",
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if "mt5" in model_args.model_name_or_path:
        config = MT5Config.from_pretrained(
            model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        )
    else:
        config = T5Config.from_pretrained(
            model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        )
    extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout",
                          "attention_dropout",  "train_adapters")
    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(training_args, p))

    # Gets the adapter config and updates the specified parameters.
    if training_args.train_adapters:
        adapter_config = AutoAdapterConfig.get("adapter")
        adapter_config.input_dim = config.d_model
        adapter_config.tasks = data_args.tasks
        adapter_config.task_to_adapter = {task:adapter for task, adapter in zip(data_args.tasks, data_args.adapters)} if data_args.adapters is not None else None
        # If this is a parametric task embedding this mapping makes sense, but in case we use any task embeddings,
        # then, we do not need any mapping as we use the pretrained task embeddings.

        extra_adapter_params = (
                                "adapter_config_name",
                                "add_layer_norm_before_adapter",
                                "add_layer_norm_after_adapter",
                                "reduction_factor",
                                "non_linearity",
                                "sparsity",
                                "share_adapter",
                                "share_encoder_decoder_single_adapter",
                                "mask_extreme_mode",
                                "mask_extreme_mode_combine_method",
                                "use_multilingual")

        for p in extra_adapter_params:
            if hasattr(adapter_args, p) and hasattr(adapter_config, p):
                setattr(adapter_config, p, getattr(adapter_args, p))
            else:
                logger.warning(f"({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute")
        assert adapter_config.adapter_config_name == 'preiffer'
        adapter_config.device = training_args.device
    else:
        adapter_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else \
            model_args.model_name_or_path,
             use_fast=False
    )
    
    
    model_class = MT5ForConditionalGeneration if "mt5" in model_args.model_name_or_path else T5ForConditionalGeneration
        
    if model_args.not_load_t5_checkpoint:
        model = model_class(config=config, adapter_config=adapter_config)
    else:
        last_checkpoint_path = training_args.output_dir
        model_path = model_args.model_name_or_path if ((training_args.optimize_from_scratch and not training_args.optimize_from_scratch_with_loading_model) or not os.path.exists(os.path.join(last_checkpoint_path, 'pytorch_model.bin')))\
            else last_checkpoint_path
        logger.warning("model path loaded from : %s", model_path)
        model = model_class.from_pretrained(
            model_path,
            from_tf=".ckpt" in model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            adapter_config=adapter_config
        )
    #model = torch.compile(model)
    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    # freezing the parameters.
    if training_args.do_train:
        freezing_params(model, training_args, model_args, adapter_args)

    if training_args.print_num_parameters:
        #logger.info(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info("Parameter name %s", name)
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total trainable parameters %s", total_trainable_params)
        logger.info("Total parameters %s", total_params)
    # Gets the training/test/validation datasets.
    dataset_class = AutoTask
    if training_args.do_train:
        train_datasets = [dataset_class.get(task, seed=data_args.data_seed).get_dataset(
            split="train",  n_obs=data_args.n_train, add_prefix=False if training_args.train_adapters or training_args.add_prefix == False else True)
            for task in data_args.train_tasks]
        dataset_sizes = [len(train_dataset) for train_dataset in train_datasets]
        train_dataset = datasets.concatenate_datasets(train_datasets)
    training_args.remove_unused_columns = False
    eval_datasets = ({task: dataset_class.get(task, seed=data_args.data_seed).get_dataset(
        split="validation" if task in data_args.train_tasks else "test",  n_obs=data_args.n_val, ## for cross lingual transfer, some task only have test set.
        add_prefix=False if training_args.train_adapters or training_args.add_prefix == False else True,
        split_validation_test=training_args.split_validation_test)
                         for task in data_args.eval_tasks}
                     if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
                     else None)

    if training_args.do_test:
        test_dataset = (
        {task: dataset_class.get(task, seed=data_args.data_seed).get_dataset(
            split="test", n_obs=data_args.n_test,
            add_prefix=False if training_args.train_adapters or training_args.add_prefix == False else True,
            split_validation_test=training_args.split_validation_test)
            for task in data_args.eval_tasks} if training_args.do_test else None
        )
    # Defines the metrics for evaluation.
    compute_metrics_fn = (
        build_compute_metrics_fn(data_args.eval_tasks, tokenizer) if training_args.predict_with_generate else None
    )
    # Defines the trainer.
    
    # if trainer.is_world_process_zero():
    #     arguments = get_training_args([model_args, data_args, training_args, adapter_args])
    #     handle_metrics("arguments", arguments, training_args.output_dir)

    # Trains the model.
    if training_args.do_train:
        trainer = T5Trainer(
            model=model,
            config=config,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_datasets,
            data_collator=TaskCollator(tokenizer, data_args, tpu_num_cores=training_args.tpu_num_cores),
            compute_metrics=None,
            multi_task_compute_metrics=compute_metrics_fn,
            data_args=data_args,
            dataset_sizes=dataset_sizes if training_args.do_train else None,
            adapter_config=adapter_config
        )
        if trainer.is_world_process_zero():
           last_checkpoint_path = training_args.output_dir
           model_path = model_args.model_name_or_path if (training_args.optimize_from_scratch or not os.path.exists(os.path.join(last_checkpoint_path, 'pytorch_model.bin')))\
             else last_checkpoint_path
        if training_args.compute_time:
           torch.cuda.synchronize()  # wait for move to complete
           start = torch.cuda.Event(enable_timing=True)
           end = torch.cuda.Event(enable_timing=True)
           start.record()
        trainer.train(
            #get_last_checkpoint_path(training_args.output_dir) \
            model_path=model_path \
                if (os.path.exists(training_args.output_dir) and not training_args.optimize_from_scratch) else None,
        )
        if training_args.compute_time: 
           torch.cuda.synchronize()  # wait for all_reduce to complete
           end.record()
           total_time = {"total_time": start.elapsed_time(end)}
           print("###### total_time ", total_time)
        trainer.save_model()
        to_save_state_dict = {}
        model_state_dict = torch.load(os.path.join(training_args.output_dir, "pytorch_model.bin"))
        model_param = dict(model.named_parameters())

        for param_name in model_state_dict:
            #only save those that requires grad
            # if a parameter is shared across layer. named_parameters will only return one copy of it, but state_dict will return all (same) copies of it.
            if param_name in model_param and model_param[param_name].requires_grad:
          
                if param_name in model_state_dict and "mask_scores" in param_name and "param" in param_name:
                    #save mask_scores as binaries
                    to_save_state_dict[param_name] = GetSubnet.apply(model_state_dict[param_name].abs(), adapter_config.sparsity).bool()
                    #to_save_state_dict[param_name] = model_state_dict[param_name]
                else:
                    to_save_state_dict[param_name] = model_state_dict[param_name]
     
        torch.save(to_save_state_dict, os.path.join(training_args.output_dir, "propetl_weight.bin"))
                   
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
            tokenizer.save_pretrained(training_args.output_dir)
    
    # Evaluation
    all_metrics = {}
    if training_args.do_eval or training_args.do_test:
        if True:
            # By default we load  the model from last checkpoint path,
            # in case of saving the model with the best metrics, make sure to
            # set save_total = 1 so the best model is loaded here.
            # if not exists returns the path to the output_dir.
            last_checkpoint_path = get_last_checkpoint_path(training_args.output_dir)
            #last_checkpoint_path = os.path.join(training_args.output_dir, "checkpoint-600")
            if  os.path.exists(os.path.join(training_args.output_dir, "propetl_weight.bin")):
                #last_checkpoint_path = model_args.model_name_or_path #we just load pretrained weight (T5-base), and seperately load the propetl weight
                propetl_state_dict = torch.load(os.path.join(training_args.output_dir, "propetl_weight.bin"))
                
                
            config = T5Config.from_pretrained(
                last_checkpoint_path,
                cache_dir=model_args.cache_dir)
            #propetl will be loaded seperately
            model = T5ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                from_tf=".ckpt" in training_args.output_dir,
                config=config,
                cache_dir=model_args.cache_dir,
                adapter_config=adapter_config
            )
            if training_args.train_adapters:
                if data_args.adapters is not None:
                    for name, sub_module in model.named_modules():
                        task_to_adapter = {eval_task: adapter for eval_task, adapter in
                                        zip(data_args.eval_tasks, data_args.adapters)}
                        if isinstance(sub_module, AdapterController):
                            sub_module.set_task_to_adapter_map(task_to_adapter)
            if  os.path.exists(os.path.join(training_args.output_dir, "propetl_weight.bin")):
 
                for param_name in propetl_state_dict:
                    if "mask_scores" in param_name and "param" in param_name:
                        
                        propetl_state_dict[param_name] = propetl_state_dict[param_name].float() #make sure we load the mask_scores as float

                model.load_state_dict(propetl_state_dict, strict=False)

            
            # NOTE: if trainer is not re-defined, there is a bug in the codes, that making
            # huggingface codes does not using the best checkpoint.
            trainer = T5Trainer(
                model=model,
                config=config,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_datasets,
                data_collator=TaskCollator(tokenizer, data_args, tpu_num_cores=training_args.tpu_num_cores),
                compute_metrics=None,
                multi_task_compute_metrics=compute_metrics_fn,
                data_args=data_args,
                dataset_sizes=dataset_sizes if training_args.do_train else None,
                adapter_config=adapter_config
            )

        

    if training_args.do_eval:
        metrics = trainer.evaluate()
        if trainer.is_world_process_zero():
            handle_metrics("val", metrics, training_args.output_dir)
            all_metrics.update(metrics)

    if training_args.do_test:
        metrics = trainer.evaluate(test_dataset, is_test=True)
        if trainer.is_world_process_zero():
            handle_metrics("test", metrics, training_args.output_dir)
            all_metrics.update(metrics)

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = torch.cuda.max_memory_allocated()/1024**2
        print(
            "Memory utilization",
            peak_memory,
            "MB"
        )
        memory_usage = {"peak_memory": peak_memory}
    return all_metrics


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
