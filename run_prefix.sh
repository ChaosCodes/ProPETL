
export CUDA_VISIBLE_DEVICES=4
# export TASK_NAME=mnli

# export WANDB_API_KEY=None


# define metric
declare -A metrics
metrics['cola']=matthews_correlation
metrics['sst2']=accuracy
metrics['mrpc']=combined_score
metrics['qqp']=combined_score
metrics['stsb']=combined_score
metrics['rte']=accuracy
metrics['qnli']=accuracy
metrics['mnli']=accuracy

declare -A t2epoch
t2epoch['cola']=20
t2epoch['sst2']=10
t2epoch['mrpc']=20
t2epoch['qqp']=10
t2epoch['stsb']=20
t2epoch['rte']=20
t2epoch['qnli']=10
t2epoch['mnli']=10

sparsity=0.1
mask_lr=1e-4
# epoch=10

extra_cmd_for_prefix=""

# small 'rte' 'mrpc' 'stsb' 'cola'
# large 'sst2' 'qqp' 'qnli' 'mnli'

# lora
# adapter_config=lora
lora_r=32

# adapter
# adapter_config=pfeiffer
adapter_reduction_factor=12

# prefix
adapter_config=prefix_tuning
prefix_length=64


for TASK_NAME in 'sst2' 'qqp' 'qnli' 'mnli' 'rte' 'mrpc' 'stsb' 'cola'
do
  export WANDB_PROJECT=ProPETprefix.${TASK_NAME}
  # export WANDB_PROJECT=test

  metric=${metrics[${TASK_NAME}]}

  

  extra_cmd=""

  for seed in 42 43 44
  do
    # share
    # exp_name=share_and_mask.sd_${seed}.arf_${adapter_reduction_factor}.spsty_${sparsity}
    exp_name=share_and_mask.sd_${seed}.prefix_length_${prefix_length}.spsty_${sparsity}.mask_lr_${mask_lr}.specifc_epoch
    extra_cmd="--share_adapter"

    # not share
    # exp_name=original.sd_${seed}.prefix_length_${prefix_length}.specifc_epoch
    # exp_name=original.sd_${seed}.arf_${adapter_reduction_factor}

    SAVE=/root/autodl-tmp/checkpoints/${TASK_NAME}/proprefix/${exp_name}/

    SAVE_FILE=/root/autodl-tmp/checkpoints/${TASK_NAME}/proprefix/${exp_name}/test_results.json

    rm -rf ${SAVE}; mkdir -p ${SAVE}

    # --load_adapter checkpoints/cola/20221220/spsty_0.5.sd_44.arf_12/checkpoint-268/cola/ \
    until [ -e ${SAVE_FILE} ]
    do

      python examples/pytorch/text-classification/run_glue.py \
        --model_name_or_path roberta-base \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_predict \
        --fp16 \
        --max_seq_length 128 \
        --per_device_train_batch_size 128 \
        --learning_rate 1e-4 \
        --mask_learning_rate ${mask_lr} \
        --num_train_epochs ${t2epoch[${TASK_NAME}]} \
        --prefix_length ${prefix_length} \
        --lora_r ${lora_r} \
        --overwrite_output_dir \
        --train_adapter \
        --warmup_ratio 0.1 \
        --save_total_limit=2 \
        --adapter_config ${adapter_config} \
        --evaluation_strategy epoch \
        --sparsity ${sparsity} \
        --save_strategy epoch \
        --load_best_model_at_end True \
        --metric_for_best_model ${metric} \
        --weight_decay 0.1 \
        --run_name ${TASK_NAME}.${exp_name} \
        --output_dir ${SAVE} \
        --seed ${seed} \
        --adapter_reduction_factor ${adapter_reduction_factor} ${extra_cmd}
    done
  done
done