import json
import numpy as np

task_metric = {
    'cola': ["eval_matthews_correlation"],
    'sst2': ["eval_accuracy"],
    'qnli': ["eval_accuracy"],
    'mnli': ["eval_accuracy"],
    'rte': ["eval_accuracy"],
    'mrpc': ["eval_combined_score", "eval_accuracy", "eval_f1"],
    'stsb': ["eval_combined_score", "eval_pearson", "eval_spearmanr"],
    'qqp': ["eval_combined_score", "eval_accuracy", "eval_f1"],
}

lora_alpha=32
sparsity=0.5
adapter_reduction_factor=12


ave_task_acc=[]


res_string = ""
for TASK_NAME in ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte']:
# for TASK_NAME in ['sst2']:
    acc = []
    second_acc = []
    for seed in [42, 43, 44]:
        # directory_name = f'adapter/original.sd_{seed}.arf_{adapter_reduction_factor}.spsty_{sparsity}.mask_lr_3e-3.specifc_epoch'
        directory_name = f'adapter/share_and_mask.sd_{seed}.arf_{adapter_reduction_factor}.spsty_{sparsity}.mask_lr_3e-3.specifc_epoch'

        test_results = f'checkpoints/{TASK_NAME}/{directory_name}/test_results.json'
        with open(test_results, 'r') as j:
            res = json.loads(j.read())
        if TASK_NAME in ['mrpc','qqp', 'stsb']:
            acc.append(res[task_metric[TASK_NAME][1]] * 100)
            second_acc.append(res[task_metric[TASK_NAME][2]] * 100)
            ave_task_acc.extend([res[task_metric[TASK_NAME][1]] * 100, res[task_metric[TASK_NAME][2]] * 100])
        else:
            acc.append(res[task_metric[TASK_NAME][0]] * 100)
            ave_task_acc.append(res[task_metric[TASK_NAME][0]] * 100)

    print(TASK_NAME)
    # res_string+=f' {np.mean(acc):.2f}_{{\pm {np.std(acc):.2f}}} &'
    print(np.mean(acc), np.std(acc))
    if len(second_acc) == 0:
        res_string+=f' {np.mean(acc):.2f} &'
    else:
        res_string+=f' {np.mean(acc):.2f}/{np.mean(second_acc):.2f} &'
        print(np.mean(second_acc), np.std(second_acc))
res_string+=f' {np.mean(ave_task_acc): .2f}'

print(np.mean(ave_task_acc))
print(res_string)

# res_string = ""
# for TASK_NAME in ['mrpc','qqp', 'stsb']:
# # for TASK_NAME in ['mrpc', 'stsb']:
#     acc = [[], []]
#     for seed in [42, 43, 44]:
#         directory_name = f'adapter/test.sd_{seed}.arf_12_mlr_{mask_lr}'
#         # directory_name = f'lora/share_and_mask.sd_{seed}.lora_r_32.lora_alpha_{lora_alpha}'
#         # directory_name = f'lora/true_original.sd_{seed}.lora_r_32.lora_alpha_32'
#         # directory_name = f'sparsity/share_and_mask.sd_{seed}.spsty_{sparsity}.arf{adapter_reduction_factor}'


#         test_results = f'checkpoints/{TASK_NAME}/{directory_name}/test_results.json'
#         with open(test_results, 'r') as j:
#             res = json.loads(j.read())
#         for i in range(2):
#             acc[i].append(res[task_metric[TASK_NAME][i + 1]] * 100)
#     for i in range(2):
#         print(TASK_NAME)
#         print(np.mean(acc[i]), np.std(acc[i]))
#         # res_string += f' {np.mean(acc[i]):.2f}_{{\pm {np.std(acc[i]):.2f}}} '
#         res_string += f'{np.mean(acc[i]):.2f}'
#         if i == 0:
#             res_string += "/"
#     res_string+='&'
# print(res_string)
