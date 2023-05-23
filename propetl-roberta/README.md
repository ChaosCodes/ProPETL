# One Network, Many Masks: Towards More Parameter-Efficient Transfer Learning
This dicrectory contains code to run all the RoBERTa experiments in our paper. The code is based on [adapterhub](https://github.com/adapter-hub/adapter-transformers).

# Environments

We use Pytorch 1.11.0+cu113 and A100.
Before running the code, please install the requirements and the propetl package by
```
python install -r requirements
python install .
```

# How to run the models
To reproduce the experiment in the paper Table 1, you can simply run the following 3 shell (Adapter, LoRA and prefix).

Model | CoLA | SST-2 | MRPC | QQP | STS-B | MNLI | QNLI | RTE | Avg
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
ProAdapter | 65.43 | 94.15 | 88.24/91.41 | 89.40/86.04 | 91.34/90.95 | 86.53 | 92.58 | 76.50 | 86.6
ProLoRA | 61.81 | 94.00 | 87.42/91.00 | 88.85/85.22 | 90.48/90.47 | 85.73 | 91.05 | 63.79 | 84.53
ProPrefix | 62.16 | 93.62 | 88.73/91.80 | 87.59/83.71 | 90.92/90.83 | 85.30 | 91.75 | 72.66 | 85.37

```
# propetl adapter
bash scripts/run_adapter.sh
# propetl LoRA
bash scripts/run_lora.sh
# propetl prefix tuning
bash scripts/run_prefix.sh
```
