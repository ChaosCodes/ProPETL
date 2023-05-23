# One Network, Many Masks: Towards More Parameter-Efficient Transfer Learning
This repository is for our ACL' 23 paper:
One Network, Many Masks: Towards More Parameter-Efficient Transfer Learning
The code is based on [adapterhub](https://github.com/adapter-hub/adapter-transformers).

# Environments

We use Pytorch 1.11.0+cu113 and A100.
Before running the code, please install the requirements and the propetl package by
```
python install -r requirements
python install .
```

# How to run the models
We have 3 shell files to run the 3 models (Adapter, LoRA and prefix) in the paper.

```
# propetl adapter
bash scripts/run_adapter.sh
# propetl LoRA
bash scripts/run_lora.sh
# propetl prefix tuning
bash scripts/run_prefix.sh
```

# Reference
If you find this repository useful, please cite our paper:
```
@inproceedings{zeng2023onenetwork,
  title={One Network, Many Masks: Towards More Parameter-Efficient Transfer Learning},
  author={Guangtao Zeng and Peiyuan Zhang and Wei Lu},
  booktitle={Proceedings of ACL},
  year={2023}
}
```