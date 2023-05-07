# One Network, Many Masks: Towards More Parameter-Efficient Transfer Learning

This repo contains code  to run all the RoBERTa experiments in our anounymous paper submission:  One Network, Many Masks: Towards More Parameter-Efficient Transfer Learning


# Environments

We use Pytorch 1.11.0+cu113 and A100.
For other pakages, simply install them via

```
python install -r requirements
python install .
```

# How to run the models

Simply run propetl by

```
# propetl adapter
bash run_adapter.sh
# propetl LoRA
bash run_lora.sh
# propetl prefix tuning
bash run_prefix.sh
```

Config settings can be set in the 3 shell files above

# Future Release

We will clean up the code, add more comments and docs when we release the code to the public in the future.

