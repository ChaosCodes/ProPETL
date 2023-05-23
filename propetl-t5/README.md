# ProPETL-T5

This folder contains a comprehensive set of scripts aimed at reproducing the experiments listed in Table 2 and Figure 3 in our paper "One Network, Many Masks: Towards More Parameter-Efficient Transfer Learning". 

## Setup & Installation

The project uses a Python environment. All necessary dependencies can be installed with the following command:

```bash
pip install -r requirements.txt
```

## Codebase Overview

The codebase is structured as follows:

configs/: This directory contains all configuration files for different experiments. \
third_party/: Here, we provide a modified version of the T5 modeling code and trainer from Hugging Face Transformers. \
finetune_t5_trainer.py: This is the main script responsible for running experiments.\
metrics.py: This script contains the metrics used for different datasets.\
data.py: This script handles data processing tasks for different datasets.\
adapter.py: This script contains the adapter code related to our research.


## Usage
The project includes numerous experiment configurations located within the configs/ directory. To run a specific experiment, use the following command format:
```bash
CUDA_VISIBLE_DEVICES=0 python3 finetune_t5_trainer.py configs/<config_filename> <random_seed>
```
Here, <config_filename> should be replaced with the configuration file you wish to run, and <random_seed> should be replaced with a numeric seed for random number generation.

For instance, to reproduce the ProPEFT results from Table 2 of our paper, you would run:
```bash
CUDA_VISIBLE_DEVICES=0 python3 finetune_t5_trainer.py configs/glue/propetl_adapter_reduction12.json 42
```

We welcome any questions or feedback related to our work.
