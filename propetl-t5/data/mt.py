from collections import OrderedDict

import abc
import datasets
import functools
import logging
import numpy as np
import torch
from metrics import metrics
from typing import Callable, Dict, Mapping, List
from .abstract_task import AbstractTaskDataset
from .utils import round_stsb_target, compute_task_max_decoding_length


class WMT15_FR_EN_TaskDataset(AbstractTaskDataset):
    name = "wmt15-fr-en"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"fr-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt15", self.pair,
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["Translate Frecnch to English:",example['translation']["fr"]]
        tgt_texts = [example['translation']["en"]]
        return self.seq2seq_format( src_texts, tgt_texts, add_prefix)
    
    
class WMT18_CS_EN_TaskDataset(AbstractTaskDataset):
    name = "wmt18-cs-en"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"cs-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt18", self.pair,
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["Translate Czech to English:",example['translation']["cs"]]
        tgt_texts = [example['translation']["en"]]
        return self.seq2seq_format( src_texts, tgt_texts, add_prefix)
    
    
    
class WMT18_DE_EN_TaskDataset(AbstractTaskDataset):
    name = "wmt18-de-en"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"de-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt18", self.pair,
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["Translate German to English:",example['translation']["de"]]
        tgt_texts = [example['translation']["en"]]
        return self.seq2seq_format( src_texts, tgt_texts, add_prefix)
    
    
class WMT18_FI_EN_TaskDataset(AbstractTaskDataset):
    name = "wmt18-fi-en"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"fi-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt18", self.pair,
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["Translate Finnish to English:",example['translation']["fi"]]
        tgt_texts = [example['translation']["en"]]
        return self.seq2seq_format( src_texts, tgt_texts, add_prefix)