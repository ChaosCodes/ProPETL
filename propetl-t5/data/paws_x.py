
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

class Paws_X_TaskDataset(AbstractTaskDataset):

    label_list = ["0", "1"]
    #label_list = ["different", "paraphrase"]
    #label2id = {"0":"different", "1":"paraphrase"}
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metrics = [metrics.accuracy]
    language = None

    def load_dataset(self, split: int):
        return datasets.load_dataset("paws-x", self.language, split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example["sentence1"], "sentence2:", example["sentence2"]]
        #tgt_texts = [self.label2id[str(example["label"])]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
    
    
class Paws_X_TaskDataset_EN(Paws_X_TaskDataset):
    name = "paws_en"
    language = "en"

class Paws_X_TaskDataset_FR(Paws_X_TaskDataset):
    name = "paws_fr"
    language = "fr"
    
class Paws_X_TaskDataset_ES(Paws_X_TaskDataset):
    name = "paws_es"
    language = "es"
    
class Paws_X_TaskDataset_DE(Paws_X_TaskDataset):
    name = "paws_de"
    language = "de"
    
class Paws_X_TaskDataset_ZH(Paws_X_TaskDataset):
    name = "paws_zh"
    language = "zh"
    
class Paws_X_TaskDataset_JA(Paws_X_TaskDataset):
    name = "paws_ja"
    language = "ja"
    
class Paws_X_TaskDataset_KO(Paws_X_TaskDataset):
    name = "paws_ko"
    language = "ko"