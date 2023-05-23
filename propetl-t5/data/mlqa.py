
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

class MLQA_EN(AbstractTaskDataset):
    name = "mlqa_en"
    task_specific_config = {'max_length': 10}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metrics = [metrics.QA_exactmatch, metrics.QA_f1]
    subset = "mlqa.en.en"

    def load_dataset(self, split: int):
        if "train" in split or "validation" in split:
            dataset = datasets.load_dataset("squad", split=split)
        else:
            dataset = datasets.load_dataset("mlqa", self.subset, split=split)
        return dataset
    def preprocessor(self, example, add_prefix=True):
        src_texts = [ "question:", example["question"], "context:", example["context"]]
        tgt_texts = [example["answers"]["text"][0]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
    
class MLQA_DE(AbstractTaskDataset):
    name = "mlqa_de"
    task_specific_config = {'max_length': 10}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metrics = [metrics.QA_exactmatch, metrics.QA_f1]
    translate_train_subset = "mlqa-translate-train.de"
    test_subset = "mlqa.de.de"
    def load_dataset(self, split: int):
        if "train" in split  or  "validation" in split:
            dataset = datasets.load_dataset("mlqa", self.translate_train_subset, split=split)
        else:
            dataset =datasets.load_dataset("mlqa", self.test_subset, split=split)
        return dataset
    def preprocessor(self, example, add_prefix=True):
        src_texts = [ "question answering:", "question:", example["question"], "context:", example["context"], "answer:"]

        tgt_texts = [str(example["answers"]["text"][0])] 
  
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MLQA_AR(MLQA_DE):
    name = "mlqa_ar"
    translate_train_subset = "mlqa-translate-train.ar"
    test_subset = "mlqa.ar.ar"
    
    
class MLQA_ES(MLQA_DE):
    name = "mlqa_es"
    translate_train_subset = "mlqa-translate-train.es"
    test_subset = "mlqa.es.es"
    
class MLQA_HI(MLQA_DE):
    name = "mlqa_hi"
    translate_train_subset = "mlqa-translate-train.hi"
    test_subset = "mlqa.hi.hi"
    
class MLQA_VI(MLQA_DE):
    name = "mlqa_vi"
    translate_train_subset = "mlqa-translate-train.vi"
    test_subset = "mlqa.vi.vi"
    
class MLQA_ZH(MLQA_DE):
    name = "mlqa_zh"
    translate_train_subset = "mlqa-translate-train.zh"
    test_subset = "mlqa.zh.zh"
    
    