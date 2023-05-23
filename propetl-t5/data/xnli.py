
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

class XNLITaskDataset(AbstractTaskDataset):
    name = "xnli_en"
    label_list = ["0", "1", "2"]
    #label_list = ["entailment", "neutral", "contradiction"]
    #label2id = {"0": "entailment", "1": "neutral", "2" : "contradiction"}
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metrics = [metrics.accuracy]
    language = "en"


    def load_dataset(self, split):
        return datasets.load_dataset('xnli', self.language, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis", example["hypothesis"],]
        tgt_texts = [str(example['label'])]
        #tgt_texts = [self.label2id[str(example['label'])]]
        #print(self.name)
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)




class XNLITaskDataset_EN(XNLITaskDataset):
    name = "xnli_en"
    language = "en"
    
    
class XNLITaskDataset_AR(XNLITaskDataset):
    name = "xnli_ar"
    langauge = "ar"
    
class XNLITaskDataset_BG(XNLITaskDataset):
    name = "xnli_bg"
    langauge = "bg"
    
class XNLITaskDataset_DE(XNLITaskDataset):
    name = "xnli_de"
    langauge = "de"

class XNLITaskDataset_EL(XNLITaskDataset):
    name = "xnli_el"
    langauge = "el"

class XNLITaskDataset_ES(XNLITaskDataset):
    name = "xnli_es"
    langauge = "es"
    
class XNLITaskDataset_FR(XNLITaskDataset):
    name = "xnli_fr"
    langauge = ""
class XNLITaskDataset_HI(XNLITaskDataset):
    name = "xnli_hi"
    langauge = "hi"
class XNLITaskDataset_RU(XNLITaskDataset):
    name = "xnli_ru"
    langauge = "ru"
class XNLITaskDataset_SW(XNLITaskDataset):
    name = "xnli_sw"
    langauge = "sw"
class XNLITaskDataset_TH(XNLITaskDataset):
    name = "xnli_th"
    langauge = "th"
class XNLITaskDataset_TR(XNLITaskDataset):
    name = "xnli_tr"
    langauge = "tr"
class XNLITaskDataset_UR(XNLITaskDataset):
    name = "xnli_ur"
    langauge = "ur"
class XNLITaskDataset_VI(XNLITaskDataset):
    name = "xnli_vi"
    langauge = "vi"
class XNLITaskDataset_ZH(XNLITaskDataset):
    name = "xnli_zh"
    langauge = "zh"



    