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

class WikiAnnTaskDataset(AbstractTaskDataset):
    name = "xnli_en"
    task_specific_config = {'max_length': 64}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metrics = [metrics.NER_f1]
    language = "en"


    def load_dataset(self, split):
        return datasets.load_dataset('wikiann', self.language, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["tag entity:"," ".join(example["tokens"])]
        tgt_texts = [' $$ '.join(example['spans'])]
        #print(self.name)
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
    
class WikiAnn_EN(WikiAnnTaskDataset):
    name = "wikiann_en"
    language = "en"
    
class WikiAnn_AR(WikiAnnTaskDataset):
    name = "wikiann_ar"
    language = "ar"
    
class WikiAnn_BR(WikiAnnTaskDataset):
    name = "wikiann_br"
    language = "br"
class WikiAnn_ZH(WikiAnnTaskDataset):
    name = "wikiann_zh"
    language = "zh"
class WikiAnn_IS(WikiAnnTaskDataset):
    name = "wikiann_is"
    language = "is"
class WikiAnn_KK(WikiAnnTaskDataset):
    name = "wikiann_kk"
    language = "kk"
class WikiAnn_TA(WikiAnnTaskDataset):
    name = "wikiann_ta"
    language = "ta"
class WikiAnn_TR(WikiAnnTaskDataset):
    name = "wikiann_tr"
    language = "tr"
class WikiAnn_YO(WikiAnnTaskDataset):
    name = "wikiann_yo"
    language = "yo"
class WikiAnn_FO(WikiAnnTaskDataset):
    name = "wikiann_fo"
    language = "fo"
class WikiAnn_GN(WikiAnnTaskDataset):
    name = "wikiann_gn"
    language = "gn"
class WikiAnn_HSB(WikiAnnTaskDataset):
    name = "wikiann_hsb"
    language = "hsb"
class WikiAnn_MT(WikiAnnTaskDataset):
    name = "wikiann_mt"
    language = "mt"
class WikiAnn_SA(WikiAnnTaskDataset):
    name = "wikiann_sa"
    language = "sa"
class WikiAnn_UG(WikiAnnTaskDataset):
    name = "wikiann_ug"
    language = "ug"
class WikiAnn_YUE(WikiAnnTaskDataset):
    name = "wikiann_yue"
    language = "zh-yue"
    