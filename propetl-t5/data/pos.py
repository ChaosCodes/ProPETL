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




class UD_TaskDataset(AbstractTaskDataset):
    name = "ud_en"
    task_specific_config = {'max_length': 256}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metrics = [metrics.POS_acc]
    language = "en_ewt"

    def load_dataset(self, split: int):
        return datasets.load_dataset("universal_dependencies", self.language, split=split)

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["tag part-of-speech:", " ".join(example["tokens"])]
        tgt_texts= [] 
        for pos_id, token in zip(example['upos'], example["tokens"]):
            tgt_texts.append(token+": "+str(pos_id))
        tgt_texts = [" $$ ".join(tgt_texts)]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
    
class UD_EN(UD_TaskDataset):
    name = "ud_en"
    language = "en_ewt"
    
class UD_AR(UD_TaskDataset):
    name = "ud_ar"
    language = "ar_padt"

class UD_BR(UD_TaskDataset):
    name = "ud_br"
    language = "br_keb"
    
class UD_ZH(UD_TaskDataset):
    name = "ud_zh"
    language = "zh_gsd"
    
class UD_IS(UD_TaskDataset):
    name = "ud_is"
    language = "is_pud"

class UD_KK(UD_TaskDataset):
    name = "ud_kk"
    language = "kk_ktb"
    
class UD_TA(UD_TaskDataset):
    name = "ud_ta"
    language = "ta_ttb"
    
class UD_TR(UD_TaskDataset):
    name = "ud_tr"
    language = "tr_imst"
    
class UD_YO(UD_TaskDataset):
    name = "ud_yo"
    language = "yo_ytb"
    
class UD_FO(UD_TaskDataset):
    name = "ud_fo"
    language = "fo_oft"
    
class UD_GN(UD_TaskDataset):
    name = "ud_gn"
    language = "gun_thomas"
    
class UD_HSB(UD_TaskDataset):
    name = "ud_hsb"
    language = "hsb_ufal"
    
class UD_MT(UD_TaskDataset):
    name = "ud_mt"
    language = "mt_mudt"
    
class UD_SA(UD_TaskDataset):
    name = "ud_sa"
    language = "sa_ufal"
    
class UD_UG(UD_TaskDataset):
    name = "ud_ug"
    language = "ug_udt"
    
class UD_YUE(UD_TaskDataset):
    name = "ud_yue"
    language = "yue_hk"
    
