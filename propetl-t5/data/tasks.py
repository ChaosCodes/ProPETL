"""Implements different tasks and defines the processors to convert each dataset
to a sequence to sequence format."""
from collections import OrderedDict

import abc
import datasets
import functools
import logging
import numpy as np
import torch
from metrics import metrics
from typing import Callable, Dict, Mapping, List
from .mlqa import *
from .paws_x import *
from .xnli import *
from .wiki_ann import *
from .abstract_task import AbstractTaskDataset
from .utils import round_stsb_target, compute_task_max_decoding_length
from .pos import *
from .mt import *
logger = logging.getLogger(__name__)


class IMDBTaskDataset(AbstractTaskDataset):
    name = "imdb"
    split_to_data_split = {"train": "train",
                           "validation": "test",
                           "test": "test"}
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example["text"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SickTaskDataset(AbstractTaskDataset):
    name = "sick"
    label_list = ["0", "1", "2"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    label_to_target = {"ENTAILMENT": 0, "CONTRADICTION": 2, "NEUTRAL": 1}
    metrics = [metrics.accuracy]

    def load_dataset(self, split: int):
        return datasets.load_dataset("csv", data_files={split: f"sick/{split}_clean.csv"})[split]

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"], "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(self.label_to_target[example["label"]])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class PawsTaskDataset(AbstractTaskDataset):
    name = "paws"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "test"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.name, 'labeled_final', split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example["sentence1"], "sentence2:", example["sentence2"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)
    
    



class SuperGLUEBoolQTaskDataset(AbstractTaskDataset):
    name = "superglue-boolq"
    label_list = ['0', '1']
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'boolq', split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"], "passage:", example["passage"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUERTETaskDataset(AbstractTaskDataset):
    name = "superglue-rte"
    label_list = ['0', '1']
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'rte', split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"],
                     "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SuperGLUECBTaskDataset(AbstractTaskDataset):
    name = "superglue-cb"
    label_list = ['0', '1', '2']
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset('super_glue', 'cb', split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"], "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SNLITaskDataset(AbstractTaskDataset):
    name = "snli"
    label_list = ["0", "1", "2"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example["premise"], "hypothesis:", example["hypothesis"]]
        tgt_texts = [str(example["label"])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class IWSLT2017RONL(AbstractTaskDataset):
    name = "iwslt2017-ro-nl"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-nl"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("iwslt2017", 'iwslt2017-ro-nl',
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["ro"]]
        tgt_texts = [example['translation']["nl"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate Romanian to Dutch")


class IWSLT2017ENNL(AbstractTaskDataset):
    name = "iwslt2017-en-nl"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"en-nl"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("iwslt2017", 'iwslt2017-en-nl',
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = [example['translation']["en"]]
        tgt_texts = [example['translation']["nl"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix="Translate English to Dutch")


class WMT16ENROTaskDataset(AbstractTaskDataset):
    name = "wmt16-en-ro"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair,
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["Translate English to Romanian:", example['translation']["en"]]
        tgt_texts = [example['translation']["ro"]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix,
                                   prefix=None)


class WMT16ROENTaskDataset(AbstractTaskDataset):
    name = "wmt16-ro-en"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"ro-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair,
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["Translate Romanian to English:",example['translation']["ro"]]
        tgt_texts = [example['translation']["en"]]
        return self.seq2seq_format( src_texts, tgt_texts, add_prefix)


class WMT16ENCSTaskDataset(AbstractTaskDataset):
    name = "wmt16-en-cs"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"cs-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair,
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["Translate English to Czech:", example['translation']["en"]]
        tgt_texts = [example['translation']["cs"]]
        return self.seq2seq_format( src_texts, tgt_texts, add_prefix,
                                   prefix=None)
        
        
        


class WMT16ENFITaskDataset(AbstractTaskDataset):
    name = "wmt16-en-fi"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"fi-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt16", self.pair,
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["Translate English to Finnish:", example['translation']["en"]]
        tgt_texts = [example['translation']["fi"]]
        return self.seq2seq_format( src_texts, tgt_texts, add_prefix,
                                   prefix=None)


class WMT14HIENTaskDataset(AbstractTaskDataset):
    name = "wmt14-hi-en"
    task_specific_config = {'max_length': 300, 'num_beams': 4}
    pair = f"hi-en"
    metrics = [metrics.bleu]

    def load_dataset(self, split):
        return datasets.load_dataset("wmt14", self.pair,
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["Translate English to Hindi:", example['translation']["en"]]
        tgt_texts = [example['translation']["hi"]]
        return self.seq2seq_format( src_texts, tgt_texts, add_prefix,
                                   prefix=None)


class TRECTaskDataset(AbstractTaskDataset):
    name = "trec"
    label_list = ["0", "1", "2", "3", "4", "5"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "test",
                           "test": "test"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset("trec", split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [str(example['label-coarse'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class YelpPolarityTaskDataset(AbstractTaskDataset):
    name = "yelp_polarity"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "test",
                           "test": "test"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset("yelp_polarity",
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['text']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class ScitailTaskDataset(AbstractTaskDataset):
    name = "scitail"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    label_map = {"entailment": 0, "neutral": 1}

    def map_label(self, label):
        return self.label_map[label]

    def load_dataset(self, split):
        return datasets.load_dataset("scitail", "snli_format",
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'], "sentence2:", example["sentence2"]]
        # To increase the transfer performance, we modified the targets to be similar to other datasets.
        tgt_texts = [str(self.map_label(example['gold_label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MRPCTaskDataset(AbstractTaskDataset):
    name = "mrpc"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.f1_score_with_invalid, metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mrpc',
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class COLATaskDataset(AbstractTaskDataset):
    name = "cola"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.matthews_corrcoef]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'cola',
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SST2TaskDataset(AbstractTaskDataset):
    name = "sst2"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'sst2',
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example['sentence']]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class STSBTaskDataset(AbstractTaskDataset):
    name = "stsb"
    label_list = [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'stsb',
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(round_stsb_target(example['label']))]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QQPTaskDataset(AbstractTaskDataset):
    name = "qqp"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.f1_score_with_invalid, metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qqp',
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question1:", example['question1'],
                     "question2:", example["question2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class MNLITaskDataset(AbstractTaskDataset):
    name = "mnli"
    label_list = ["0", "1", "2"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metrics = [metrics.accuracy]

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mnli', split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["premise:", example['premise'],
                     "hypothesis", example["hypothesis"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class QNLITaskDataset(AbstractTaskDataset):
    name = "qnli"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qnli', split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example['question'],
                     "sentence:", example["sentence"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class RTETaskDataset(AbstractTaskDataset):
    name = "rte"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'rte',
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class WNLITaskDataset(AbstractTaskDataset):
    name = "wnli"
    label_list = ["0", "1"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'wnli', split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence1:", example['sentence1'],
                     "sentence2:", example["sentence2"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class SocialIQaTaskDataset(AbstractTaskDataset):
    name = "social_i_qa"
    label_map = {"1": "0", "2": "1", "3": "2"}
    label_list = ["0", "1", "2"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"],
                     "context:", example["context"],
                     "answerA:", example["answerA"],
                     "answerB:", example["answerB"],
                     "answerC:", example["answerC"]]
        tgt_texts = [self.label_map[example['label'].rstrip()]]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class CosmosQaTaskDataset(AbstractTaskDataset):
    name = "cosmos_qa"
    label_list = ["0", "1", "2", "3"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"],
                     "context:", example["context"],
                     "answer0:", example["answer0"],
                     "answer1:", example["answer1"],
                     "answer2:", example["answer2"],
                     "answer3:", example["answer3"]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class WinograndeTaskDataset(AbstractTaskDataset):
    name = "winogrande"
    label_list = ["1", "2"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def load_dataset(self, split):
        return datasets.load_dataset('winogrande', 'winogrande_l',
                                     split=split, )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["sentence:", example["sentence"],
                     "option1:", example["option1"],
                     "option2:", example["option2"]]
        tgt_texts = [str(example['answer'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class HellaSwagTaskDataset(AbstractTaskDataset):
    name = "hellaswag"
    label_list = ["0", "1", "2", "3"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["ctx:", example["ctx"],
                     "ending0:", example["endings"][0],
                     "ending1:", example["endings"][1],
                     "ending2:", example["endings"][2],
                     "ending3:", example["endings"][3]]
        tgt_texts = [str(example['label'])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)


class CommonsenseQaTaskDataset(AbstractTaskDataset):
    name = "commonsense_qa"
    label_map = {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4"}
    label_list = ["0", "1", "2", "3", "4"]  # ["A", "B", "C", "D", "E"]
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["question:", example["question"],
                     "A:", example["choices"]["text"][0],
                     "B:", example["choices"]["text"][1],
                     "C:", example["choices"]["text"][2],
                     "D:", example["choices"]["text"][3],
                     "E:", example["choices"]["text"][4]]
        tgt_texts = [str(self.label_map[example['answerKey']])]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

    
class XSum(AbstractTaskDataset):
    name = "xsum"
    task_specific_config = {'max_length': 64, 'num_beams': 6}
    metrics = [metrics.rouge]

    def load_dataset(self, split):
        return datasets.load_dataset("xsum", split=split )

    def preprocessor(self, example, add_prefix=True):
        src_texts = ["summarize:", example['document']]
        tgt_texts = [example['summary']]
        return self.seq2seq_format(src_texts, tgt_texts, add_prefix)

    
TASK_MAPPING = OrderedDict([
    ('superglue-boolq', SuperGLUEBoolQTaskDataset),
    ('superglue-cb', SuperGLUECBTaskDataset),
    ('superglue-rte', SuperGLUERTETaskDataset),
    ('paws', PawsTaskDataset),
    ('imdb', IMDBTaskDataset),
    ('snli', SNLITaskDataset),
    ('scitail', ScitailTaskDataset),
    ('mrpc', MRPCTaskDataset),
    ('trec', TRECTaskDataset),
    ('yelp_polarity', YelpPolarityTaskDataset),
    ('wmt16-ro-en', WMT16ROENTaskDataset),
    ('wmt15_fr-en', WMT15_FR_EN_TaskDataset),
    ('wmt18_cs-en', WMT18_CS_EN_TaskDataset),
    ('wmt18_de-en', WMT18_DE_EN_TaskDataset),
    ('wmt18_fi-en', WMT18_FI_EN_TaskDataset),
    ('wmt14_hi-en', WMT14HIENTaskDataset),
    ('wmt16-en-ro', WMT16ENROTaskDataset),
    ('wmt16_en-cs', WMT16ENCSTaskDataset),
    ('wmt16-en-fi', WMT16ENFITaskDataset),
    ('iwslt2017-ro-nl', IWSLT2017RONL),
    ('iwslt2017-en-nl', IWSLT2017ENNL),
    ('cola', COLATaskDataset),
    ('sst2', SST2TaskDataset),
    ('stsb', STSBTaskDataset),
    ('qqp', QQPTaskDataset),
    ('mnli', MNLITaskDataset),
    ('qnli', QNLITaskDataset),
    ('rte', RTETaskDataset),
    ('wnli', WNLITaskDataset),
    ('social_i_qa', SocialIQaTaskDataset),
    ('cosmos_qa', CosmosQaTaskDataset),
    ('winogrande', WinograndeTaskDataset),
    ('hellaswag', HellaSwagTaskDataset),
    ('commonsense_qa', CommonsenseQaTaskDataset),
    ('sick', SickTaskDataset),
    ('xnli_ar', XNLITaskDataset_AR),
    ('xnli_bg', XNLITaskDataset_BG),
    ('xnli_de', XNLITaskDataset_DE),
    ('xnli_el', XNLITaskDataset_EL),
    ('xnli_en', XNLITaskDataset_EN),
    ('xnli_es', XNLITaskDataset_ES),
    ('xnli_fr', XNLITaskDataset_FR),
    ('xnli_hi', XNLITaskDataset_HI),
    ('xnli_ru', XNLITaskDataset_RU),
    ('xnli_sw', XNLITaskDataset_SW),
    ('xnli_th', XNLITaskDataset_TH),
    ('xnli_tr', XNLITaskDataset_TR),
    ('xnli_ur', XNLITaskDataset_UR),
    ('xnli_vi', XNLITaskDataset_VI),
    ('xnli_zh', XNLITaskDataset_ZH),
    ('paws_en', Paws_X_TaskDataset_EN),
    ('paws_fr', Paws_X_TaskDataset_FR),
    ('paws_es', Paws_X_TaskDataset_ES),
    ('paws_de', Paws_X_TaskDataset_DE),
    ('paws_zh', Paws_X_TaskDataset_ZH),
    ('paws_ja', Paws_X_TaskDataset_JA),
    ('paws_ko', Paws_X_TaskDataset_KO),
    ("mlqa_en", MLQA_EN),
    ("mlqa_de", MLQA_DE),
    ("mlqa_ar", MLQA_AR),
    ("mlqa_es", MLQA_ES),
    ("mlqa_vi", MLQA_VI),
    ("mlqa_hi", MLQA_HI),
    ("mlqa_zh", MLQA_ZH),
    ("wikiann_en", WikiAnn_EN),
    ("wikiann_ar", WikiAnn_AR),
    ("wikiann_br", WikiAnn_BR),
    ("wikiann_zh", WikiAnn_ZH),
    ("wikiann_is", WikiAnn_IS),
    ("wikiann_kk", WikiAnn_KK),
    ("wikiann_ta", WikiAnn_TA),
    ("wikiann_tr", WikiAnn_TR),
    ("wikiann_yo", WikiAnn_YO),
    ("wikiann_fo", WikiAnn_FO),
    ("wikiann_gn", WikiAnn_GN),
    ("wikiann_hsb", WikiAnn_YO),
    ("wikiann_mt", WikiAnn_MT),
    ("wikiann_sa", WikiAnn_SA),
    ("wikiann_ug", WikiAnn_UG),
    ("wikiann_yue", WikiAnn_YUE),
    
    
    ("ud_en", UD_EN),
    ("ud_ar", UD_AR),
    ("ud_br", UD_BR),
    ("ud_zh", UD_ZH),
    ("ud_is", UD_IS),
    ("ud_kk", UD_KK),
    ("ud_ta", UD_TA),
    ("ud_tr", UD_TR),
    ("ud_yo", UD_YO),
    ("ud_fo", UD_FO),
    ("ud_gn", UD_GN),
    ("ud_hsb", UD_HSB),
    ("ud_mt", UD_MT),
    ("ud_sa", UD_SA),
    ("ud_ug", UD_UG),
    ("ud_yue", UD_YUE),
    ("xsum",  XSum)
    ],
    
)


class AutoTask:
    @classmethod
    def get(self, task_name, seed=42):
        if task_name in TASK_MAPPING:
            return TASK_MAPPING[task_name](seed)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )
