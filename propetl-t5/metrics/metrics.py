"""Defines different metrics used for evaluation of tasks."""
import functools
import numpy as np
import scipy
import math
import sklearn
from logging import getLogger
from third_party.utils import calculate_rouge, calculate_bleu, lmap
from transformers import EvalPrediction, PreTrainedTokenizer
from typing import Callable, Dict, List, Tuple
from collections import Counter
from transformers import T5Tokenizer
logger = getLogger(__name__)



def __exact_match_score(prediction, ground_truth):
    """_summary_

    Args:
        prediction (_type_): _description_
        ground_truth (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(ground_truth) == len(prediction):
        if all(token1 == token2 for token1, token2 in zip(ground_truth,prediction)):
            return 1
    return 0

def __f1_score(prediction_tokens, ground_truth_tokens):
    """_summary_

    Args:
        prediction (_type_): _description_
        ground_truth (_type_): _description_

    Returns:
        type_: _description_
    """
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def __pos_acc(prediction_tokens, ground_truth_tokens):
    correct = 0
    for p_token, g_token in zip(prediction_tokens, ground_truth_tokens):
        if p_token == g_token: correct += 1
    return correct / len(ground_truth_tokens)
    
    
def POS_acc(predictions, gold_answers):
    """_summary_

    Args:
        predictions (_type_): _description_
        gold_answers (_type_): _description_

    Returns:
        _type_: _description_
    """
    acc  = 0
    for ground_truths, prediction in zip(gold_answers, predictions):
        ground_truths = ground_truths.split(' $$ ')
        prediction = prediction.split(' $$ ')
        acc += __pos_acc(prediction, ground_truths)
    return {"acc":100*acc/len(predictions)}

def NER_f1(predictions, gold_answers):
    """_summary_

    Args:
        predictions (_type_): _description_
        gold_answers (_type_): _description_

    Returns:
        _type_: _description_
    """
    f1  = 0
    for ground_truths, prediction in zip(gold_answers, predictions):
        ground_truths = ground_truths.split(' $$ ')
        prediction = prediction.split(' $$ ')
        f1 += __f1_score(prediction, ground_truths)
    return {"f1":100*f1/len(predictions)}

def QA_f1(predictions, gold_answers):
    """_summary_

    Args:
        predictions (_type_): _description_
        gold_answers (_type_): _description_

    Returns:
        _type_: _description_
    """
    f1  = 0
    for ground_truths, prediction in zip(gold_answers, predictions):
        ground_truths = ground_truths.split(' ')
        prediction = prediction.split(' ')
        f1 += __f1_score(prediction, ground_truths)
    return {"f1":100*f1/len(predictions)}
    
def QA_exactmatch(predictions, gold_answers):
    """_summary_

    Args:
        predictions (_type_): _description_
        gold_answers (_type_): _description_

    Returns:
        _type_: _description_
    """
    exact_match = 0
    for ground_truths, prediction in zip(gold_answers, predictions):
        ground_truths = ground_truths.split(' ')
        prediction = prediction.split(' ')
        exact_match += __exact_match_score(prediction, ground_truths)
    return { "exact_match" : 100*exact_match/len(predictions)}
    
def rouge(predictions, targets) -> dict:
    """Computes rouge score."""
    return calculate_rouge(predictions, targets)


def bleu(predictions, targets) -> dict:
    """Computes bleu score."""
    return calculate_bleu(predictions, targets)


def accuracy(predictions, targets) -> dict:
    """Computes the average accuracy."""
    return {"acc": 100 * ((np.array(predictions) == np.array(targets)).mean())}


def pearson_corrcoef(predictions, targets) -> dict:
    """Computes Pearson correlation coefficient."""
    pearson_corrcoef = 100 * scipy.stats.pearsonr(targets, predictions)[0]

    # Note that if all the predictions will be the same, spearman
    # correlation is nan, to gaurad against this, we check the output
    # and return 0 in this case.
    if math.isnan(pearson_corrcoef):
        pearson_corrcoef = 0
    return {"pearson_corrcoef": pearson_corrcoef}


def spearman_corrcoef(predictions, targets) -> dict:
    """Computes Spearman correlation coefficient."""
    spearman_corrcoef = 100 * scipy.stats.spearmanr(targets, predictions)[0]

    # Note that if all the predictions will be the same, spearman
    # correlation is nan, to gaurad against this, we check the output
    # and return 0 in this case.
    if math.isnan(spearman_corrcoef):
        spearman_corrcoef = 0
    return {"spearman_corrcoef": spearman_corrcoef}


def f1_score_with_invalid(predictions, targets) -> dict:
    """Computes F1 score,  with any prediction != 0 or 1 is counted as incorrect.
    Args:
      targets: list of targets, either 0 or 1
      predictions: list of predictions, any integer value
    Returns:
      F1 score, where any prediction != 0 or 1 is counted as wrong.
    """
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    # Get indices of invalid predictions.
    invalid_idx_mask = np.logical_and(predictions != 0, predictions != 1)
    # For any prediction != 0 or 1, we set the prediction to the opposite of its corresponding target.
    predictions[invalid_idx_mask] = 1 - targets[invalid_idx_mask]
    return {"f1": 100 * sklearn.metrics.f1_score(targets, predictions)}


# TODO: maybe gaurd against invalid values https://stackoverflow.com/questions/56865344/how-do-i-calculate-the-matthews-correlation-coefficient-in-tensorflow
def matthews_corrcoef(predictions, targets) -> dict:
    """Computes the Matthews correlation coefficient."""
    return {"mcc": 100 * sklearn.metrics.matthews_corrcoef(targets, predictions)}


def build_compute_metrics_fn(task_names: List[str],
                             tokenizer: PreTrainedTokenizer) -> Callable[[EvalPrediction], Dict]:
    """Builds a dictionary from each task to the task metric."""

    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        try:
            label_ids = np.where(pred.label_ids != -100, pred.label_ids, tokenizer.pad_token_id)
            label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            pred_str = lmap(str.strip, pred_str)
            label_str = lmap(str.strip, label_str)
        except e :
            print(label_ids[0])
            raise e
        return pred_str, label_str

    def compute_metrics(pred: EvalPrediction, metrics, post_processor=None) -> Dict:
        pred_str, label_str = decode_pred(pred)

        # Applies task post-processor.
        if post_processor is not None:
            pred_str = [post_processor(pred) for pred in pred_str]
            label_str = [post_processor(label) for label in label_str]

        eval_results = {}
        for metric in metrics:
            eval_results.update(metric(pred_str, label_str))
            if metric.__name__ in ['bleu', 'rouge']:
                gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
                eval_results.update({"gen_len": gen_len})
        return eval_results

    def tasks_metrics(task) -> Dict:
        from data.tasks import TASK_MAPPING
        from data.postprocessors import get_post_processor
        return functools.partial(compute_metrics, metrics=TASK_MAPPING[task].metrics,
                                 post_processor=get_post_processor(task))

    return {task: tasks_metrics(task) for task in task_names}
