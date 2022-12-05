## SQuAD evaluation script. Modifed slightly for this notebook

from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)



def evaluate_oneanswer(gold_answers1,predictions):
    # print('gold_answers1: ',gold_answers1)
    # print('predictions: ', predictions)
    f1 = exact_match = total = 0

    for ground_truths1, prediction in zip(gold_answers1,predictions):
        total += 1
        exact_match11 = metric_max_over_ground_truths(
            exact_match_score, prediction, [ground_truths1])
        f111 = metric_max_over_ground_truths(
            f1_score, prediction, [ground_truths1])
        exact_match += exact_match11
        f1 += f111
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


def evaluate(gold_answers1,gold_answers, predictions):
    f1 = exact_match = total = 0

    for ground_truths1, ground_truths, prediction in zip(gold_answers1,gold_answers, predictions):
        total += 1
        exact_match1 = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f11 = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        # exact_match11 = metric_max_over_ground_truths(
        #     exact_match_score, prediction, ground_truths1)
        # f111 = metric_max_over_ground_truths(
        #     f1_score, prediction, ground_truths1)

        if total<10:
            print('multi-ans: ',ground_truths)
            print('prediction: ', prediction)
            print('f1:', f11)
            print('em:', exact_match1)

        exact_match += exact_match1
        f1 += f11
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    print('f1: ',f1)
    print('exact_match: ',exact_match)

    return {'exact_match': exact_match, 'f1': f1}


