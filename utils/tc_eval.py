import re
from os import listdir
from os.path import isfile, join, exists
import numpy as np
import string
from zhon.hanzi import punctuation

def _pad_punctuation(text):
  # Pad everything except for: underscores (_), whitespace (\s),
  # numbers (\p{N}), letters (\p{L}) and accent characters (\p{M}).
  text = re.sub(r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ', text)
  # Collapse consecutive whitespace into one space.
  text = re.sub(r'\s+', ' ', text)
  return text


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        text1 = ''.join(ch for ch in text if ch not in exclude) # del the English punctuation...
        text2 = ''.join(ch for ch in text if ch not in punctuation) # del the Chinese punctuation...
        return text2

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def score_string_similarity(str1, str2):
    str1 = normalize_answer(str1)
    str2 = normalize_answer(str2)
    if str1 == str2:
        return 3.0   # Better than perfect token match
    str1 = fix_buggy_characters(replace_punctuation(str1))
    str2 = fix_buggy_characters(replace_punctuation(str2))
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0

def replace_punctuation(str):
    # return str.replace("\"", "").replace("'", "")
    punctuation_string = string.punctuation
    for i in punctuation_string:
        str = str.replace(i, '')

    # del chinese punctuation...
    c_pun_str = punctuation
    for i in c_pun_str:
        str = str.replace(i, '')
    return str


# Temporary fix for bug where {}^<\` characters roundtrip into \u2047 (??) character
def fix_buggy_characters(str):
    return re.sub("[{}^\\\\`\u2047<]", " ", str)



def evaluate(gold_answers,candidates_list, predictions):
    candidates_list = [normalize_answer(x) for x in candidates_list]
    candidates_list = [' '.join(x.split('.')).strip() for x in candidates_list]
    accuracy = []
    for i,(gold_answer, prediction) in enumerate(zip(gold_answers,predictions)):
        # print('compute....')
        gold_answer = gold_answer[0]
        gold_answer = normalize_answer(gold_answer)
        gold_answer = ' '.join(gold_answer.split('.')).strip()
        scores = [score_string_similarity(x, prediction) for x in candidates_list]
        pred_idx = np.argmax(scores)
        if gold_answer in candidates_list:
            gold_idx = candidates_list.index(gold_answer)
        else:
            print("Error! gold_answer is not in candidates-list. you check it!")
            print("gold_answer: %s"%(gold_answer))
            print("candidates_list answers: ",candidates_list)
            break
        acc_flag = 0
        if pred_idx == gold_idx:
            accuracy.append(1)
            acc_flag = 1
        else:
            accuracy.append(0)


        if i<5:
            print('gold_answer: ',gold_answer)
            print('prediction: ', prediction)
            print('acc_flag: ', acc_flag)
            print('candidates_list: ', candidates_list)
            print('scores: ', scores)
    print('total gold_answer: ',len(gold_answers))
    print('total predictions: ', len(predictions))


    acc = 100.0 * sum(accuracy) / len(accuracy)

    return {'acc':acc}


