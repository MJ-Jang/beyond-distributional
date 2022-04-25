# -*- coding: utf-8 -*-

import json

import yaml
import os
import pandas as pd
import numpy as np

model_list = ['bert-base', 'albert-base', 'roberta-base', 'electra-small',
              'bert-large', 'albert-large', 'roberta-large', 'electra-large',
              "meaning_matching-bert-large", "meaning_matching-roberta-large"]

dir_path_list = ['exp1', 'exp2']
dir_path = 'output/exp1'

all_output_dict = {
    "model_name": model_list,
    "top1_hit": [],
    "top3_hit": []
}

for m in model_list:
    # load result

    # if m.startswith('meaning'):
    #     file_name = f"{m}-prediction.jsonl"
    # else:
    #     file_name = f"{m}-prediction.json"
    file_name = f"{m}-prediction.jsonl"
    result = []
    with open(os.path.join(dir_path, file_name), 'r') as loadFile:
        for line in loadFile:
            result.append(json.loads(line))

    top1_hit, top3_hit = [], []
    for res_ in result:
        answers = res_['wrong_prediction']
        if res_['word'] in answers:
            answers.remove(res_['word'])

        if res_['opposite_prediction']['tokens'][0] in answers:
            top1_hit.append(1)
        else:
            top1_hit.append(0)

        intersect_ = set(res_['opposite_prediction']['tokens'][:3]).intersection(set(answers))
        if len(intersect_) > 0:
            top3_hit.append(1)
        else:
            top3_hit.append(0)
    all_output_dict['top1_hit'].append(np.mean(top1_hit))
    all_output_dict['top3_hit'].append(np.mean(top3_hit))

pd.DataFrame(all_output_dict)


# save model
# import os
# import torch
# from transformers import BertTokenizerFast, BertForSequenceClassification
# bert_model_path = os.path.join(os.getcwd(), '../../meaning_matching-bert-large-n_neg10.ckpt')
#
# clf = BertForSequenceClassification.from_pretrained('bert-large-cased')
# clf.load_state_dict(torch.load(bert_model_path))
#
# bert = clf.bert
# tokenizer = BertTokenizerFast.from_pretrained('bert-large-cased')
#
# bert.save_pretrained(os.path.join(os.getcwd(), '../meaning-match-bert-large'))
# tokenizer.save_pretrained(os.path.join(os.getcwd(), '../meaning-match-bert-large'))
#
# from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
# roberta_model_path = os.path.join(os.getcwd(), '../../meaning_matching-roberta-large-n_neg10.ckpt')
#
# clf = RobertaForSequenceClassification.from_pretrained('roberta-large')
# clf.load_state_dict(torch.load(roberta_model_path))
#
# roberta = clf.roberta
# tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')
#
# roberta.save_pretrained(os.path.join(os.getcwd(), '../meaning-match-roberta-large'))
# tokenizer.save_pretrained(os.path.join(os.getcwd(), '../meaning-match-roberta-large'))