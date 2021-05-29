# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        Copyright (c) -2021 - Mtumbuka F. M and Jang M.                                         #
#        All rights reserved.                                                                       #
#                                                                                                   #
#        Redistribution and use in source and binary forms, with or without modification, are       #
#        permitted provided that the following conditions are met:                                  #    
#        1. Redistributions of source code must retain the above copyright notice, this list of     #
#           conditions and the following disclaimer.                                                #
#        2. Redistributions in binary form must reproduce the above copyright notice, this list of  #
#           conditions and the following disclaimer in the documentation and/or other materials     #
#           provided with the distribution.                                                         #
#                                                                                                   #
#        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY      #
#        EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    #
#        MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE #
#        COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,   #
#        EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF         #
#        SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     #
#        HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR   #
#        TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         #
#        SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                               #
#                                                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


__license__ = "BSD-2-Clause"
__version__ = "2021.1"
__date__ = "28 05 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"

import typing
import torch
import os
import json
import argparse
import yaml
import numpy as np
from tqdm import tqdm
from transformers import pipeline


def top_hitrate(pred: typing.List, pred_score: typing.List, target: typing.List, top_k: int = 5):
    """

    Args:
        pred: list of prediction tokens
        pred_score: list of predicted confidence score
        target: wrong prediction token list
        top_k: top-k for calculating hit rate

    Returns: result Dictionary

    """
    pred_ = pred[:top_k]
    score = pred_score[:top_k]

    hit_ = [1 if p in target else 0 for p in pred_]

    hr = sum(hit_) / top_k
    w_hr = np.dot(np.array(hit_), np.array(score)) / sum(score)

    return {"hr": hr, "w_hr": w_hr}


def prediction(model, data: typing.List, batch_size: int = 16):
    """

    Args:
        model: unmasker-model
        data: list of data instance: [{"word": ..., "input_sent": ..., "wrong_prediction": ...}]
        batch_size: inference batch size

    Returns: prediction list, update list of data instance

    """
    mask_token = model.tokenizer.mask_token
    input_sents = [d['input_sent'].replace('Y', mask_token) for d in data]

    preds = list()
    for start in tqdm(range(0, len(input_sents), batch_size), desc='Predicting...'):
        # original
        batch_ = input_sents[start:start + batch_size]
        original_results_ = model(batch_)
        original_preds_ = list()
        for r in original_results_:
            tokens = [r_['token_str'].encode('ascii', 'ignore').decode('utf-8') for r_ in r]
            scores = [r_['score'] for r_ in r]
            original_preds_.append({'tokens': tokens, 'scores': scores})
        preds += original_preds_

    # update prediction to data list
    for i, p in enumerate(preds):
        data[i]['prediction'] = p
    return preds, data


def evaluate(preds: typing.List, data: typing.List):
    """

    Args:
        preds: list of prediction results
        data: list of data instance

    Returns: evaluation metric dictionary

    """
    # 1. record metrics of each instance
    result_dict = {
        "HR@1": [],
        "W_HR@1": [],
        "HR@3": [],
        "W_HR@3": [],
        "HR@5": [],
        "W_HR@5": [],
        "question_type": []
    }

    for i in range(len(data)):
        # set elements
        pred_token_ = preds[i]['tokens']
        pred_score_ = preds[i]['scores']
        wrong_prediction_ = data[i]['wrong_prediction']
        input_sent_ = data[i]['input_sent']

        top1_res_ = top_hitrate(pred_token_, pred_score_, wrong_prediction_, top_k=1)
        top3_res_ = top_hitrate(pred_token_, pred_score_, wrong_prediction_, top_k=3)
        top5_res_ = top_hitrate(pred_token_, pred_score_, wrong_prediction_, top_k=5)

        # save results
        result_dict['HR@1'].append(top1_res_['hr'])
        result_dict['W_HR@1'].append(top1_res_['w_hr'])
        result_dict['HR@3'].append(top3_res_['hr'])
        result_dict['W_HR@3'].append(top3_res_['w_hr'])
        result_dict['HR@5'].append(top5_res_['hr'])
        result_dict['W_HR@5'].append(top5_res_['w_hr'])
        if 'antonym' in input_sent_:
            result_dict['question_type'].append('ask_antonym')
        if 'antonym' in input_sent_:
            result_dict['question_type'].append('ask_synonym')

    # 2. calculate statistics of metrics:

    output_metric_dict = {
        "All": {},
        "Ask_Antonym": {},
        "Ask_Synonym": {}
    }

    synonym_idx = [i for i, t in enumerate(result_dict['question_type']) if t == 'ask_synonym']
    antonym_idx = [i for i, t in enumerate(result_dict['question_type']) if t == 'ask_antonym']

    for key, value in result_dict.items():
        # only take average of hit rate
        if 'HR' in key:
            output_metric_dict['All'][f'avg_{key}'] = float(np.mean(value))
            output_metric_dict['Ask_Synonym'][f'avg_{key}'] = float(np.mean([r for i, r in enumerate(result_dict[key])
                                                                       if i in synonym_idx]))
            output_metric_dict['Ask_Antonym'][f'avg_{key}'] = float(np.mean([r for i, r in enumerate(result_dict[key])
                                                                       if i in antonym_idx]))
    return output_metric_dict


def main(args):
    # 1. Load pretrained model
    print(f"Loading {args.model_type} model...")
    pretrain_model_dict = {
        "electra-large": 'google/electra-large-generator',
        "electra-small": "google/electra-small-generator",
        "bert-base": "bert-base-cased",
        "bert-large": "bert-large-cased",
        "roberta-base": "roberta-base",
        "roberta-large": "roberta-large",
        "albert-base": "albert-base-v2",
        "albert-large": "albert-large-v2"
    }

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = 0
    else:
        device = -1
    unmasker = pipeline('fill-mask', model=pretrain_model_dict[args.model_type], device=device, top_k=args.top_k)

    # 2. Load data
    data = []
    with open(os.path.join(args.resource_dir, 'exp1_dataset.jsonl'), 'r', encoding='utf-8') as loadFile:
        for line in loadFile:
            data.append(json.loads(line))

    # 3. Do prediction
    preds, data = prediction(unmasker, data, batch_size=args.batch_size)

    # 4. Evaluate result
    result = evaluate(preds, data)

    os.makedirs(args.save_dir, exist_ok=True)
    result_file_name = os.path.join(args.save_dir, f"{args.model_type}-result.yaml")
    pred_file_name = os.path.join(args.save_dir, f"{args.model_type}-prediction.json")

    with open(result_file_name, 'w') as resultFile:
        yaml.dump(result, resultFile, default_flow_style=False)

    print(json.dumps(result, indent="\t"))

    with open(pred_file_name, 'w', encoding='utf-8') as predFile:
        for obs in data:
            json.dump(obs, predFile)
            predFile.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--resource_dir', type=str, default='resources',
                        help='directory path where data are located')
    parser.add_argument('--save_dir', type=str, default='output',
                        help='directory path where results will be saved')
    # params
    parser.add_argument('--model_type', type=str, default='bert-base',
                        help='type of pre-trained models for Masked Word Prediction')
    parser.add_argument('--top_k', type=int, default=10,
                        help='top-k predictions for Masked word prediction.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for inference')

    args = parser.parse_args()
    main(args)






