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
__date__ = "25 06 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"

import os
import json
import typing
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from src.metrics import jaccard_similarity, cosine_similarity, top_hitrate

ANTONYM_TEMPLATE = ['X is an antonym of Y.', "X is the opposite of Y.", "X is different from Y."]
SYNONUM_TEMPLATE = ['X is a synonym of Y.', "X is another form of Y.", "X is a rephrasing of Y."]

output_dir = 'output/exp1'


def split_same_predictions(pred_list: typing.List[dict]):
    same_outp, diff_outp = [], []
    for pred_ in tqdm(pred_list):
        pred_tokens_ = pred_['prediction']['tokens']
        word_ = pred_['word']
        if word_ in pred_tokens_:
            same_outp.append(pred_)
        else:
            diff_outp.append(pred_)
    outp = {
        "same_pred": same_outp,
        "diff_pred": diff_outp
    }
    return outp


def evaluate_hitrate(
        preds: typing.List[typing.Dict],
        wrong_prediction: typing.List,
) -> typing.Dict:
        """
        Args:
            preds: list of prediction results
            wrong_prediction: list of wrong predictions

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
        }

        for i in tqdm(range(len(preds)), desc='Measuring metrics'):
            # set elements
            pred_token_ = preds[i]['tokens']
            pred_score_ = preds[i]['scores']
            wrong_prediction_ = wrong_prediction[i]

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
        return result_dict


def evaluate_similarity(
        original_preds: typing.List[typing.Dict],
        negated_preds: typing.List[typing.Dict]
    ) -> typing.Dict:

        jaccard, const_cos, cos = list(), list(), list()
        for o, n in zip(original_preds, negated_preds):
            jaccard_ = jaccard_similarity(o['tokens'], n['tokens'])
            const_cos_ = cosine_similarity(o['tokens'], n['tokens'], o['scores'], n['scores'], const_weight=True)
            cos_ = cosine_similarity(o['tokens'], n['tokens'], o['scores'], n['scores'], const_weight=False)

            jaccard.append(jaccard_)
            const_cos.append(const_cos_)
            cos.append(cos_)

        outp = {
            "jaccard": jaccard,
            "const_cos": const_cos,
            "cos": cos
        }
        return outp


def evaluate(pred_list: typing.List[dict], do_summarise: bool = True):
    preds = [p['prediction'] for p in pred_list]
    wrong_prediction = [p['wrong_prediction'] for p in pred_list]
    negated_predictions = [p['opposite_prediction'] for p in pred_list]

    hr_result = evaluate_hitrate(preds, wrong_prediction)
    sim_result = evaluate_similarity(preds, negated_predictions)

    output = {}
    if do_summarise:
        for key, value in hr_result.items():
            output[key] = np.mean(value)

        for key, value in sim_result.items():
            output[key] = np.mean(value)
    else:
        output.update(hr_result)
        output.update(sim_result)
    return output


file_list = [f for f in os.listdir(output_dir) if f.endswith('.jsonl')]
names, outputs = [], {}

for f_name in tqdm(file_list):
    model_name = f_name.split('-prediction')[0]
    # load prediction results
    predicted_result = []
    with open(os.path.join(output_dir, f_name), 'r') as loadFile:
        for line in loadFile:
            predicted_result.append(json.loads(line))
    predicted_result = [p for p in predicted_result if len(p['word']) > 1]

    # divide antonym, synonym questions
    antonym_ques = [p for p in predicted_result if p['template'] in ANTONYM_TEMPLATE]
    synonym_ques = [p for p in predicted_result if p['template'] in SYNONUM_TEMPLATE]

    antonym_splits = split_same_predictions(antonym_ques)
    antonym_same_pred = antonym_splits['same_pred']
    antonym_diff_pred = antonym_splits['diff_pred']
    
    synonym_splits = split_same_predictions(synonym_ques)
    synonym_same_pred = synonym_splits['same_pred']
    synonym_diff_pred = synonym_splits['diff_pred']

    antonym_diff_outp = evaluate(antonym_diff_pred)
    antonym_diff_outp.update({"data_ratio": len(antonym_splits['diff_pred']) / len(antonym_ques)})

    antonym_same_outp = evaluate(antonym_same_pred)
    antonym_same_outp.update({"data_ratio": len(antonym_splits['same_pred']) / len(antonym_ques)})
    
    antonym_outp = evaluate(antonym_ques)
    antonym_outp.update({"data_ratio": 1})
    
    synonym_diff_outp = evaluate(synonym_diff_pred)
    synonym_diff_outp.update({"data_ratio": len(synonym_splits['diff_pred']) / len(synonym_ques)})

    synonym_same_outp = evaluate(synonym_same_pred)
    synonym_same_outp.update({"data_ratio": len(synonym_splits['same_pred']) / len(synonym_ques)})

    synonym_outp = evaluate(synonym_ques)
    synonym_outp.update({"data_ratio": 1})

    # update results
    if not outputs:
        for key in synonym_outp.keys():
            outputs[key] = []
            
    # update synonym diff
    names.append(f"{model_name}-synonym-diff")
    for key in outputs.keys():
        outputs[key].append(synonym_diff_outp[key])

    # update synonym same
    names.append(f"{model_name}-synonym-same")
    for key in outputs.keys():
        outputs[key].append(synonym_same_outp[key])

    # update synonym
    names.append(f"{model_name}-synonym")
    for key in outputs.keys():
        outputs[key].append(synonym_outp[key])

    # update antonym diff
    names.append(f"{model_name}-antonym-diff")
    for key in outputs.keys():
        outputs[key].append(antonym_diff_outp[key])

    # update antonym same
    names.append(f"{model_name}-antonym-same")
    for key in outputs.keys():
        outputs[key].append(antonym_same_outp[key])
        
    # update antonym
    names.append(f"{model_name}-antonym")
    for key in outputs.keys():
        outputs[key].append(antonym_outp[key])


output_df = pd.DataFrame(outputs)
output_df.insert(loc=0, column='models', value=names)
os.makedirs('output/prediction_analysis/', exist_ok=True)
output_df.to_csv('./output/prediction_analysis/same_word_prediction.tsv', sep='\t', index=False)