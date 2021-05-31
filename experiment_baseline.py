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
__date__ = "30 05 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"

import os
import json
import argparse
import yaml
import numpy as np
import numpy as np
from tqdm import tqdm
from eval_metrics import top_hitrate, jaccard_similarity, cosine_similarity


def exp1_conceptnet_baseline(args):

    with open(args.thesaursus_file, 'r', encoding='utf-8') as loadFile:
        thesaursus = json.load(loadFile)

    # 1. conduct prediction
    # check whether top-k hitrate between synonym+hypernym and antonym, and vice versa
    # omitted weighted HR since weights of ConceptNet doesn't have specific meaning
    result_dict = {
        "HR@1": [],
        "W_HR@1": [],
        "HR@3": [],
        "W_HR@3": [],
        "HR@5": [],
        "W_HR@5": [],
        'jaccard': [],
        "const_cos": [],
        "cos": [],
        "pos_tag": [],
        "question_type": []
    }

    for word, value in thesaursus.items():
        pos_tag_ = value['tag']

        synonyms_ = value.get('synonym')
        antonyms_ = value.get('antonym')
        hypernyms_ = value.get('hypernym')

        antonym_tokens_ = antonyms_['tokens']
        antonym_weights_ = antonyms_['weights']

        synonym_tokens_ = synonyms_['tokens']
        synonym_weights_ = synonyms_['weights']

        # 1. check is_antonym contains synonyms or hypernyms (question type: ask_antonym)
        wrong_answers_ = synonyms_['tokens'] + hypernyms_['tokens']

        top1_res_ = top_hitrate(antonym_tokens_, antonym_weights_, wrong_answers_, top_k=1)
        top3_res_ = top_hitrate(antonym_tokens_, antonym_weights_, wrong_answers_, top_k=3)
        top5_res_ = top_hitrate(antonym_tokens_, antonym_weights_, wrong_answers_, top_k=5)

        jaccard_ = jaccard_similarity(antonym_tokens_, synonym_tokens_)
        const_cos_ = cosine_similarity(antonym_tokens_, synonym_tokens_, antonym_weights_, synonym_weights_,
                                       const_weight=True)
        cos_ = cosine_similarity(antonym_tokens_, synonym_tokens_, antonym_weights_, synonym_weights_,
                                       const_weight=False)
        # remove instances that have no predictions
        if antonym_tokens_:
            # save results
            result_dict['HR@1'].append(top1_res_['hr'])
            result_dict['W_HR@1'].append(top1_res_['w_hr'])
            result_dict['HR@3'].append(top3_res_['hr'])
            result_dict['W_HR@3'].append(top3_res_['w_hr'])
            result_dict['HR@5'].append(top5_res_['hr'])
            result_dict['W_HR@5'].append(top5_res_['w_hr'])
            result_dict['jaccard'].append(jaccard_)
            result_dict['const_cos'].append(const_cos_)
            result_dict['cos'].append(cos_)
            result_dict['question_type'].append('ask_antonym')
            result_dict['pos_tag'].append(pos_tag_)

        # 2. check is_synonym contains antonyms (question type: ask_synonym)
        wrong_answers_ = antonyms_['tokens']

        top1_res_ = top_hitrate(synonym_tokens_, synonym_weights_, wrong_answers_, top_k=1)
        top3_res_ = top_hitrate(synonym_tokens_, synonym_weights_, wrong_answers_, top_k=3)
        top5_res_ = top_hitrate(synonym_tokens_, synonym_weights_, wrong_answers_, top_k=5)

        jaccard_ = jaccard_similarity(synonym_tokens_, antonym_tokens_)
        const_cos_ = cosine_similarity(synonym_tokens_, antonym_tokens_, synonym_weights_, antonym_weights_,
                                       const_weight=True)
        cos_ = cosine_similarity(synonym_tokens_, antonym_tokens_, synonym_weights_, antonym_weights_,
                                       const_weight=False)

        # remove instances that have no predictions
        if synonym_tokens_:
            # save results
            result_dict['HR@1'].append(top1_res_['hr'])
            result_dict['W_HR@1'].append(top1_res_['w_hr'])
            result_dict['HR@3'].append(top3_res_['hr'])
            result_dict['W_HR@3'].append(top3_res_['w_hr'])
            result_dict['HR@5'].append(top5_res_['hr'])
            result_dict['W_HR@5'].append(top5_res_['w_hr'])
            result_dict['jaccard'].append(jaccard_)
            result_dict['const_cos'].append(const_cos_)
            result_dict['cos'].append(cos_)
            result_dict['question_type'].append('ask_synonym')
            result_dict['pos_tag'].append(pos_tag_)

    # 2. calculate statistics of metrics:
    output_metric_dict = {
        "All": {},
        "Ask_Antonym": {},
        "Ask_Synonym": {},
        "Noun": {},
        "Adjective": {},
        "Adverb": {}
    }

    synonym_idx = [i for i, t in enumerate(result_dict['question_type']) if t == 'ask_synonym']
    antonym_idx = [i for i, t in enumerate(result_dict['question_type']) if t == 'ask_antonym']
    noun_idx = [i for i, tag in enumerate(result_dict['pos_tag']) if tag == 'Noun']
    adj_idx = [i for i, tag in enumerate(result_dict['pos_tag']) if tag == 'Adjective']
    adv_idx = [i for i, tag in enumerate(result_dict['pos_tag']) if tag == 'Adverb']

    for key, value in tqdm(result_dict.items(), desc='Summarising statistics...'):
        # only take average of hit rate
        if 'HR' in key or key in ['jaccard', 'const_cos', 'cos']:
            output_metric_dict['All'][f'avg_{key}'] = float(np.mean(value))
            output_metric_dict['Ask_Synonym'][f'avg_{key}'] = float(np.mean([r for i, r in enumerate(result_dict[key])
                                                                       if i in synonym_idx]))
            output_metric_dict['Ask_Antonym'][f'avg_{key}'] = float(np.mean([r for i, r in enumerate(result_dict[key])
                                                                       if i in antonym_idx]))
            output_metric_dict['Noun'][f'avg_{key}'] = float(np.mean([r for i, r in enumerate(result_dict[key])
                                                                       if i in noun_idx]))
            output_metric_dict['Adjective'][f'avg_{key}'] = float(np.mean([r for i, r in enumerate(result_dict[key])
                                                                       if i in adj_idx]))
            output_metric_dict['Adverb'][f'avg_{key}'] = float(np.mean([r for i, r in enumerate(result_dict[key])
                                                                       if i in adv_idx]))

    save_dir = os.path.join(args.save_dir, 'exp1')
    os.makedirs(save_dir, exist_ok=True)
    result_file_name = os.path.join(save_dir, f"baseline-result.yaml")

    with open(result_file_name, 'w') as resultFile:
        yaml.dump(output_metric_dict, resultFile, default_flow_style=False, sort_keys=False)
    print(json.dumps(output_metric_dict, indent="\t"))


def expl2_conceptnet_baseline(args):
    # 1. load partial conceptnet
    with open('./conceptnet_partial.json', 'r', encoding='utf-8') as loadFile:
        kg_dict = json.load(loadFile)

    # 2. load data
    data = []
    with open(os.path.join(args.resource_dir, 'exp2_dataset.jsonl'), 'r', encoding='utf-8') as dataFile:
        for line in dataFile:
            data.append(json.loads(line))

    # 3. do prediction
    for d in tqdm(data):
        word_ = d['word']
        relation_ = d['relation']

        opposite_pred_, opposite_weight_ = kg_dict[word_][relation_]['tokens'], kg_dict[word_][relation_]['weights']
        # predict answer for negated version by using [Pred_object, DistinctFrom/Antonym, [Obj]]
        prediction_, weight_ = list(), list()
        for o_p in opposite_pred_:
            if kg_dict.get(o_p):
                if kg_dict[o_p].get('Antonym'):
                    prediction_ += kg_dict[o_p]['Antonym']['tokens']
                    weight_ += kg_dict[o_p]['Antonym']['weights']
                if kg_dict[o_p].get('DistinctFrom'):
                    prediction_ += kg_dict[o_p]['DistinctFrom']['tokens']
                    weight_ += kg_dict[o_p]['DistinctFrom']['weights']
        merged_ = [(p, w) for p, w in zip(prediction_, weight_)]
        # sort in higher order
        merged_ = sorted(merged_, key=lambda x: x[1], reverse=True)
        prediction_ = [m[0] for m in merged_]
        weight_ = [m[1] for m in merged_]

        d['prediction'] = {
            "tokens": prediction_,
            "scores": weight_
        }

        d['opposite_prediction'] = {
            "tokens": opposite_pred_,
            "scores": opposite_weight_
        }

    # 4. calculate metrics
    result_dict = {
        "HR@1": [],
        "W_HR@1": [],
        "HR@3": [],
        "W_HR@3": [],
        "HR@5": [],
        "W_HR@5": [],
        'jaccard': [],
        "const_cos": [],
        "cos": [],
        "relation": []
    }

    for d in data:
        pred_token_ = d['prediction']['tokens']
        pred_score_ = d['prediction']['scores']
        opposite_pred_token_ = d['opposite_prediction']['tokens']
        opposite_pred_score_ = d['opposite_prediction']['scores']
        wrong_prediction_ = d['wrong_prediction']

        top1_res_ = top_hitrate(pred_token_, pred_score_, wrong_prediction_, top_k=1)
        top3_res_ = top_hitrate(pred_token_, pred_score_, wrong_prediction_, top_k=3)
        top5_res_ = top_hitrate(pred_token_, pred_score_, wrong_prediction_, top_k=5)

        jaccard_ = jaccard_similarity(pred_token_, opposite_pred_token_)
        const_cos_ = cosine_similarity(pred_token_, opposite_pred_token_,
                                       pred_score_, opposite_pred_score_, const_weight=True)
        cos_ = cosine_similarity(pred_token_, opposite_pred_token_,
                                 pred_score_, opposite_pred_score_, const_weight=False)

        # save results
        if pred_token_:
            result_dict['HR@1'].append(top1_res_['hr'])
            result_dict['W_HR@1'].append(top1_res_['w_hr'])
            result_dict['HR@3'].append(top3_res_['hr'])
            result_dict['W_HR@3'].append(top3_res_['w_hr'])
            result_dict['HR@5'].append(top5_res_['hr'])
            result_dict['W_HR@5'].append(top5_res_['w_hr'])
            result_dict['jaccard'].append(jaccard_)
            result_dict['const_cos'].append(const_cos_)
            result_dict['cos'].append(cos_)
            result_dict['relation'].append(d['relation'])

    output_metric_dict = {
        "All": {},
    }
    relations = [d['relation'] for d in data]
    unique_rel = list(set(relations))
    for rel_ in unique_rel:
        output_metric_dict[rel_] = {}

    index_dict = {}
    for rel_ in unique_rel:
        index_dict[rel_] = [i for i, t in enumerate(result_dict['relation']) if t == rel_]

    for key, value in result_dict.items():
        # only take evaluation metrics
        if 'HR' in key or key in ['jaccard', 'const_cos', 'cos']:
            output_metric_dict['All'][f'avg_{key}'] = float(np.mean(value))

            for relation_, index_ in index_dict.items():
                output_metric_dict[relation_][f'avg_{key}'] = float(
                    np.mean([r for i, r in enumerate(result_dict[key])
                             if i in index_]))

    save_dir = os.path.join(args.save_dir, 'exp2')
    os.makedirs(save_dir, exist_ok=True)
    result_file_name = os.path.join(save_dir, f"baseline-result.yaml")

    with open(result_file_name, 'w') as resultFile:
        yaml.dump(output_metric_dict, resultFile, default_flow_style=False, sort_keys=False)
    print(json.dumps(output_metric_dict, indent="\t"))

    pred_file_name = os.path.join(save_dir, f"baseline-pred.yaml")
    with open(pred_file_name, 'w') as resultFile:
        yaml.dump(data, resultFile, default_flow_style=False, sort_keys=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # neccesary
    parser.add_argument('--experiment_type', type=int, default=1, choices=[1, 2],
                        help='experiment number')
    # path
    parser.add_argument('--resource_dir', type=str, default='resources',
                        help='directory path where data are located')
    parser.add_argument('--save_dir', type=str, default='output',
                        help='directory path where results will be saved')
    parser.add_argument('--thesaursus_file', type=str, default='./cn_thesaursus.json',
                        help='path to constructed thesaursus dictionary.')
    parser.add_argument('--partial_kg_file', type=str, default='./conceptnet_partial.json',
                        help="path to constructed partial conceptnet file")
    args = parser.parse_args()

    if args.experiment_type == 1:
        exp1_conceptnet_baseline(args)
    elif args.experiment_type == 2:
        expl2_conceptnet_baseline(args)