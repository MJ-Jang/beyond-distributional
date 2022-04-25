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
import sys
from tqdm import tqdm
from transformers import pipeline, BertForMaskedLM, RobertaForMaskedLM, AutoTokenizer
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from metrics import jaccard_similarity, cosine_similarity, top_hitrate


PWD = os.path.dirname(os.path.abspath(__file__))


def load_plm_state_dict(file_name, plm_name):
    aa = torch.load(file_name)
    new_dict = {}
    for key in aa.keys():
        if key.startswith(plm_name):
            if key.startswith(f'{plm_name}.pooler'):
                continue
            new_dict[key.replace(f"{plm_name}.", "")] = aa[key]
    return new_dict


class ExperimentOperator:

    def __init__(
            self,
            args: argparse.Namespace
    ):
        print(f"Loading {args.model_type} model...")
        pretrain_model_dict = {
            "electra-large": 'google/electra-large-generator',
            "electra-base": "google/electra-base-generator",
            "bert-base": "bert-base-cased",
            "bert-large": "bert-large-cased",
            "roberta-base": "roberta-base",
            "roberta-large": "roberta-large",
            "albert-base": "albert-base-v2",
            "albert-large": "albert-large-v2",
            "meaning_matching-bert-base": "meaning_matching-bert-base-n_neg10",
            "meaning_matching-roberta-base": "meaning_matching-roberta-base-n_neg10",
            "meaning_matching-bert-large": "meaning_matching-bert-large-n_neg10",
            "meaning_matching-roberta-large": "meaning_matching-roberta-large-n_neg10"
        }

        use_gpu = torch.cuda.is_available()
        device = 0 if use_gpu else -1

        if args.model_type != 'baseline' and not args.model_type.startswith('meaning_matching'):
            self.unmasker = pipeline(
                'fill-mask',
                model=pretrain_model_dict[args.model_type],
                device=device,
                top_k=args.top_k
            )
            self.mask_token = self.unmasker.tokenizer.mask_token
        elif args.model_type.startswith('meaning_matching'):
            file_path = os.path.join(PWD, "../mm_experiment/model_binary/",
                                     f"{pretrain_model_dict[args.model_type]}.ckpt")
            if '-bert-' in args.model_type:
                normalized_model_name = args.model_type.replace("meaning_matching-", "")
                model = BertForMaskedLM.from_pretrained(pretrain_model_dict[normalized_model_name])
                tokenizer = AutoTokenizer.from_pretrained(pretrain_model_dict[normalized_model_name])
                model.bert.load_state_dict(load_plm_state_dict(file_path, 'bert'))
            elif '-roberta-' in args.model_type:
                normalized_model_name = args.model_type.replace("meaning_matching-", "")
                model = RobertaForMaskedLM.from_pretrained(pretrain_model_dict[normalized_model_name])
                tokenizer = AutoTokenizer.from_pretrained(pretrain_model_dict[normalized_model_name])
                model.roberta.load_state_dict(load_plm_state_dict(file_path, 'roberta'))
            else:
                raise NotImplementedError
            self.unmasker = pipeline(
                'fill-mask',
                model=model,
                tokenizer=tokenizer,
                device=device,
                top_k=args.top_k
            )
            self.mask_token = self.unmasker.tokenizer.mask_token

        else:
            self.unmasker = None
            self.mask_token = None

        self.top_k = args.top_k
        self.batch_size = args.batch_size

    def __call__(
            self,
            experiment_type: int
    ):
        # 1 Load data
        data = self.load_data(args.resource_dir, experiment_type=experiment_type)

        if experiment_type == 1:
            input_sents = [d['input_sent'].replace('Y', self.mask_token) for d in data]
            opposite_sents = [d['opposite_sent'].replace('Y', self.mask_token) for d in data]
        elif experiment_type == 2:
            input_sents = [d['input_sent'].replace('[MASK]', self.mask_token) for d in data]
            opposite_sents = [d['opposite_sent'].replace('[MASK]', self.mask_token) for d in data]
        else:
            raise NotImplementedError

        wrong_predictions = [d['wrong_prediction'] for d in data]
        pos_tags = [d['pos_tag'] for d in data] if experiment_type == 1 else None
        templates = [d['template'] for d in data] if experiment_type == 1 else None
        relations = [d['relation'] for d in data] if experiment_type == 2 else None

        # 2. Inference
        # for input sents
        input_sent_preds = self.prediction(input_sents, batch_size=self.batch_size)
        # for opposite meaning sents
        opposite_sent_preds = self.prediction(opposite_sents, batch_size=self.batch_size)

        # 3. Evaluate result
        result_hitrate = self.evaluate_hitrate(input_sent_preds, wrong_prediction=wrong_predictions)
        result_similarity = self.evaluate_similarity(input_sent_preds, negated_preds=opposite_sent_preds)

        if experiment_type == 1:
            summary_output = self.summarise_exp1(
                result_hitrate,
                result_similarity,
                input_sents=input_sents,
                pos_tags=pos_tags,
                templates=templates
            )
            save_dir = os.path.join(args.save_dir, 'exp1')

        elif experiment_type == 2:
            summary_output = self.summarise_exp2(
                result_hitrate,
                result_similarity,
                input_sents=input_sents,
                relations=relations
            )
            save_dir = os.path.join(args.save_dir, 'exp2')
        else:
            raise NotImplementedError

        # 4. Update results
        for i, (ip_, op_) in enumerate(zip(input_sent_preds, opposite_sent_preds)):
            data[i]["prediction"] = ip_
            data[i]["opposite_prediction"] = op_

        os.makedirs(save_dir, exist_ok=True)
        result_file_name = os.path.join(save_dir, f"{args.model_type}-result.yaml")
        pred_file_name = os.path.join(save_dir, f"{args.model_type}-prediction.jsonl")

        with open(result_file_name, 'w') as resultFile:
            yaml.dump(summary_output, resultFile, default_flow_style=False, sort_keys=False)

        print(json.dumps(summary_output, indent="\t"))

        with open(pred_file_name, 'w', encoding='utf-8') as predFile:
            for obs in data:
                json.dump(obs, predFile)
                predFile.write("\n")

    def prediction(self, input_sents: typing.List, batch_size: int = 16):
        """

        Args:
            model: unmasker-model
            input_sents: list of data instance: [{"word": ..., "input_sent": ..., "wrong_prediction": ...}]
            batch_size: inference batch size

        Returns: prediction list, update list of data instance

        """
        preds = list()
        for start in tqdm(range(0, len(input_sents), batch_size), desc='Predicting...'):
            # original
            batch_ = input_sents[start:start + batch_size]
            original_results_ = self.unmasker(batch_)
            original_preds_ = list()
            for r in original_results_:
                tokens = [r_['token_str'].encode('ascii', 'ignore').decode('utf-8').strip() for r_ in r]
                scores = [r_['score'] for r_ in r]
                original_preds_.append({'tokens': tokens, 'scores': scores})
            preds += original_preds_
        return preds

    @staticmethod
    def load_data(resource_dir, experiment_type: int):
        data = []
        file_path = 'exp1_dataset.jsonl' if experiment_type == 1 else 'exp2_dataset.jsonl'
        with open(os.path.join(resource_dir, file_path), 'r', encoding='utf-8') as loadFile:
            for line in loadFile:
                data.append(json.loads(line))
        return data

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def summarise_exp1(
            result_hitrate: typing.Dict,
            result_similarity: typing.Dict,
            input_sents: typing.List,
            pos_tags: typing.List,
            templates: typing.List
    ) -> typing.Dict:
        # 1. merge and update result dict
        result_dict = {}
        for key, value in result_hitrate.items():
            result_dict[key] = value
        for key, value in result_similarity.items():
            result_dict[key] = value

        ANTONYM_TEMPLATE = ['X is an antonym of Y.', "X is the opposite of Y.", "X is different from Y."]
        SYNONUM_TEMPLATE = ['X is a synonym of Y.', "X is the same as Y.", "X is a rephrasing of Y."]

        result_dict['question_type'] = ['ask_antonym' if temp_ in ANTONYM_TEMPLATE else 'ask_synonym'
                                        for temp_ in templates]

        result_dict['pos_tag'] = pos_tags
        result_dict['templates'] = templates

        # 2. calculate statistics of metrics:
        output_metric_dict = {
            "All": {},
            "ask_antonym": {},
            "ask_synonym": {},
        }

        index_dict = {
            "ask_antonym": [i for i, t in enumerate(result_dict['question_type']) if t == 'ask_antonym'],
            "ask_synonym": [i for i, t in enumerate(result_dict['question_type']) if t == 'ask_synonym']
        }

        unique_tag = list(set(pos_tags))
        for tag_ in unique_tag:
            output_metric_dict[tag_] = {}
        for tag_ in unique_tag:
            index_dict[tag_] = [i for i, t in enumerate(result_dict['pos_tag']) if t == tag_]

        unique_temp = list(set(templates))
        for temp_ in unique_temp:
            output_metric_dict[temp_] = {}
        for temp_ in unique_temp:
            index_dict[temp_] = [i for i, t in enumerate(result_dict['templates']) if t == temp_]

        for key, value in tqdm(result_dict.items(), desc="Summarising statistics according to categories..."):
            # only take evaluation metrics
            if 'HR' in key or key in ['jaccard', 'const_cos', 'cos']:
                output_metric_dict['All'][f'avg_{key}'] = float(np.mean(value))

                for key_, index_ in index_dict.items():
                    output_metric_dict[key_][f'avg_{key}'] = float(
                        np.mean([r for i, r in enumerate(result_dict[key])
                                 if i in index_]))
        return output_metric_dict

    @staticmethod
    def summarise_exp2(
            result_hitrate: typing.Dict,
            result_similarity: typing.Dict,
            input_sents: typing.List,
            relations: typing.List,
    ) -> typing.Dict:
        # 1. merge and update result dict

        result_dict = {}
        for key, value in result_hitrate.items():
            result_dict[key] = value
        for key, value in result_similarity.items():
            result_dict[key] = value
        result_dict['question_type'] = ['ask_antonym' if 'antonym' in s else 'ask_synonym' for s in input_sents]
        result_dict['relation'] = relations

        # 2. calculate statistics of metrics:
        output_metric_dict = {
            "All": {},
        }
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
        return output_metric_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # neccesary
    parser.add_argument('--experiment_type', type=int, default=1, choices=[1,2],
                        help='experiment number')
    # path
    parser.add_argument('--resource_dir', type=str, default='resources',
                        help='directory path where data are located')
    parser.add_argument('--save_dir', type=str, default='output',
                        help='directory path where results will be saved')
    # params
    parser.add_argument('--model_type', type=str, default='meaning_matching-bert-base',
                        help='type of pre-trained models for Masked Word Prediction')
    parser.add_argument('--top_k', type=int, default=10,
                        help='top-k predictions for Masked word prediction.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for inference')

    args = parser.parse_args()
    operator = ExperimentOperator(args)
    operator(args.experiment_type)







