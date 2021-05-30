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
__date__ = "29 05 2021"
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


def jaccard_similarity(list1, list2):
    intersection = set(list1).intersection(set(list2))
    union = set(list1).union(set(list2))
    return len(intersection) / len(union)


def cosine_similarity(words1, words2, scores1, scores2, const_weight: bool = False):
    union = list(set(words1).union(set(words2)))
    if const_weight:
        vec1 = [1 if v in words1 else 0 for i, v in enumerate(union)]
        vec2 = [1 if v in words2 else 0 for i, v in enumerate(union)]
    else:
        vec1 = [scores1[words1.index(v)] if v in words1 else 0 for i, v in enumerate(union)]
        vec2 = [scores2[words2.index(v)] if v in words2 else 0 for i, v in enumerate(union)]
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


def predict_plm(
        model,
        original_sents: typing.List,
        negated_sents: typing.List,
        batch_size: int = 16
) -> typing.Tuple[typing.List, typing.List]:

    mask_token = model.tokenizer.mask_token
    original_sents = [s.replace('Y', mask_token) for s in original_sents]
    negated_sents = [s.replace('Y', mask_token) for s in negated_sents]

    original_preds, negated_preds = list(), list()
    for start in tqdm(range(0, len(original_sents), batch_size), desc='Predicting...'):
        # original
        batch_o_ = original_sents[start:start + batch_size]
        batch_n_ = negated_sents[start:start + batch_size]

        original_results_ = model(batch_o_)
        original_preds_ = list()
        for r in original_results_:
            tokens = [r_['token_str'].encode('ascii', 'ignore').decode('utf-8') for r_ in r]
            scores = [r_['score'] for r_ in r]
            original_preds_.append({'tokens': tokens, 'scores': scores})

        negated_results_ = model(batch_n_)
        negated_preds_ = list()
        for r in negated_results_:
            tokens = [r_['token_str'].encode('ascii', 'ignore').decode('utf-8') for r_ in r]
            scores = [r_['score'] for r_ in r]
            negated_preds_.append({'tokens': tokens, 'scores': scores})

        original_preds += original_preds_
        negated_preds += negated_preds_
    return original_preds, negated_preds


def evaluate(
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
        "jaccard": float(np.mean(jaccard)),
        "const_cos": float(np.mean(const_cos)),
        "cos": float(np.mean(cos))
    }
    return outp


def main(args):

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

    if args.model_type == 'baseline':
        raise NotImplementedError
    else:
        # 1. load model
        print(f"Loading {args.model_type} model...")
        use_gpu = torch.cuda.is_available()
        device = 0 if use_gpu else -1
        model = pipeline('fill-mask', model=pretrain_model_dict[args.model_type], device=device, top_k=args.top_k)

        # 2. load data
        with open(os.path.join(args.resource_dir, 'exp2_dataset.jsonl'), 'r', encoding='utf-8') as loadFile:
            data = []
            for line in loadFile:
                data.append(json.loads(line))

        # 3. conduct inference
        original_sents = [d['original_sent'] for d in data]
        negated_sents = [d['negated_sent'] for d in data]

        original_preds, negated_preds = predict_plm(model, original_sents, negated_sents, batch_size=args.batch_size)
        result = evaluate(original_preds, negated_preds)

        # update prediction to data
        for i, d in enumerate(data):
            d['original_pred'] = original_preds[i]
            d['negated_pred'] = negated_preds[i]

        # 4. save results
        os.makedirs(args.save_dir, exist_ok=True)
        result_file_name = os.path.join(args.save_dir, f"{args.model_type}-result.yaml")
        pred_file_name = os.path.join(args.save_dir, f"{args.model_type}-prediction.json")

        with open(result_file_name, 'w') as resultFile:
            yaml.dump(result, resultFile, default_flow_style=False, sort_keys=False)
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
    parser.add_argument('--save_dir', type=str, default='output/exp2',
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




