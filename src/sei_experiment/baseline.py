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
__date__ = "20 07 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"


import torch
import os
import pandas as pd
import numpy as np
import json
import pickle
import argparse
from typing import Text, List, Dict
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import AutoTokenizer
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


pretrain_model_dict = {
    "electra-small": "google/electra-small-generator",
    "electra-base": "google/electra-base-discriminator",
    "electra-large": 'google/electra-large-generator',
    "bert-base": "bert-base-cased",
    "bert-large": "bert-large-cased",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "albert-base": "albert-base-v2",
    "albert-large": "albert-large-v2",
}


class WordVectorGenerator:

    def __init__(self, model_name: Text, device):
        if model_name in pretrain_model_dict:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model_dict[model_name])
            self.model = AutoModel.from_pretrained(pretrain_model_dict[model_name])
        elif "meaning_matching" in model_name:
            backbone_model = model_name.replace("meaning_matching-", "").split("-n_neg")[0]
            self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model_dict[backbone_model])
            model_clf = AutoModelForSequenceClassification.from_pretrained(pretrain_model_dict[backbone_model])

            # load model from binary file
            dir_path = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(dir_path, "../mm_experiment/model_binary/", f"{model_name}.ckpt")
            model_clf.load_state_dict(torch.load(file_path))

            if backbone_model.startswith("roberta"):
                self.model = model_clf.roberta
            elif backbone_model.startswith("electra"):
                self.model = model_clf.electra
            elif backbone_model.startswith('bert'):
                self.model = model_clf.bert
            elif backbone_model.startswith('albert'):
                self.model = model_clf.albert
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self.model.to(device)
        self.special_tokens = [
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id
        ]

        self.emb_dim = self.model.config.hidden_size
        self.device = device
        self.batch_size = 64

    def __call__(self, words: List):
        outp = []
        for start in range(0, len(words), self.batch_size):
            batch_words = words[start:start+self.batch_size]

            token_outputs = self.tokenizer(
                batch_words,
                truncation=True,
                padding=True
            )

            # generate mask
            mask = []
            for tokens_ in token_outputs['input_ids']:
                mask_ = [1 if t not in self.special_tokens else 0 for t in tokens_]
                mask.append(mask_)

            inputs = {key: torch.LongTensor(value).to(self.device) for key, value in token_outputs.items()}
            embeds = self.model(**inputs)
            hidden_reps = embeds[0]

            # transform mask
            mask_expansion_ = torch.ones([1, self.emb_dim], dtype=torch.float).to(self.device)
            new_mask = torch.matmul(
                torch.LongTensor(mask).unsqueeze(-1).type(torch.FloatTensor).to(self.device), mask_expansion_
            )

            vecs = hidden_reps * new_mask

            outp_ = []
            for m, v in zip(mask, vecs):
                if sum(m) == 1:
                    # sing subwords
                    outp_.append(v.sum(dim=0).cpu().detach().tolist())
                else:
                    # multiple subwords - take average (mean pooling)
                    outp_.append(v.mean(dim=0).cpu().detach().tolist())
            outp += outp_
        return outp


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir_path, '../../data/SEI_data')

    val_acc, test_acc = list(), list()

    for data in tqdm(['dev', 'test'], desc='Processing datasets'):
        test_set = pd.read_csv(os.path.join(data_path, f'{data}.tsv'), sep='\t')

        word1 = test_set['word1'].tolist()
        word2 = test_set['word2'].tolist()
        label_idx = test_set['label_idx'].tolist()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        candidates = [
            "meaning_matching-roberta-base-n_neg3",
            "meaning_matching-roberta-base-n_neg5",
            "meaning_matching-roberta-base-n_neg10",
            "meaning_matching-roberta-base-n_neg20",
            "meaning_matching-roberta-large-n_neg10",
            "meaning_matching-bert-base-n_neg10",
            "meaning_matching-bert-large-n_neg10",
            "meaning_matching-electra-small-n_neg10",
            "meaning_matching-electra-base-n_neg10",
            "meaning_matching-electra-large-n_neg10",
            "meaning_matching-albert-base-n_neg10",
            "meaning_matching-albert-large-n_neg10",
        ]
        candidates += list(pretrain_model_dict.keys())

        for key in tqdm(candidates, total=len(candidates)):
            generator = WordVectorGenerator(model_name=key, device=device)

            word1_vecs = generator(word1)
            word2_vecs = generator(word2)

            similarity = [cos_sim(w1, w2) for w1, w2 in zip(word1_vecs, word2_vecs)]
            median_similarity = np.median(similarity)  # threshold: median

            pred = [1 if s >= median_similarity else 0 for s in similarity]
            accuracy = accuracy_score(label_idx, pred)

            if data == 'dev':
                val_acc.append(accuracy)
            else:
                test_acc.append(accuracy)

    outp = {
        "model": candidates,
        "val_acc": val_acc,
        "test_acc": test_acc
    }
    outp_df = pd.DataFrame(outp)

    save_dir = os.path.join(dir_path, '../../output/sei_experiment')
    os.makedirs(save_dir, exist_ok=True)
    outp_df.to_csv(os.path.join(save_dir, 'baseline_result.tsv'), sep='\t', encoding='utf-8')


if __name__ == '__main__':
    main()
