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
__date__ = "01 07 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"

import torch
import json
import pickle
import argparse
from typing import Text, List, Dict
from transformers import AutoModel
from transformers import AutoTokenizer
from tqdm import tqdm


class WordVectorGenerator:

    def __init__(self, model_name: Text, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.special_tokens = [
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id
        ]

        self.emb_dim = self.model.config.hidden_size
        self.device = device

    def __call__(self, words: List):
        token_outputs = self.tokenizer(
            words,
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

        outp = []
        for m, v in zip(mask, vecs):
            if sum(m) == 1:
                # sing subwords
                outp.append(v.sum(dim=0).cpu().detach().tolist())
            else:
                # multiple subwords - take average (mean pooling)
                outp.append(v.mean(dim=0).cpu().detach().tolist())
        return outp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, default='bert-base')

    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

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

    # 1) load words
    with open('./cn_thesaursus.json', 'r') as loadFile:
        thesaursus = json.load(loadFile)

    word_list = []
    for key, value in thesaursus.items():
        word_list.append(key)
        if value.get('synonym'):
            word_list += value.get('synonym')['tokens']

        if value.get('antonym'):
            word_list += value.get('antonym')['tokens']

        if value.get('hypernym'):
            word_list += value.get('hypernym')['tokens']
    unique_words = list(set(word_list))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = WordVectorGenerator(model_name=pretrain_model_dict[args.model_type], device=device)

    # generate vectors
    word_vec_dict = {}
    for start in tqdm(range(0, len(unique_words), args.batch_size)):
        batch_ = unique_words[start:start+args.batch_size]
        vectors_ = generator(batch_)
        for w, v in zip(batch_, vectors_):
            word_vec_dict[w] = v

    # save vectors
    with open(f'./{args.model_type}-vectors.dict', 'wb') as saveFile:
        pickle.dump(word_vec_dict, saveFile)
