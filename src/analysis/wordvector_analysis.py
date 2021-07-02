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
__date__ = "02 07 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"

import pickle
import json
import os
import pandas as pd
import numpy as np

from typing import Dict, Text
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
from scipy.stats import ttest_ind


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


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


def calculate_score(word: Text, preds: Dict, vector_dict: Dict):
    if not preds['tokens']:
        return False
    word_vec = vector_dict.get(word)

    sorted_pred = sorted([[t,w] for t,w in zip(preds['tokens'], preds['weights'])], key=lambda x:x[1], reverse=True)
    tgt_word = sorted_pred[0][0] # take the word with the highest weight
    tgt_word_vec = vector_dict.get(tgt_word)
    return cos_sim(word_vec, tgt_word_vec)
    # tokens = preds['tokens']
    # outp = []
    # for t in tokens:
    #     tgt_word_vec = vector_dict.get(t)
    #     outp.append(cos_sim(word_vec, tgt_word_vec))
    # return outp



def generate_statistics(model_name: Text, thesaursus: Dict):
    # load vectors
    with open(f'../../data/{model_name}-vectors.dict', 'rb') as readFile:
        vector = pickle.load(readFile)

    antonym_sim, synonym_sim, hypernym_sim, rand_sim = list(), list(), list(), list()
    for key, value in tqdm(thesaursus.items()):
        # 1. synonym
        sim_score = calculate_score(key, value.get('synonym'), vector)
        if sim_score:
            synonym_sim.append(sim_score)

        # 2. antonym
        sim_score = calculate_score(key, value.get('antonym'), vector)
        if sim_score:
            antonym_sim.append(sim_score)

        # 3. hypernym
        sim_score = calculate_score(key, value.get('hypernym'), vector)
        if sim_score:
            hypernym_sim.append(sim_score)

    # t-test
    _, pvalue_syn_ant = ttest_ind(
        a=synonym_sim,
        b=antonym_sim,
        equal_var=False
    )

    _, pvalue_hyp_ant = ttest_ind(
        a=hypernym_sim,
        b=antonym_sim,
        equal_var=False
    )

    _, pvalue_hyp_syn = ttest_ind(
        a=hypernym_sim,
        b=synonym_sim,
        equal_var=False
    )

    outp = {
        "mean_synonym": round(np.mean(synonym_sim), 3),
        "std_synonym": round(np.std(synonym_sim), 3),
        "mean_antonym": round(np.mean(antonym_sim), 3),
        "std_antonym": round(np.std(antonym_sim), 3),
        "mean_hypernym": round(np.mean(hypernym_sim), 3),
        "std_hypernym": round(np.std(hypernym_sim), 3),
        "pvalue_ant_syn": round(pvalue_syn_ant, 3),
        "pvalue_ant_hyp": round(pvalue_hyp_ant, 3),
        "pvalue_syn_hyp": round(pvalue_hyp_syn, 3)
    }
    return outp


if __name__ == '__main__':
    # 1. load words
    with open('../../data/cn_thesaursus.json', 'r') as loadFile:
        thesaursus = json.load(loadFile)

    result = dict()
    for key, _ in tqdm(pretrain_model_dict.items()):
        try:
            outp_ = generate_statistics(key, thesaursus)
            for key, value in outp_.items():
                if result.get(key):
                    result[key].append(value)
                else:
                    result[key] = [value]
        except FileNotFoundError:
            raise NotImplementedError

    result_df = pd.DataFrame(result)
    outp_path = '../../output/vector_analysis'
    os.makedirs(outp_path, exist_ok=True)
    result_df.insert(loc=0, column='models', value=list(result.keys()))
    result_df.to_csv(os.path.join(outp_path, 'vector_similarity_result.tsv'), sep='\t', index=False)


