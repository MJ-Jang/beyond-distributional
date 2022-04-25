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
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from scipy.spatial import distance



def divide_word_vecs(vector_dict):
    word_list, vecs = list(), list()
    for key, value in vector_dict.items():
        word_list.append(key)
        vecs.append(value)
    return word_list, vecs

with open('./data/cn_thesaursus.json', 'r') as loadFile:
    thesaursus = json.load(loadFile)

model_name = 'bert-base'
# load vectors
with open(f'./data/{model_name}-vectors.dict', 'rb') as readFile:
    vector = pickle.load(readFile)

words, vector = divide_word_vecs(vector)

knn = NearestNeighbors(n_neighbors=11, metric='cosine')
knn.fit(np.array(vector))

for i, (key, value) in tqdm(enumerate(thesaursus.items()), total=len(thesaursus)):
    if value['antonym'].get('tokens'):
        _, index = knn.kneighbors(np.array(vector[i]).reshape(1, -1))
        selected_words = [words[i] for i in index[0][1:]]

        antonyms = value['antonym']['tokens']
        inter_ant = set(selected_words).intersection(set(antonyms))

        print(len(inter_ant))



