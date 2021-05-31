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

import typing
import numpy as np


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
    score = pred_score[:len(pred_)]

    hit_ = [1 if p in target else 0 for p in pred_]

    hr = sum(hit_) / top_k
    if score:
        w_hr = np.dot(np.array(hit_), np.array(score)) / sum(score)
    else:
        w_hr = 0.0
    return {"hr": hr, "w_hr": w_hr}


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
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2) + 1e-6)
