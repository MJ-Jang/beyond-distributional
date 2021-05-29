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
__date__ = "15 05 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"

import spacy
import typing
import re
import json
import requests

# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")


def spacy_postag(text: typing.Text) -> typing.List[typing.Tuple[typing.Any, typing.Any]]:
    doc = nlp(text)
    return [(token, token.tag_) for token in list(doc)]


def extract_verbs(
        pos_outp: typing.List,
) -> typing.List:
    """
    Extract target pos-tag tokens
    Args:
        pos_outp: pos-tag output list

    Returns:

    """
    outp = []
    for i, (word, tag) in enumerate(pos_outp):
        if tag.startswith('VB') or tag == 'MD':
            outp.append((word, tag))
    return outp


def cn_api_for(word: typing.Text, relation: typing.Text, weight_threshold: float = 1.0):
    assert relation in ['Antonym', 'DistinctFrom', 'HasA', 'IsA', 'Synonym']
    rm_pattern = re.compile('a |A |an |An |')
    try:
        response = requests.get(f'http://api.conceptnet.io/query?start=/c/en/{word}&rel=/r/{relation}&limit=1000')
        obj = response.json()
    except json.decoder.JSONDecodeError:
        return set(), list()

    outp = list()
    weights = list()
    for e in obj['edges']:
        weight = e['weight']
        end_, ln_ = e['end']['label'], e['end']['language']

        if ln_ == 'en' and len(end_.split(" ")) == 1: # replace to only one token antonym
            outp.append(rm_pattern.sub('', end_).strip())
            weights.append(weight)
    return set(outp), weights
