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
__date__ = "14 05 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"


import typing
import json
import os
import pandas as pd
import sys
sys.path.append('..')
from nltk.stem.wordnet import WordNetLemmatizer
from util import spacy_postag, extract_verbs
from tqdm import tqdm

resource_path = '../resources'
TGT_RELATIONS = ["IsA", "CapableOf", "PartOf", "HasA", "UsedFor", "NotDesires", "MadeOf", "HasProperty"]
file_path = os.path.join(resource_path, 'test.jsonl')


def load_data(file_path: typing.Text) -> typing.List:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_subject_object(data: typing.List):
    sub_ = [d['sub'] for d in data if d.get('sub')]
    obj_ = [d['obj_label'] for d in data]
    return list(set(sub_ + obj_))


def negate_verb(verb: typing.Text, tag: typing.Text):
    verb_lemma = WordNetLemmatizer().lemmatize(verb, 'v')
    if verb == 'can':
        return 'cannot'
    elif verb in ['is', 'are', 'do', 'does']:
        return verb + "n't"
    elif tag == 'MD':
        return verb + "n't"
    elif tag == 'VBZ':
        return "doesn't " + verb_lemma
    elif tag == 'VBD':
        return "didn't" + verb_lemma
    else:
        return "don't " + verb


def negate_cn_sentences(sents: typing.List):
    original_sents, negate_sents = [], []
    for sent_ in tqdm(sents):
        tags_ = spacy_postag(sent_)
        verbs_ = extract_verbs(tags_)

        # negate sentence where only one verb appears
        if len(verbs_) == 1:
            original_sents.append(sent_)

            verb_, v_tag_ = str(verbs_[0][0]), verbs_[0][1]
            neg_verb_ = negate_verb(verb_, v_tag_)
            negate_sents.append(sent_.replace(verb_, neg_verb_).strip())

    return original_sents, negate_sents


def process_lama_negation():
    data = load_data(file_path)
    outp = [d for d in data if d['pred'] in TGT_RELATIONS]
    sents = [o['masked_sentences'][0] for o in outp]
    print(f"Extracted: {len(sents)} | Unique: {len(list(set(sents)))}")
    original_sents, negated_sents = negate_cn_sentences(list(set(sents)))

    outp_df = pd.DataFrame(
        {
            "original_sents": original_sents,
            "negated_sents": negated_sents
        }
    )
    outp_df = outp_df.drop_duplicates()

    save_filename = os.path.join(resource_path, 'lama_neg.tsv')
    if os.path.isfile(save_filename):
        os.remove(save_filename)
    outp_df.to_csv(save_filename, sep='\t', index=False, encoding='utf-8')
    print(f"{len(outp_df)} sentences are generated")


if __name__ == '__main__':
    process_lama_negation()
