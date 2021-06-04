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
import re
import argparse
import sys
sys.path.append('../src/')
from nltk.stem.wordnet import WordNetLemmatizer
from utils import spacy_postag, extract_verbs, cn_api_for
from tqdm import tqdm

resource_path = 'resources'
TGT_RELATIONS = ["IsA", "CapableOf", "PartOf", "HasA", "UsedFor", "NotDesires", "MadeOf"]
# Some relations are not appropriate because a negated sentence and original sentence are not always mutually exclusive
# ex. HasProperty: both "Some adults are immature" and "Some adults aren't immature" can be true
file_path = os.path.join(resource_path, 'test.jsonl')


def load_data(file_path: typing.Text) -> typing.List:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


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


def negate_cn_sentence(sent: typing.Text):
    # if a sentence is a negation
    pattern_neg = re.compile("\snot\s|n't\s")
    # only one negation
    if pattern_neg.findall(sent) and len(pattern_neg.findall(sent)) == 1:
        negated_sent = pattern_neg.sub(' ', sent)
        return negated_sent

    # if a sentence is not a negation
    tags_ = spacy_postag(sent)
    verbs_ = extract_verbs(tags_)

    # negate sentence where only one verb appears
    if len(verbs_) == 1:
        verb_, v_tag_ = str(verbs_[0][0]), verbs_[0][1]
        neg_verb_ = negate_verb(verb_, v_tag_)
        negated_sent = sent.replace(verb_, neg_verb_).strip()
    else:
        negated_sent = ''

    return negated_sent


def extract_and_transform_data(data: typing.List[typing.Dict]):
    pattern_only_num = re.compile('\d+')
    outp = []
    for i, d_ in enumerate(tqdm(data)):
        relation_ = d_['pred']
        if relation_ in TGT_RELATIONS:
            sub_ = d_['sub'] if d_.get('sub') else d_['sub_label']
            if pattern_only_num.findall(sub_) and pattern_only_num.findall(sub_)[0] == sub_:
                continue

            sent_ = d_['masked_sentences'][0]
            negated_sent_ = negate_cn_sentence(sent_)

            if not negated_sent_:
                continue

            # set original sent as opposite sent since we cant find wrong_predictions of negated sents
            inst_ = {
                    "word": sub_,
                    "input_sent": negated_sent_,
                    "opposite_sent": sent_,
                    "relation": relation_
                }
            if inst_ in outp:
                continue
            else:
                outp.append(inst_)
    return outp


def process_lama_negation(args):
    data = load_data(file_path)
    data_for_use = extract_and_transform_data(data)
    print(f"Extracted: {len(data_for_use)}")

    # build partial KG dictionary for experiment
    if args.build_new_kg:
        kg_dict = {}
        all_ext_objects = []
        for d in tqdm(data_for_use, desc='constructing partial Conceptnet KG'):
            sub_ = d['word']
            relation_ = d['relation']
            if not kg_dict.get(sub_):
                kg_dict[sub_] = {}
            tokens_, weights_ = cn_api_for(sub_, relation_)
            kg_dict[sub_][relation_] = {
                "tokens": tokens_,
                "weights": weights_
            }
            all_ext_objects += tokens_

        # extract object of DistinctFrom relation for all extracted objects (to use baseline for experiment 2)
        all_ext_objects = list(set(all_ext_objects))
        for ext_obj_ in tqdm(all_ext_objects, "Extracting DistinctFrom/Antonym objects"):
            tokens_, weights_ = cn_api_for(ext_obj_, 'DistinctFrom')
            if tokens_:
                if not kg_dict.get(ext_obj_):
                    kg_dict[ext_obj_] = {}
                kg_dict[ext_obj_]['DistinctFrom'] = {
                    "tokens": tokens_,
                    "weights": weights_
                }

            tokens_, weights_ = cn_api_for(ext_obj_, 'Antonym')
            if tokens_:
                if not kg_dict.get(ext_obj_):
                    kg_dict[ext_obj_] = {}
                kg_dict[ext_obj_]['Antonym'] = {
                    "tokens": tokens_,
                    "weights": weights_
                }

        # save
        kg_path = 'conceptnet_partial.json'
        with open(kg_path, 'w', encoding='utf-8') as saveFile:
            json.dump(kg_dict, saveFile)
    else:
        kg_path = 'conceptnet_partial.json'
        assert os.path.isfile(kg_path)
        with open(kg_path, 'r', encoding='utf-8') as loadFile:
            kg_dict = json.load(loadFile)

    # get wrong_predictions from ConceptNet API
    for d in tqdm(data_for_use):
        word_ = d['word']
        relation_ = d['relation']
        tokens_ = kg_dict[word_][relation_]['tokens']
        d['wrong_prediction'] = list(set(tokens_))

    # save as jsonl for consistency in data format
    save_filename = 'exp2_dataset.jsonl'
    with open(save_filename, 'w', encoding='utf-8') as saveFile:
        for line in data_for_use:
            json.dump(line, saveFile)
            saveFile.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--build_new_kg', type=bool, default=False,
                        help='build new partial Conceptnet KG or not')

    args = parser.parse_args()

    process_lama_negation(args)
