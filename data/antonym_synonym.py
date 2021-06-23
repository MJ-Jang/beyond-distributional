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
__date__ = "25 05 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"


import os
import typing
import json
import argparse
import sys
sys.path.append('../src/')
from textblob import TextBlob
from tqdm import tqdm
from collections import Counter
from utils import cn_api_for, spacy_postag


resource_path = 'resources'

ANTONYM_TEMPLATE = ['X is an antonym of Y.', "X is the opposite of Y.", "X is different from Y."]
SYNONUM_TEMPLATE = ['X is a synonym of Y.', "X is another form of Y.", "X is a rephrasing of Y."]


def load_snli(file_path: typing.Text) -> typing.List:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_sents_from_snli(data_list: typing.List):
    sent1 = [d['sentence1'] for d in data_list]
    sent2 = [d['sentence2'] for d in data_list]
    return sent1 + sent2


def do_singularise(word: typing.Text) -> typing.Text:
    transformed = TextBlob(word).words
    if transformed:
        return str(TextBlob(word).words[0].singularize())
    else:
        return word


def do_pluralise(word: typing.Text) -> typing.Text:
    transformed = TextBlob(word).words
    if transformed:
        return str(TextBlob(word).words[0].pluralize())
    else:
        return word


def build_thesarsus(word_dict: typing.Dict, thesarsus: typing.Dict):
    """

    Args:
        word_dict: key: pos-tag, value: words. (e.g. {{"Noun": [...]}})
        thesarsus: thesarsus dictionary

    Returns: updated thesarsus

    """

    # 1. reform data for efficiency
    words, tags = list(), list()
    for key, value in word_dict.items():
        words += value
        tags += [key] * len(value)

    candidates = [(w,t) for w,t in zip(words, tags)]

    # 2. check subject and object have both synonym and antonym in ConceptNet
    final_candidates = []
    for w, t in tqdm(candidates):
        if thesarsus.get(w):
            continue
        antonyms_, a_weights = cn_api_for(w, 'Antonym')
        synonyms_, s_weights = cn_api_for(w, 'Synonym')
        if len(antonyms_) > 0 or len(synonyms_) > 0:
            final_candidates.append(w)

            hypernyms_, h_weights = cn_api_for(w, 'IsA') # word is hyponym of [Something] -> Something is hypernym of word
            # vehicle is a machine -> machine is hypernym of vehicle

            thesarsus[w] = {
                "tag": t,
                "synonym": {
                    "tokens": list(synonyms_),
                    "weights": s_weights
                },
                "antonym": {
                    "tokens": list(antonyms_),
                    "weights": a_weights
                },
                "hypernym": {
                    "tokens": list(hypernyms_),
                    "weights": h_weights
                },
            }

    print(f"{len(final_candidates)} / {len(candidates)} candidate words are extracted")
    return thesarsus


def extract_words(sents: typing.List, min_cnt:int=1):
    """
    Extract only Noun, Adjective, and Adverb.
    Args:
        sents: List of Sentences
        min_cnt: Minimun number of appearance of words

    Returns: extract word dictionary. Key: pos-tag, Value: list of words

    """
    word_dict = {
        "Noun": [],
        "Adjective": [],
        "Adverb": []
    }

    for s in tqdm(sents, desc='Extracting words'):
        outp = spacy_postag(s)
        for w_, t_ in outp:
            w_ = str(w_)
            if t_.startswith('NN'):
                # if noun -> singularise
                w_ = do_singularise(w_)
                word_dict['Noun'].append(w_)
            if t_.startswith('RB'):
                word_dict['Adverb'].append(w_)
            if t_.startswith('JJ'):
                word_dict['Adjective'].append(w_)

    # extract word appeared more than min_cnt
    for key, value in word_dict.items():
        counts = Counter(value)
        new_words = [word for word, cnt in counts.items() if cnt >= min_cnt]
        word_dict[key] = new_words

    for key, value in word_dict.items():
        word_dict[key] = list(set(value)) # use unique words only

    print(f"# of Extract Noun: {len(word_dict['Noun'])}")
    print(f"# of Extract Adjective: {len(word_dict['Adjective'])}")
    print(f"# of Extract Adverb: {len(word_dict['Adverb'])}")
    return word_dict


def generate_sentences(thesarsus):
    # 3. generate template-based sentences
    words, input_sents, wrong_predictions, pos_tags, opposite_sents, templates =\
        list(), list(), list(), list(), list(), list()
    for word in thesarsus.keys():
        # If the word has synonym, use antonym-template to generate inputs.
        # If the model predicts synonyms or hypernyms as predictions, this is a wrong behaviour
        if thesarsus[word].get('synonym'):
            ant_sents_ = [t.replace('X', word) for t in ANTONYM_TEMPLATE]
            opposite_sents_ = [t.replace('X', word) for t in SYNONUM_TEMPLATE]

            # add the word itself and it's plural form because it is definitely not an antonym
            if thesarsus[word].get('hypernym'):
                wrong_prediction_ = [thesarsus[word].get('synonym')['tokens'] +
                                     thesarsus[word].get('hypernym')['tokens'] + [word, do_pluralise(word)]] \
                                    * len(ant_sents_)

            else:
                wrong_prediction_ = [thesarsus[word].get('synonym')['tokens'] + [word], do_pluralise(word)] \
                                    * len(ant_sents_)

            input_sents += ant_sents_
            templates += ANTONYM_TEMPLATE
            wrong_predictions += wrong_prediction_
            words += [word] * len(ant_sents_)
            pos_tags += [thesarsus[word].get('tag')] * len(ant_sents_)
            opposite_sents += opposite_sents_

        # If the word has antonym, use synonym-template to generate inputs.
        # If the model predicts antonyms as predictions, this is a wrong behaviour
        if thesarsus[word].get('antonym'):
            syn_sents_ = [t.replace('X', word) for t in SYNONUM_TEMPLATE]
            opposite_sents_ = [t.replace('X', word) for t in ANTONYM_TEMPLATE]

            wrong_prediction_ = [thesarsus[word].get('antonym')['tokens']] * len(syn_sents_)

            input_sents += syn_sents_
            templates += SYNONUM_TEMPLATE
            wrong_predictions += wrong_prediction_
            words += [word] * len(syn_sents_)
            pos_tags += [thesarsus[word].get('tag')] * len(syn_sents_)
            opposite_sents += opposite_sents_

    assert len(input_sents) == len(wrong_predictions)
    assert len(input_sents) == len(words)

    outp = []
    for i in range(len(input_sents)):
        instance_ = {
            "word": words[i],
            "input_sent": input_sents[i],
            "wrong_prediction": list(set(wrong_predictions[i])),
            "pos_tag": pos_tags[i],
            "opposite_sent": opposite_sents[i],
            "template": templates[i]
        }
        if instance_['wrong_prediction']:
            # sometimes the word has weight but no following words (need to check)
            outp.append(instance_)

    save_filename = 'exp1_dataset.jsonl'
    if os.path.isfile(save_filename):
        os.remove(save_filename)
    with open(save_filename, 'w', encoding='utf-8') as saveFile:
        for obs in outp:
            json.dump(obs, saveFile)
            saveFile.write("\n")
    print(f"{len(outp)} sentences are generated for experiment 1!")


def main(args):
    # 1. load data
    train_path = os.path.join(resource_path, 'snli_1.0_train.jsonl')
    train = load_snli(train_path)
    train_sents = extract_sents_from_snli(train)
    
    dev_path = os.path.join(resource_path, 'snli_1.0_dev.jsonl')
    dev = load_snli(dev_path)
    dev_sents = extract_sents_from_snli(dev)

    test_path = os.path.join(resource_path, 'snli_1.0_test.jsonl')
    test = load_snli(test_path)
    test_sents = extract_sents_from_snli(test)

    # 2. build thesarsus and back-up
    os.makedirs('output', exist_ok=True)
    thesarsus_path = 'cn_thesaursus.json'
    if os.path.isfile(thesarsus_path):
        with open(thesarsus_path, 'r', encoding='utf-8') as loadFile:
            thesarsus = json.load(loadFile)
            print("Thesarsus is loaded")
    else:
        thesarsus = {}

    # First build or update thesarsus
    if args.update_thesarsus:
        word_dict = extract_words(train_sents + dev_sents + test_sents, min_cnt=args.min_cnt)
        thesarsus = build_thesarsus(word_dict, thesarsus)

        with open('cn_thesaursus.json', 'w', encoding='utf-8') as saveFile:
            json.dump(thesarsus, saveFile)
    # 3. generate template-based sentences
    generate_sentences(thesarsus)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--update_thesarsus', type=bool, default=False,
                        help='whehter to update thesarsus we are going to use. Should be True at the first time.')
    parser.add_argument('--min_cnt', type=int, default=5,
                        help='Minimum occurance of words we are going to extract')

    args = parser.parse_args()

    main(args)
