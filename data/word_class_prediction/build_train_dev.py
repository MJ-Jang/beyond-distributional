# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

from nltk.corpus import wordnet
from tqdm import tqdm
from typing import Text


# construct dataset
def construct_dataset():
    all_synsets = list()
    for w in wordnet.all_synsets():
        all_synsets.append(w)

    cnt = 0
    example_cnt = []
    sents, words, pos = list(), list(), list()
    for synset_ in tqdm(all_synsets):
        examples_ = synset_.examples()
        if len(examples_) == 0:
            cnt += 1
            continue
        else:
            example_cnt.append(len(examples_))
        pos_ = synset_._pos
        word_ = synset_._name.split(".")[0].replace("_", " ")

        sents += examples_
        words += [word_] * len(examples_)
        if pos_ == 's':
            pos_ = 'a'  # change adjective satelites to adjective
        pos += [pos_] * len(examples_)

        if word_ == 'embody':
            print(synset_)

    outp = {
        "Word": words,
        "Sentence": sents,
        "Pos": pos
    }
    print(f"No examples: {cnt} | Have examples: {len(all_synsets) - cnt}")
    print(f"# of examples mean: {np.mean(example_cnt)} | # of examples std: {np.std(example_cnt)}")
    print(f"# of examples max: {np.max(example_cnt)} | # of examples median: {np.median(example_cnt)}")

    return outp


def check_word_sent_overlap(word: Text, sent: Text):
    res = ""
    for i in word:
        if i in sent and res + i in sent:
            res += i

    if res in word and len(res) > 1:
        return True
    else:
        return False


def main():
    data = construct_dataset()
    df = pd.DataFrame(data)

    drop_idx = []
    for i, d in df.iterrows():
        if check_word_sent_overlap(d['Word'], d['Sentence']):
            continue
        else:
            drop_idx.append(i)
    df = df.drop(drop_idx)
    print(f"Total number of data: {len(df)}")

    word_, pos_ = df['Word'].tolist(), df['Pos'].tolist()
    unique_pos = set(pos_)
    print(unique_pos)
    # n: noun, v: verb, a: adjective, r: adverb, s: adjective satellite

    word_pos_dict = {}
    for w, p in zip(word_, pos_):
        if word_pos_dict.get(w):
            word_pos_dict[w].add(p)
        else:
            word_pos_dict[w] = {p}

    n_class = []
    for key, value in word_pos_dict.items():
        n_class.append(len(value))
    print(f"Avg of class(pos)/word: {np.mean(n_class)} | Std of class(pos)/word: {np.std(n_class)}")
    print(f"Max of class(pos)/word: {np.max(n_class)} | Median of class(pos)/word: {np.median(n_class)}")

    # shuffle
    df = df.sample(frac=1)
    train = df[:int(len(df) * 0.95)]
    dev = df[int(len(df) * 0.95):]
    print(f"# of training data: {len(train)} | # of dev data: {len(dev)}")

    pwd = os.path.dirname(os.path.abspath(__file__))
    train.to_csv(os.path.join(pwd, 'train.tsv'), sep='\t', index=False, encoding='utf-8')
    dev.to_csv(os.path.join(pwd, 'dev.tsv'), sep='\t', index=False, encoding='utf-8')


if __name__ == '__main__':
    main()
