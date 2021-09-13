# -*- coding: utf-8 -*-

import json
import os
import random
import pandas as pd
from tqdm import tqdm

DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    data = []
    with open(os.path.join(DIR, 'meaning_matching_all.jsonl'), 'r', encoding='utf-8') as readFile:
        for line in readFile:
            data.append(json.loads(line))

    ratio = 3
    df_dict = {
        "word": [],
        "definition": [],
        "label": []
    }

    for i, d in enumerate(tqdm(data, total=len(data))):
        len_def = len(d['definition'])

        for def_ in d['definition']:
            population = [i_ for i_ in range(len(data)) if i_ != i]
            sample_idx_ = random.sample(population, ratio)
            neg_samples = [d for i, d in enumerate(data) if i in sample_idx_]

            ws_ = [d['word']] * (ratio + 1)
            neg_def_ = [dn['definition'][0] for dn in neg_samples]
            defs_ = [def_] + neg_def_
            label_ = [1] + [0] * ratio

            df_dict['word'] += ws_
            df_dict['definition'] += defs_
            df_dict['label'] += label_

    df = pd.DataFrame(df_dict)
    df = df.sample(frac=1)

    tr_split = int(len(df) * 0.85)
    train = df[:tr_split]
    dev = df[tr_split:]

    train.to_csv(os.path.join(DIR, 'train.tsv'), sep='\t', encoding="utf-8", index=False)
    dev.to_csv(os.path.join(DIR, 'dev.tsv'), sep='\t', encoding="utf-8", index=False)