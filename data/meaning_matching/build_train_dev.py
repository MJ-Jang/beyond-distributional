# -*- coding: utf-8 -*-

import json
import os
import random
import pandas as pd
import argparse

from tqdm import tqdm

DIR = os.path.dirname(os.path.abspath(__file__))


def main(args):
    data = []
    with open(os.path.join(DIR, 'meaning_matching_all.jsonl'), 'r', encoding='utf-8') as readFile:
        for line in readFile:
            data.append(json.loads(line))

    df_dict = {
        "word": [],
        "definition": [],
        "label": []
    }

    for i, d in enumerate(tqdm(data, total=len(data))):
        for def_ in d['definition']:
            population = [i_ for i_ in range(len(data)) if i_ != i]
            sample_idx_ = random.sample(population, args.n_neg_sample)
            neg_samples = [d for i, d in enumerate(data) if i in sample_idx_]

            ws_ = [d['word']] * (args.n_neg_sample + 1)
            neg_def_ = [dn['definition'][0] for dn in neg_samples]
            defs_ = [def_] + neg_def_
            label_ = [1] + [0] * args.n_neg_sample

            df_dict['word'] += ws_
            df_dict['definition'] += defs_
            df_dict['label'] += label_

    df = pd.DataFrame(df_dict)
    df = df.sample(frac=1)

    tr_split = int(len(df) * 0.85)
    train = df[:tr_split]
    dev = df[tr_split:]

    train.to_csv(os.path.join(DIR, f'train-n_neg{args.n_neg_sample}.tsv'), sep='\t', encoding="utf-8", index=False)
    dev.to_csv(os.path.join(DIR, f'dev-{args.n_neg_sample}.tsv'), sep='\t', encoding="utf-8", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_neg_sample', type=int, default=3,
                        help='number of negative samples')
    args = parser.parse_args()
    main(args)
