# -*- coding: utf-8 -*-

import json
import os
import pandas as pd
import argparse

from collections import OrderedDict
from tqdm import tqdm


def main(args):
    dir_path = os.path.dirname(os.path.abspath(__file__))

    outp = OrderedDict(
        {
            "model": [],
            "data": [],
            "val_acc": [],
            "val_f1": []
        }
    )

    result_saved_path = os.path.join(dir_path, args.repeats)
    model_list = [f for f in os.listdir(result_saved_path) if not f.endswith('.DS_Store')]
    model_list = [f for f in model_list if not f.endswith('.py')]
    model_list = sorted(model_list, reverse=False)

    for m_ in tqdm(model_list):
        file_list = os.listdir(f'{result_saved_path}/{m_}')
        data_list = list(set([s.split('-')[0] for s in file_list]))
        data_list = sorted(data_list, reverse=False)

        for d_ in data_list:
            if d_ == 'mnli':
                val = json.load(open(f'{result_saved_path}/{m_}/{d_}-validation_matched.json', 'r'))
                acc = val['accuracy']
                f1 = val['f1']

                outp['model'].append(m_)
                outp['data'].append(f"{d_}_matched")
                outp['val_acc'].append(acc)
                outp['val_f1'].append(f1)

                val = json.load(open(f'{result_saved_path}/{m_}/{d_}-validation_mismatched.json', 'r'))
                acc = val['accuracy']
                f1 = val['f1']

                outp['model'].append(m_)
                outp['data'].append(f"{d_}_mismatched")
                outp['val_acc'].append(acc)
                outp['val_f1'].append(f1)
            elif d_ == 'cola':
                val = json.load(open(f'{result_saved_path}/{m_}/{d_}-validation.json', 'r'))
                acc = val['matthews_cor']  # for cola, not accuracy but matthews correlation
                f1 = val['f1']

                outp['model'].append(m_)
                outp['data'].append(d_)
                outp['val_acc'].append(acc)
                outp['val_f1'].append(f1)
            else:
                val = json.load(open(f'{result_saved_path}/{m_}/{d_}-validation.json', 'r'))
                acc = val['accuracy']
                f1 = val['f1']

                outp['model'].append(m_)
                outp['data'].append(d_)
                outp['val_acc'].append(acc)
                outp['val_f1'].append(f1)

    outp_df = pd.DataFrame(outp)
    outp_df.sort_values(by=['model'])
    outp_df.to_csv(os.path.join(dir_path, f"summary_{args.repeats}.tsv"), sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=str, default='1')
    args = parser.parse_args()

    main(args)
