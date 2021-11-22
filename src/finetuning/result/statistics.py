# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os


def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))

    # file_list = [f for f in os.listdir(dir_path) if f.startswith('summary')]
    file_list = [f for f in os.listdir(dir_path) if f.startswith('summary')]
    file_list = sorted(file_list)

    # Save all results
    outp_dict = {}
    for f in file_list:
        df = pd.read_csv(os.path.join(dir_path, f), sep='\t')

        for idx, row in df.iterrows():
            model_, data_ = row['model'], row['data']
            if not outp_dict.get(model_):
                outp_dict[model_] = {}

            metrics = [key for key in row.keys() if key not in ['model', 'data']]
            if not outp_dict.get(model_).get(data_):
                outp_dict[model_][data_] = {key: [] for key in metrics}

            for key in metrics:
                outp_dict[model_][data_][key].append(row[key])

    # Save in new statistic
    stats = {}
    for model_ in sorted(list(outp_dict.keys()), reverse=False):
        for data_ in sorted(list(outp_dict[model_].keys()), reverse=False):
            if not stats.get('model'):
                stats['model'] = []
            if not stats.get('data'):
                stats['data'] = []

            stats['model'].append(model_)
            stats['data'].append(data_)

            # mean
            for key, value in outp_dict[model_][data_].items():
                if not stats.get(f"{key}_mean"):
                    stats[f"{key}_mean"] = []
                stats[f"{key}_mean"].append(round(np.mean(value), 4))

            # std
            for key, value in outp_dict[model_][data_].items():
                if not stats.get(f"{key}_std"):
                    stats[f"{key}_std"] = []
                stats[f"{key}_std"].append(round(np.std(value), 4))

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(dir_path, 'statistics.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    main()
