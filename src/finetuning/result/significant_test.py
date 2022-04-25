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

    # conduct_t-test
    # 1) roberta-large
    def calculate_pvalue(backbone_modelname):
        mm_key = [m for m in outp_dict.keys() if backbone_modelname in m and m.startswith('meaning')][0]
        original_key = [m for m in outp_dict.keys() if backbone_modelname in m and not m.startswith('meaning')][0]

        mm_res_ = outp_dict[mm_key]
        res_ = outp_dict[original_key]

        from scipy.stats import ttest_ind
        outp = {}
        for data_ in mm_res_.keys():
            if data_ not in ['mnli', 'cola', 'sst', 'qqp', 'mrpc', 'qnli', 'rte']:
                continue
            mm_acc_ = mm_res_[data_]['val_acc']
            original_acc_ = res_[data_]['val_acc']

            _, p_value = ttest_ind(mm_acc_, original_acc_)
            outp[data_] = p_value
        return outp

    roberta_result = calculate_pvalue('roberta-large')
    bert_result = calculate_pvalue('bert-large')
    electra_result = calculate_pvalue('electra-large')

    result = {
        "roberta": [value for key, value in roberta_result.items()],
        "bert": [value for key, value in bert_result.items()],
        "electra": [value for key, value in electra_result.items()]
    }
    result_df = pd.DataFrame(result)
    result_df.index = list(roberta_result.keys())
    result_df.to_csv(os.path.join(dir_path, 'p_values.tsv'), sep='\t')


if __name__ == '__main__':
    main()
