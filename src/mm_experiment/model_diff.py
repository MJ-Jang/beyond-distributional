# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import copy
import os
import torch
import json

from transformers import AutoModelForSequenceClassification
from tqdm import tqdm


pretrain_model_dict = {
    "electra-small": ("google/electra-small-discriminator", "meaning_matching-electra-small-n_neg10"),
    "electra-base": ("google/electra-base-discriminator", "meaning_matching-electra-base-n_neg10"),
    "electra-large": ('google/electra-large-discriminator', "meaning_matching-electra-large-n_neg10"),
    "bert-base": ("bert-base-cased", "meaning_matching-bert-base-n_neg10"),
    "bert-large": ("bert-large-cased", "meaning_matching-bert-large-n_neg10"),
    "roberta-base": ("roberta-base", "meaning_matching-roberta-base-n_neg10"),
    "roberta-large": ("roberta-large", "meaning_matching-roberta-large-n_neg10"),
    "albert-base": ("albert-base-v2", "meaning_matching-albert-base-n_neg10"),
    "albert-large": ("albert-large-v2", "meaning_matching-albert-large-n_neg10")
}


def calculate_diff(plm1, plm2):
    assert plm1.keys() == plm2.keys()
    scores = []
    for key in plm1.keys():  # for each layer

        mat1 = plm1[key].numpy()
        mat2 = plm2[key].numpy()

        pow_ = np.power(mat1 - mat2, 2)
        shape_ = pow_.shape
        if len(shape_) == 1:
            len_ = shape_
        elif len(shape_) == 2:
            len_ = shape_[0] * shape_[1]
        else:
            raise NotImplementedError

        frobenius_norm = np.sqrt(np.sum(pow_))  # calculate the frobenius norm
        avg_f_norm = frobenius_norm / len_  # average
        scores.append(avg_f_norm)
    return scores


def main():
    outp = {
        "model": [],
        "mean": [],
        "std": []
    }

    score_outputs = {}

    for key, (m1, m2) in tqdm(pretrain_model_dict.items(), total=len(pretrain_model_dict.keys())):
        model1 = AutoModelForSequenceClassification.from_pretrained(m1)
        # load model from binary file
        dir_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(dir_path, "../mm_experiment/model_binary/", f"{m2}.ckpt")

        model2 = AutoModelForSequenceClassification.from_pretrained(m1)
        model2.load_state_dict(torch.load(file_path))

        if key.startswith('bert'):
            plm1 = copy.deepcopy(model1.bert.state_dict())
            plm2 = copy.deepcopy(model2.bert.state_dict())
        elif key.startswith('electra'):
            plm1 = copy.deepcopy(model1.electra.state_dict())
            plm2 = copy.deepcopy(model2.electra.state_dict())
        elif key.startswith('roberta'):
            plm1 = copy.deepcopy(model1.roberta.state_dict())
            plm2 = copy.deepcopy(model2.roberta.state_dict())
        elif key.startswith('albert'):
            plm1 = copy.deepcopy(model1.albert.state_dict())
            plm2 = copy.deepcopy(model2.albert.state_dict())
        else:
            raise NotImplementedError

        del model1, model2

        scores = calculate_diff(plm1, plm2)
        mean_, std_ = np.mean(scores)[0], np.std(scores)[0]
        
        outp['model'].append(key)
        outp['mean'].append(mean_)
        outp['std'].append(std_)
        score_outputs[key] = scores
    
    out_df = pd.DataFrame(outp)
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../output/mm_experiment')
    os.makedirs(save_dir, exist_ok=True)

    out_df.to_csv(os.path.join(save_dir, 'model_param_diff.tsv'), sep='\t', encoding='utf-8', index=False)

    with open(os.path.join(save_dir, "model_diff_outputs.json"), 'wb') as saveFile:
        json.dump(saveFile, score_outputs)


if __name__ == '__main__':
    main()




