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
__date__ = "22 07 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"

import torch
import transformers
import os
import yaml
import json
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data import MeaningMatchingnDataModule

pretrain_model_dict = {
    "electra-small": "google/electra-small-discriminator",
    "electra-base": "google/electra-base-discriminator",
    "electra-large": 'google/electra-large-discriminator',
    "bert-base": "bert-base-cased",
    "bert-large": "bert-large-cased",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "albert-base": "albert-base-v2",
    "albert-large": "albert-large-v2",
}


class MMInferencer:

    def __init__(
            self,
            model,
            batch_size: int = 64
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.batch_size = batch_size
        self.model.to(self.device)

    def __call__(self, dataset) -> List:
        outp = []
        for start in tqdm(range(0, len(dataset), self.batch_size)):
            batch = dataset[start:start + self.batch_size]

            inputs = dict()
            for key, value in batch.items():
                if key == 'labels':
                    continue
                inputs[key] = value.to(self.device)

            logits = self.model(**inputs)['logits']
            preds = logits.argmax(dim=-1)
            preds = preds.detach().cpu().tolist()
            outp += preds
        return outp


def main(args):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dir_path, 'config.yaml'), 'r') as readFile:
        config_file = yaml.load(readFile, Loader=yaml.SafeLoader)
    cfg = config_file.get('cfg')

    if args.backbone_model_name in pretrain_model_dict:
        tokenizer = AutoTokenizer.from_pretrained(pretrain_model_dict[args.backbone_model_name])
        model = AutoModelForSequenceClassification.from_pretrained(pretrain_model_dict[args.backbone_model_name])
    elif "meaning_matching" in args.backbone_model_name:
        backbone_model = args.backbone_model_name.replace("meaning_matching-", "").split("-n_neg")[0]
        tokenizer = AutoTokenizer.from_pretrained(pretrain_model_dict[backbone_model])
        model = AutoModelForSequenceClassification.from_pretrained(pretrain_model_dict[backbone_model])

        # load model from binary file
        dir_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(dir_path, "./model_binary/", f"{args.backbone_model_name}.ckpt")

        model.load_state_dict(torch.load(file_path))

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.backbone_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(args.backbone_model_name)

    data_dir_path = os.path.join(dir_path, '../../data/meaning_matching')

    data_module = MeaningMatchingnDataModule(
        tokenizer,
        data_dir_path=data_dir_path,
        n_neg=cfg.get('n_neg'),
        max_length=cfg.get('max_length'),
    )

    feature_dict = data_module()

    # load model
    model.eval()

    if 'large' in args.backbone_model_name:
        batch_size = cfg.get("batch_size_large")
    else:
        batch_size = cfg.get("batch_size_base")

    inferencer = MMInferencer(model, batch_size=batch_size)

    for data_type in ['validation']:
        pred_dataset = feature_dict[data_type]
        predictions = inferencer(dataset=pred_dataset)

        perf_dict = {}
        acc = accuracy_score(y_true=pred_dataset['label'], y_pred=predictions)

        perf_dict['accuracy'] = acc
        print(f"{args.backbone_model_name}|{data_type}| Accuracy: {acc}")

        outputs = {
            "idx": [i for i in range(len(predictions))],
            "preds": predictions,
            "acc": acc
        }
        outputs.update(perf_dict)

        # Here
        save_path = os.path.join(dir_path, args.save_dir,
                                 f"{args.backbone_model_name}")
        os.makedirs(save_path, exist_ok=True)
        file_name = f"{data_type}.json"
        with open(os.path.join(save_path, file_name), 'w') as saveFile:
            json.dump(outputs, saveFile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone_model_name', type=str, default='google/electra-large-discriminator',
                        help='type or pre-trained models')
    parser.add_argument('--save_dir', type=str, default='../../output/mm_experiment',
                        help='directory to save results')

    args = parser.parse_args()

    main(args)

