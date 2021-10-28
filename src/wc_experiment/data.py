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
__date__ = "20 07 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"


import pandas as pd
import os
from typing import Text, List
from datasets import Dataset, DatasetDict, ClassLabel


class WordClassPredictionDataModule:

    def __init__(
            self,
            tokenizer,
            data_dir_path: Text,
            max_length: int = 64,
            padding: Text = 'max_length',
            truncation: Text = 'longest_first'
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.data_dir_path = data_dir_path

    def __call__(self, *args, **kwargs):
        dataset = self.load_wc_dataset(self.data_dir_path)
        features_dict = {}

        for phase, phase_dataset in dataset.items():
            features_dict[phase] = phase_dataset.map(
                self.convert_to_features,
                batched=True,
                load_from_cache_file=False,
            )
            try:
                features_dict[phase].set_format(
                    type="torch",
                    columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'],
                )
            except:
                features_dict[phase].set_format(
                    type="torch",
                    columns=['input_ids', 'attention_mask', 'labels'],
                )
        return features_dict

    def convert_to_features(self, example_batch):
        inputs = list(zip(example_batch['Word'], example_batch['Sentence']))
        features = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=self.max_length,
            padding='max_length',
            truncation='longest_first'
        )
        features["labels"] = example_batch["label"]
        return features

    def load_wc_dataset(self, data_dir_path: Text):
        train = pd.read_csv(os.path.join(data_dir_path, f'train.tsv'), sep='\t')
        dev = pd.read_csv(os.path.join(data_dir_path, f'dev.tsv'), sep='\t')
        
        train = train.dropna()
        dev = dev.dropna()

        train['label'] = [self.map_label_to_idx(p) for p in train['Pos']]
        train = train.drop(columns='Pos')
        dev['label'] = [self.map_label_to_idx(p) for p in dev['Pos']]
        dev = dev.drop(columns='Pos')

        train = Dataset.from_dict({key: train[key].tolist() for key in train.keys()})
        train.features['label'] = ClassLabel(num_classes=4, names=["n", "v", "a", "r"])

        dev = Dataset.from_dict({key: dev[key].tolist() for key in dev.keys()})
        dev.features['label'] = ClassLabel(num_classes=4, names=["n", "v", "a", "r"])

        outp_dict = DatasetDict(
            {
                "train": train,
                "validation": dev,
            }
        )
        return outp_dict

    @staticmethod
    def map_label_to_idx(labels: List):
        label_dict = {
            "n": 0,
            "v": 1,
            "a": 2,
            "r": 3
        }
        return [label_dict[l] for l in labels]


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/electra-small-generator')
    module = WordClassPredictionDataModule(tokenizer, '../../data/word_class_prediction')
    feature_dict = module()
