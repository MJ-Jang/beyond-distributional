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
from typing import Text
from datasets import Dataset, DatasetDict, ClassLabel


class MeaningMatchingnDataModule:

    def __init__(
            self,
            tokenizer,
            data_dir_path: Text,
            n_neg: int,
            max_length: int = 64,
            padding: Text = 'max_length',
            truncation: Text = 'longest_first'
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.data_dir_path = data_dir_path
        self.n_neg = n_neg

    def __call__(self, *args, **kwargs):
        dataset = self.load_mm_dataset(self.data_dir_path, self.n_neg)
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
        inputs = list(zip(example_batch['word'], example_batch['definition']))
        features = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=self.max_length,
            padding='max_length',
            truncation='longest_first'
        )
        features["labels"] = example_batch["label"]
        return features

    @staticmethod
    def load_mm_dataset(data_dir_path: Text, n_neg: int):
        train = pd.read_csv(os.path.join(data_dir_path, f'train-n_neg{n_neg}.tsv'), sep='\t')
        dev = pd.read_csv(os.path.join(data_dir_path, f'dev-n_neg{n_neg}.tsv'), sep='\t')
        
        train = train.dropna()
        dev = dev.dropna()        
        
        train = Dataset.from_dict({key: train[key].tolist() for key in train.keys()})
        train.features['label'] = ClassLabel(num_classes=2, names=["False", "True"])

        dev = Dataset.from_dict({key: dev[key].tolist() for key in dev.keys()})
        dev.features['label'] = ClassLabel(num_classes=2, names=["False", "True"])

        outp_dict = DatasetDict(
            {
                "train": train,
                "validation": dev,
            }
        )
        return outp_dict


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/electra-small-generator')
    module = MeaningMatchingnDataModule(tokenizer, '../../data/meaning_matching')
    feature_dict = module()
