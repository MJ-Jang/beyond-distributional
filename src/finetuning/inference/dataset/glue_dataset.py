# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Text


class GLUEAuToInferenceDataset(Dataset):

    task_name = 'glue'
    padding = 'max_length'
    max_length = 128
    truncation = 'longest_first'

    def __init__(
            self,
            tokenizer,
            data_type: Text = 'test',
    ):
        assert data_type in ['test', 'validation', 'validation_matched', 'validation_mismatched']

        data_ = load_dataset('glue', self.task_name)
        data = data_[data_type]

        input_1, input_2 = self.process_input(data)

        self.label = data['label']
        if input_2 is None:
            self.input_encodes = tokenizer(
                input_1,
                padding=self.padding,
                max_length=self.max_length,
                truncation=self.truncation
            )
        else:
            self.input_encodes = tokenizer(
                input_1,
                input_2,
                padding=self.padding,
                max_length=self.max_length,
                truncation=self.truncation
            )

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        outputs = {key: torch.LongTensor(value[item]) for key, value in self.input_encodes.items()}
        outputs['labels'] = torch.LongTensor([self.label[item]])
        return outputs

    @staticmethod
    def process_input(data):
        raise NotImplementedError


class MNLIAutoInferenceDataset(GLUEAuToInferenceDataset):
    task_name = 'mnli'

    @staticmethod
    def process_input(data):
        return data['hypothesis'], data['premise']


class MRPCAutoInferenceDataset(GLUEAuToInferenceDataset):
    task_name = 'mrpc'

    @staticmethod
    def process_input(data):
        return data['sentence1'], data['sentence2']


class QNLIAutoInferenceDataset(GLUEAuToInferenceDataset):
    task_name = 'qnli'

    @staticmethod
    def process_input(data):
        return data['question'], data['sentence']


class QQPAutoInferenceDataset(GLUEAuToInferenceDataset):
    task_name = 'qqp'

    @staticmethod
    def process_input(data):
        return data['question1'], data['question2']


class RTEAutoInferenceDataset(GLUEAuToInferenceDataset):
    task_name = 'rte'

    @staticmethod
    def process_input(data):
        return data['sentence1'], data['sentence2']


class SSTAutoInferenceDataset(GLUEAuToInferenceDataset):
    task_name = 'rte'

    @staticmethod
    def process_input(data):
        return data['sentence'], None