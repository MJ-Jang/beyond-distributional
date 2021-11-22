# -*- coding: utf-8 -*-

import torch
import os

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
    task_name = 'sst2'

    @staticmethod
    def process_input(data):
        return data['sentence'], None


class NegRTEAutoInferenceDataset(GLUEAuToInferenceDataset):
    task_name = 'neg_rte'

    def __init__(
            self,
            tokenizer,
            data_type: Text = 'test',
    ):
        super(NegRTEAutoInferenceDataset,).__init__(tokenizer, data_type)
        assert data_type in ['test', 'validation', 'validation_matched', 'validation_mismatched']

        PWD = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(PWD, 'neg_nli/RTE.txt')

        data = {
            "sentence1": [],
            "sentence2": [],
            "label": []
        }
        with open(file_name, 'r', encoding='latin-1') as readFile:
            for line in readFile.readlines():
                idx_, sent1_, sent2_, label_ = line.split('\t')
                if label_.strip() == 'gold_label':
                    continue
                label_ = 0 if label_.strip() == 'entailment' else 1
                data['sentence1'].append(sent1_.strip())
                data['sentence2'].append(sent2_.strip())
                data['label'].append(label_)

        self.label = data['label']
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

    @staticmethod
    def process_input(data):
        return data['sentence1'], data['sentence2']


class NegMNLIAutoInferenceDataset(GLUEAuToInferenceDataset):
    task_name = 'neg_mnli'

    def __init__(
            self,
            tokenizer,
            data_type: Text = 'test',
    ):
        super(NegMNLIAutoInferenceDataset, ).__init__(tokenizer, data_type)
        assert data_type in ['test', 'validation', 'validation_matched', 'validation_mismatched']

        PWD = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(PWD, 'neg_nli/MNLI.txt')

        data = {
            "premise": [],
            "hypothesis": [],
            "label": []
        }
        with open(file_name, 'r', encoding='latin-1') as readFile:
            for line in readFile.readlines():
                idx_, premise_, hypothesis_, label_ = line.split('\t')
                if label_.strip() == 'gold_label':
                    continue
                label_ = 0 if label_.strip() == 'entailment' else 1 if label_.strip() == 'neutral' else 2
                data['premise'].append(premise_.strip())
                data['hypothesis'].append(hypothesis_.strip())
                data['label'].append(label_)

        self.label = data['label']
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

    @staticmethod
    def process_input(data):
        return data['hypothesis'], data['premise']