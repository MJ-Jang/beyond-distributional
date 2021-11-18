# -*- coding: utf-8 -*-

import pandas as pd
import os
import torch
import yaml
import re
import argparse
import json

from typing import List, Text, Dict
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from data import FinetuneDataLoader, FinetuneDataMudule
from util import load_model

# DATA_LIST = ["kobest_boolq", "kobest_copa", "kobest_wic"]


class InferenceDataLoader(FinetuneDataLoader):

    def __init__(self, data_name: Text, data_type: Text):
        self.data_name = data_name
        self.data_type = data_type
        assert data_type in ['test', 'validation']

    def __call__(self, *args, **kwargs):
        print(f"Load data from {self.kobest_path}...")
        # load data
        dataset_dict = {
            "kobest_boolq": self.load_kobest_boolq(),
            "kobest_copa": self.load_kobest_copa(),
            "kobest_wic": self.load_kobest_wic(),
        }

        assert self.data_name in list(dataset_dict.keys())
        return dataset_dict[self.data_name]

    @staticmethod
    def generate_new_dataset(test_inp, test_label):

        test_dict = {
            "input": test_inp,
            "label": test_label
        }

        outp = DatasetDict(
            {
                "test": Dataset.from_dict(test_dict),
            }
        )
        return outp

    def load_kobest_boolq(self):
        # load dataset
        df = pd.read_csv(os.path.join(self.kobest_path, self.cfg.get("BoolQ").get(self.data_type)),
                         sep='\t', encoding='utf-8')
        df = df.rename(columns={
            'Answer(FALSE = 0, TRUE = 1)': 'label',
            'Question': '질문',
            'Text': '예시문'
        })

        test = Dataset.from_pandas(df)

        # process inputs
        test_inputs = self.input_preprocess(test, 'kobest_boolq')

        # process labels
        label_dict = {
            0: "false",
            1: "true"
        }  # free-text label

        test_labels = [int(i) for i in test['label']]

        outp = self.generate_new_dataset(test_inputs, test_labels)
        return outp

    def load_kobest_copa(self):
        # load dataset
        df = pd.read_csv(os.path.join(self.kobest_path, self.cfg.get("COPA").get(self.data_type)),
                         sep='\t', encoding='utf-8')
        # change column name
        df = df.rename(columns={
            'sentence': '예시문',
            'question': '질문',
            '1': '문장1',
            '2': '문장2',
            'Answer': 'label'
        })

        test = Dataset.from_pandas(df)

        # process inputs
        test_inputs = self.input_preprocess(test, 'kobest_copa')

        # process labels
        label_dict = {
            1: "문장1",
            2: "문장2",
        }  # free-text label

        test_labels = [int(i)-1 for i in test['label']]

        outp = self.generate_new_dataset(test_inputs, test_labels)
        return outp

    def load_kobest_wic(self):
        # load dataset
        df = pd.read_csv(os.path.join(self.kobest_path, self.cfg.get("WiC").get(self.data_type)),
                         sep='\t', encoding='utf-8')
        # change column name
        df = df.rename(columns={
            'Target': '단어',
            'SENTENCE1': '문장1',
            'SENTENCE2': '문장2',
            'ANSWER': 'label'
        })

        test = Dataset.from_pandas(df)

        # process inputs
        test_inputs = self.input_preprocess(test, 'kobest_wic')

        # process labels
        label_to_idx = {
            "False": 0,
            "True": 1
        }

        test_labels = [label_to_idx[str(i)] for i in test['label']]

        outp = self.generate_new_dataset(test_inputs, test_labels)
        return outp


class Inferencer:

    def __init__(
            self,
            model,
            batch_size: int = 64,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.batch_size = batch_size
        self.model.to(self.device)

    def __call__(self, dataset) -> List:
        outp = []
        for start in tqdm(range(0, len(dataset), self.batch_size)):
            batch = dataset[start:start+self.batch_size]

            inputs = dict()
            for key, value in batch.items():
                if key == 'labels':
                    continue
                inputs[key] = value.to(self.device)

            logits = self.model(**inputs)['logits']
            preds_ = logits.argmax(dim=-1)
            preds_ = preds_.detach().cpu().tolist()
            outp += preds_
        return outp


def postprocess_outp(predictions: List):
    for i,  p in enumerate(predictions):
        predictions[i] = re.sub(pattern='</?\\w+>', string=p, repl='').strip()
    return predictions


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9


def restore_predictions(preds: List, data_name: Text):
    if data_name == 'kobest_boolq':
        return preds
    elif data_name == 'kobest_copa':
        return [p+1 for p in preds]
    elif data_name == 'kobest_wic':
        idx_to_label = {
            0: "False",
            1: "True"
        }
        return [idx_to_label[p] for p in preds]
    else:
        raise NotImplementedError


def main(args):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dir_path, 'config.yaml'), 'r') as readFile:
        config_file = yaml.load(readFile, Loader=yaml.SafeLoader)
    cfg = config_file.get('cfg')
    rsc = config_file.get('resource')

    print("Loading model...")
    tokenizer, model = load_model(args.model_name, rsc)
    model.eval()

    print(f"Model is Loaded!")
    print(f"Name: {args.model_name}")
    n_params = count_parameters(model)
    if n_params < 0:
        n_params = f"{int(n_params * 1000)}M"
    else:
        n_params = f"{round(n_params,1)}B"
    print(f"# of parameters: {n_params}")

    output = dict()
    for data_name in DATA_LIST:
        model_name = args.model_name.split('/')[-1]
        save_prefix = f"{model_name}-{data_name}-ft"
        model.load_state_dict(torch.load(f'{cfg.get("output_dir")}/{save_prefix}.ckpt'))

        if tokenizer.pad_token is None or model.config.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        print("Model is loaded succesfully!!")
        data_module = FinetuneDataMudule(
            InferenceDataLoader(data_name, args.data_type),
            tokenizer,
            input_max_len=cfg.get('input_max_len'),
            target_max_len=cfg.get('target_max_len'),
        )
        test_dataset = data_module.build_dataset()

        print("Start Inference...")
        inferencer = Inferencer(model, batch_size=8)
        pred = inferencer(test_dataset["test"])

        label = test_dataset['test']['label']
        acc = accuracy_score(label, pred)
        print(f"Data name: {data_name} | Acc: {acc}")

        predictions = restore_predictions(pred, data_name)
        output[data_name] = predictions

    save_dir = os.path.join(dir_path, "../result", args.repeats)
    os.makedirs(save_dir, exist_ok=True)
    save_file_name = f"{args.model_name}-single-ft-{args.data_type}.json"
    with open(os.path.join(save_dir, save_file_name), 'w', encoding='utf-8') as saveFile:
        json.dump(output, saveFile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='gpt3_1B',
                        choices=['gpt3_125M', 'gpt3_1B', 'koelectra', 'kobert', 'kobart'])
    parser.add_argument('--data_type', type=str, default='test',
                        choices=['test', 'validation'])
    parser.add_argument('--repeats', type=str, default='1')
    args = parser.parse_args()

    main(args)
