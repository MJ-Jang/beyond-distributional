# -*- coding: utf-8 -*-

import torch
import os
import argparse
import json

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5TokenizerFast, T5ForConditionalGeneration
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef


PWD = os.path.dirname(os.path.abspath(__file__))
GLUE_TASK_LIST = ['rte', 'mrpc', 'mnli', 'qnli', 'qqp', 'sst', 'cola']


class AutoModelInferencer:

    def __init__(
            self,
            model,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model

        self.model.to(self.device)

    def __call__(self, data_loader):
        outp, probs = [], []
        for batch in tqdm(data_loader):
            inputs = dict()
            for key, value in batch.items():
                if key == 'labels':
                    continue
                inputs[key] = value.to(self.device)

            logits = self.model(**inputs)['logits']
            preds = logits.argmax(dim=-1)
            preds = preds.detach().cpu().tolist()
            outp += preds

            prob = torch.nn.Softmax(dim=-1)(logits).detach().cpu().tolist()
            probs += prob
        return outp, probs


def prepare_model(args):
    n_class_dict = {
        "mnli": 3,
        "rte": 2,
        "qqp": 2,
        "qnli": 2,
        "mrpc": 2,
        "sst": 2,
        "snli": 3,
        "mnli_neg": 3,
        "rte_neg": 2,
        "snli_neg": 3
    }

    print(f'model type: {args.model_type}')
    model = AutoModelForSequenceClassification.from_pretrained(args.model_type,
                                                               num_labels=n_class_dict.get(args.dataset))
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    return model, tokenizer


def prepare_data(args, tokenizer):
    from dataset import MRPCAutoInferenceDataset, MNLIAutoInferenceDataset, QQPAutoInferenceDataset,\
    QNLIAutoInferenceDataset, RTEAutoInferenceDataset, SSTAutoInferenceDataset, NegRTEAutoInferenceDataset,\
    NegMNLIAutoInferenceDataset,  NegSNLIAutoInferenceDataset, SNLIAutoInferenceDataset, COLAAutoInferenceDataset

    if args.dataset == 'rte':
        return RTEAutoInferenceDataset(tokenizer, data_type=args.data_type)

    elif args.dataset == 'mrpc':
        return MRPCAutoInferenceDataset(tokenizer, data_type=args.data_type)

    elif args.dataset == 'mnli':
        return MNLIAutoInferenceDataset(tokenizer, data_type=args.data_type)

    elif args.dataset == 'qnli':
        return QNLIAutoInferenceDataset(tokenizer, data_type=args.data_type)

    elif args.dataset == 'qqp':
        return QQPAutoInferenceDataset(tokenizer, data_type=args.data_type)

    elif args.dataset == 'sst':
        return SSTAutoInferenceDataset(tokenizer, data_type=args.data_type)

    elif args.dataset == 'cola':
        return COLAAutoInferenceDataset(tokenizer, data_type=args.data_type)

    elif args.dataset == 'snli':
        return SNLIAutoInferenceDataset(tokenizer, data_type=args.data_type)

    elif args.dataset == 'rte_neg':
        return NegRTEAutoInferenceDataset(tokenizer, data_type=args.data_type)

    elif args.dataset == 'mnli_neg':
        return NegMNLIAutoInferenceDataset(tokenizer, data_type=args.data_type)

    elif args.dataset == 'snli_neg':
        return NegSNLIAutoInferenceDataset(tokenizer, data_type=args.data_type)
    else:
        raise NotImplementedError


def load_model_from_statedict(model, args):
    """
    This function is implemented to match the keys in state dict
    """
    normalized_dataset = args.dataset.split('_')[0]
    if torch.cuda.is_available():
        print('> load model path : ', os.path.join(args.dir_path, args.model_dir,
                                                   f'{args.model_type}-{normalized_dataset}.ckpt'))
        savefile = torch.load(os.path.join(args.dir_path, args.model_dir,
                                           f'{args.model_type}-{normalized_dataset}.ckpt'))
    else:
        savefile = torch.load(os.path.join(args.dir_path, args.model_dir,
                                           f'{args.model_type}-{normalized_dataset}.ckpt'),
                              map_location=torch.device('cpu'))
    new_state_dict = {}
    for key, value in savefile['state_dict'].items():
        new_state_dict[key.replace('model.', '')] = value # match keys
    model.load_state_dict(new_state_dict)
    return model


def main(args):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    args.dir_path = dir_path

    print("Loading models...")
    model, tokenizer = prepare_model(args)

    print("Loading data...")
    dataset = prepare_data(args, tokenizer)
    data_loader = DataLoader(dataset, batch_size=args.batch_size)

    model = load_model_from_statedict(model, args)
    inferencer = AutoModelInferencer(model)

    model.eval()

    predictions, probs = inferencer(data_loader)
    perf_dict = {}
    if 'validation' in args.data_type:
        acc = accuracy_score(y_true=dataset.label, y_pred=predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=dataset.label,
            y_pred=predictions,
            average='weighted'
        )
        perf_dict['accuracy'] = acc
        perf_dict['precision'] = precision
        perf_dict['recall'] = recall
        perf_dict['f1'] = f1
        if args.dataset == 'cola':
            perf_dict['matthews_cor'] = matthews_corrcoef(y_true=dataset.label, y_pred=predictions)
        print(f"{args.model_type}|{args.dataset}| Accuracy: {acc}")

    outputs = {
        "idx": [i for i in range(len(predictions))],
        "preds": predictions,
    }
    if perf_dict:
        outputs.update(perf_dict)

    save_path = os.path.join(dir_path, args.save_dir, args.model_type.split('/')[-1])
    os.makedirs(save_path, exist_ok=True)

    file_name = f"{args.dataset}-{args.data_type}.json"
    with open(os.path.join(save_path, file_name), 'w') as saveFile:
        json.dump(outputs, saveFile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, default='bert-large-cased',
                        help='type or pre-trained models')
    parser.add_argument('--dataset', type=str, default='mnli',
                        help='finetuning task name')
    parser.add_argument('--data_type', type=str, default='test', choices=['validation', 'validation_matched',
                                                                          'validation_mismatched'],
                        help='type of data for inference')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='size of batch for inference')
    parser.add_argument('--save_dir', type=str, default='../result/',
                        help='directory to save results')
    parser.add_argument('--model_dir', type=str, default='../../model_binary/',
                        help='directory path where binary file is saved')
    args = parser.parse_args()

    main(args)
