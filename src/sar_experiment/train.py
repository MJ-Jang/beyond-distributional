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


# -*- coding: utf-8 -*-
import transformers
import os
import torch
import argparse
import yaml
import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer, EarlyStoppingCallback, AutoModelForSequenceClassification
from data import SemanticIdentificationDataModule


metric = load_metric("accuracy")


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


def save_state_dict(model, save_path: str, save_prefix: str):
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, save_prefix + '.ckpt')
    model = model.cpu()
    torch.save(model.state_dict(), filename)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def freeze_encoder(model):
    if model.base_model_prefix == 'roberta':
        for param in model.roberta.parameters():
            param.requires_grad = False
    if model.base_model_prefix == 'bert':
        for param in model.bert.parameters():
            param.requires_grad = False
    if model.base_model_prefix == 'albert':
        for param in model.albert.parameters():
            param.requires_grad = False
    if model.base_model_prefix == 'electra':
        for param in model.electra.parameters():
            param.requires_grad = False
    return model


def load_plm_state_dict(file_name, plm_name):
    aa = torch.load(file_name)
    new_dict = {}
    for key in aa.keys():
        if key.startswith(plm_name):
            new_dict[key.replace(f"{plm_name}.", "")] = aa[key]
    return new_dict


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
        file_path = os.path.join(dir_path, "../meaning_match_experiment/model_binary/", f"{args.backbone_model_name}.ckpt")

        if backbone_model.startswith("roberta"):
            model.roberta.load_state_dict(load_plm_state_dict(file_path, 'roberta'))
        elif backbone_model.startswith('electra'):
            model.electra.load_state_dict(load_plm_state_dict(file_path, 'electra'))
        elif backbone_model.startswith('bert'):
            model.bert.load_state_dict(load_plm_state_dict(file_path, 'bert'))
        elif backbone_model.startswith('albert'):
            model.albert.load_state_dict(load_plm_state_dict(file_path, 'albert'))
        else:
            raise NotImplementedError
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.backbone_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(args.backbone_model_name)

    if args.freeze_enc:
        model = freeze_encoder(model)

    data_dir_path = os.path.join(dir_path, '../../data/SAR')

    data_module = SemanticIdentificationDataModule(
        tokenizer,
        data_dir_path=data_dir_path,
        is_balanced=args.is_balanced,
        max_length=cfg.get('max_length'),
    )
    feature_dict = data_module()

    train_dataset = feature_dict['train']
    eval_dataset = feature_dict['validation']

    if 'large' in args.backbone_model_name:
        batch_size = cfg.get("batch_size_large")
    else:
        batch_size = cfg.get("batch_size_base")

    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(
            output_dir=os.path.join(dir_path, cfg.get('output_dir')),
            overwrite_output_dir=True,
            learning_rate=float(cfg.get('learning_rate')),
            do_train=True,
            num_train_epochs=cfg.get('epochs'),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            metric_for_best_model='accuracy',
            load_best_model_at_end=True,
            greater_is_better=True,
            evaluation_strategy=transformers.IntervalStrategy('epoch')
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.get('patience'))]
    )

    trainer.train()

    trained_model = trainer.model

    model_name = args.backbone_model_name.split('/')[-1]
    save_prefix = f"sar-{model_name}-balanced_{args.is_balanced}-freeze_{args.freeze_enc}"
    save_state_dict(trained_model, os.path.join(dir_path, './model_binary'), save_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone_model_name', type=str, default='albert-base')
    parser.add_argument('--freeze_enc', action='store_true')
    parser.add_argument('--is_balanced', action='store_true')

    args = parser.parse_args()
    main(args)