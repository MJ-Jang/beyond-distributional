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
__date__ = "19 07 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"

import pandas as pd
import numpy as np
import os
import json
import argparse
import re
from typing import Text, List
from tqdm import tqdm


def is_contain_number(word: Text):
    if re.findall(pattern='\\d+', string=word):
        return True
    return False


def extract_instance(df: pd.DataFrame):
    new_outp = []
    for id_, datum in tqdm(df.iterrows(), total=len(df)):
        subject, relation, object = datum['subject'], datum['relation'], datum['object']
        if is_contain_number(subject) or is_contain_number(object):
            continue
        elif len(subject) <= 2 or len(object) <= 2:
            continue
        elif subject == object:
            continue
        elif len(subject.split(' ')) > 3 or len(object.split(' ')) > 3:
            continue
        else:
            save_dict_inst = {
                    "subject": subject,
                    "relation": relation,
                    "object": object
                }
            new_outp.append(json.dumps(save_dict_inst))
    new_outp = list(set(new_outp))
    new_outp = [json.loads(d) for d in new_outp]
    return new_outp


def transform_to_df(input_list: List):
    outp_dict = {
        "word1": [],
        "word2": [],
        "label": [],
        "label_idx": []
    }

    label2idx = {
        "Antonym": 0,
        "Synonym": 1
    }
    for inst_ in input_list:
        outp_dict['word1'].append(inst_['subject'])
        outp_dict['word2'].append(inst_['object'])
        outp_dict['label'].append(inst_['relation'])
        outp_dict['label_idx'].append(label2idx[inst_['relation']])
    return pd.DataFrame(outp_dict)


def main(args):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    # ConceptNet version: 5.7.0
    conceptnet_path = os.path.join(dir_path, '../', 'conceptnet_en.tsv')
    conceptnet = pd.read_csv(conceptnet_path, sep='\t')

    ant = conceptnet[conceptnet['relation'] == 'Antonym'].dropna()
    syn = conceptnet[conceptnet['relation'] == 'Synonym'].dropna()

    new_ant = extract_instance(ant)
    new_syn = extract_instance(syn)
    
    # divide train / dev / test
    np.random.seed(1234)
    new_ant = np.random.permutation(new_ant)
    
    train_ant = new_ant[:-1500]
    dev_ant = new_ant[-1500:-1000]
    test_ant = new_ant[-1000:]

    np.random.seed(1234)
    new_syn = np.random.permutation(new_syn)
    
    if args.equal_train_size:
        train_syn = new_syn[:len(train_ant)]
    else:
        train_syn = new_syn[:-1500]
    dev_syn = new_syn[-1500:-1000]
    test_syn = new_syn[-1000:]

    # merge antonym and synonym
    train = train_ant.tolist() + train_syn.tolist()
    dev = dev_ant.tolist() + dev_syn.tolist()
    test = test_ant.tolist() + test_syn.tolist()

    np.random.shuffle(train)
    np.random.shuffle(dev)
    np.random.shuffle(test)

    print(f"Train | Synonym {len(train_syn)}, Antonym {len(train_ant)}")
    print(f"Dev | Synonym {len(dev_syn)}, Antonym {len(dev_ant)}")
    print(f"Test | Synonym {len(test_syn)}, Antonym {len(test_ant)}")

    train_df = transform_to_df(train)
    dev_df = transform_to_df(dev)
    test_df = transform_to_df(test)

    save_dir = os.path.join(dir_path, 'SEI_data')
    os.makedirs(save_dir, exist_ok=True)

    train_df.to_csv(os.path.join(dir_path, "train.tsv"), sep='\t', index=False, encoding='utf-8')
    dev_df.to_csv(os.path.join(dir_path, "dev.tsv"), sep='\t', index=False, encoding='utf-8')
    test_df.to_csv(os.path.join(dir_path, "test.tsv"), sep='\t', index=False, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--equal_train_size', action='store_true')

    args = parser.parse_args()

    main(args)