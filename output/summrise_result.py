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
__date__ = "03 06 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"


import yaml
import os
import pandas as pd
import numpy as np

model_list = ['baseline', 'bert-base', 'albert-base', 'roberta-base', 'electra-small',
              'bert-large', 'albert-large', 'roberta-large', 'electra-large']

dir_path_list = ['exp1', 'exp2']


def process_summary(dir_path):
    all_output_dict = {
        "model_name": model_list
    }

    category_output_dict = {
    }

    for m in model_list:
        # load result
        with open(os.path.join(dir_path, f"{m}-result.yaml"), 'r') as loadFile:
            data = yaml.load(loadFile, Loader=yaml.SafeLoader)

        # all result summary
        for key, value in data['All'].items():
            new_key = key.replace("avg_", "")
            if all_output_dict.get(new_key) is None:
                all_output_dict[new_key] = [value]
            else:
                all_output_dict[new_key].append(value)

        # category result
        if m == 'baseline':
            continue

        for key in data.keys():
            if key == 'All':
                continue
            if key not in category_output_dict.keys():
                category_output_dict[key] = {}

            for metric_key, value in data[key].items():
                new_key = metric_key.replace("avg_", "")
                if category_output_dict[key].get(new_key) is None:
                    category_output_dict[key][new_key] = [value]
                else:
                    category_output_dict[key][new_key].append(value)

    # transform category_output_dict
    category_outp_new = {
        'category': []
    }
    for key in category_output_dict.keys():
        category_outp_new['category'].append(key)
        for metric_key, value in category_output_dict[key].items():
            if category_outp_new.get(metric_key):
                category_outp_new[metric_key].append(np.mean(value))
            else:
                category_outp_new[metric_key] = [np.mean(value)]
    return all_output_dict, category_outp_new


def main():
    for dir_path in dir_path_list:
        all_dict, category_dict = process_summary(dir_path)

        all_df = pd.DataFrame(all_dict)
        category_df = pd.DataFrame(category_dict)

        all_df.to_csv(os.path.join(f'{dir_path}_all_result.tsv'), sep='\t')
        category_df.to_csv(os.path.join(f'{dir_path}_category_result.tsv'), sep='\t')


if __name__ == '__main__':
    main()