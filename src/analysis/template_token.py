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
__date__ = "21 06 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"


import yaml
import json
import os
import re
import numpy as np

category_dict = {
    "X is the opposite of Y.": " opposite of ",
    "X is different from Y.": " different from ",
    "X is an antonym of Y.": " antonym of ",
    "X is a synonym of Y.": " synonym of ",
    "X is another form of Y.": " another form of ",
    "X is a rephrasing of Y.": " rephrasing of "
}

ANTONYM_TEMPLATE = ['X is an antonym of Y.', "X is the opposite of Y.", "X is different from Y."]
SYNONUM_TEMPLATE = ['X is a synonym of Y.', "X is another form of Y.", "X is a rephrasing of Y."]

data_dir = './data'
output_dir = './output/exp1'
save_dir = './output/template_analysis'
metric = 'HR@1'


def min_max_scale(list):
    max_ = max(list)
    min_ = min(list)
    outp = [(x - min_)/(max_ - min_) for x in list]
    return outp


with open(os.path.join(data_dir, 'template_counts.json'), 'r') as readFile:
    counts = json.load(readFile)

counts['total'] = {}
for key in counts['wiki'].keys():
    counts['total'][key] = counts['wiki'][key] + counts['bookcorpus'][key]

import matplotlib.pyplot as plt

x_names = list(category_dict.keys())
n_groups = len(x_names)
index = np.arange(n_groups)
bar_width = 0.4
opacity = 0.4

# get all values
file_list = [f for f in os.listdir(output_dir) if f.endswith('.yaml') and 'baseline' not in f]
outp = []
for fl_ in file_list:
    with open(os.path.join(output_dir, fl_), 'r') as readFile:
        result = yaml.load(readFile, Loader=yaml.SafeLoader)
    hr = [result[x_][f'avg_{metric}'] for x_ in x_names]
    outp.append(hr)

outp_tr = np.transpose(outp)
ant_perf, ant_cnt = list(), list()
syn_perf, syn_cnt = list(), list()
for i, x_ in enumerate(x_names):
    cnt_ = counts['total'][category_dict[x_]]
    perf_ = outp_tr[i]
    if x_ in ANTONYM_TEMPLATE:
        ant_perf += perf_.tolist()
        ant_cnt += [1/cnt_] * len(perf_)
    else:
        syn_cnt += perf_.tolist()
        syn_perf += [1/cnt_] * len(perf_)

from scipy.stats import pearsonr, spearmanr
pearsonr(ant_perf, ant_cnt)
pearsonr(syn_perf, syn_cnt)



# Draw graph
mean_hr = np.array(outp).mean(axis=0).tolist()

for hr, x in zip(mean_hr, x_names):
    print(x, hr)

cnt_y = min_max_scale([counts['total'][category_dict[x_]] for x_ in x_names])

# Antonym
hr_ant = mean_hr[:int(n_groups/2)]
plt.scatter(index[:len(hr_ant)], hr_ant, alpha=1.0, color='r', label=f'{metric} of antonym questions')
plt.bar(index[:len(hr_ant)], cnt_y[:len(hr_ant)], alpha=opacity, color='r', label='# of antonym templates')

# synonym
hr_syn = mean_hr[int(n_groups/2):]
plt.scatter(index[len(hr_syn):], hr_syn, alpha=1.0, color='b', label=f'{metric} of synonym questions')
plt.bar(index[len(hr_syn):], cnt_y[len(hr_syn):], alpha=opacity, color='b', label='# of syonym templates')

x_tickts = [re.sub(pattern='X is| the | a | Y.', string=s, repl='').strip() for s in x_names]
plt.xticks(index, x_tickts, fontsize=10, rotation=20)
plt.legend(loc='upper right',prop={'size': 10})
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, f'{metric}_count_plot.png'))
plt.close()

