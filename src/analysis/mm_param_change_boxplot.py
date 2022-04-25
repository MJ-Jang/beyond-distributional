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
__date__ = "22 07 2021"
__author__ = "Myeongjun Jang"
__maintainer__ = "Myeongjun Jang"
__email__ = "myeongjun.jang@cs.ox.ac.uk"
__status__ = "Development"

import json
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


with open('./output/mm_experiment/model_diff_outputs.json', 'r') as readFile:
    score_dict = json.load(readFile)


new_dict = {
    "avg. Frobenius Norm": [],
    "Model": [],
    "Model size": []
}
for key, value in score_dict.items():
    if key.startswith('albert'):
        continue
    # value = ["{:e}".format(v) for v in value]
    new_dict['avg. Frobenius Norm'] += value
    model_ = [key.split('-')[0]] * len(value)
    model_size_ = [key.split('-')[1]] * len(value)

    new_dict['Model'] += model_
    new_dict['Model size'] += model_size_

new_df = pd.DataFrame(new_dict)

plt.figure(figsize=(8, 6), dpi=80)
ax = sn.boxplot(x="Model", y="avg. Frobenius Norm", hue="Model size", data=new_df, palette="Set3")
plt.xticks(fontsize=20)
plt.xlabel('Model', fontsize=25)
plt.ylabel('avg. Frobenius Norm', fontsize=25)
plt.legend(title="Model size", fontsize=20, title_fontsize=20)
# plt.show()
# plt.close()

plt.savefig("model_diff.png")
