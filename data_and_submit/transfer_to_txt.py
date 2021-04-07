# coding:utf-8
import numpy as np
import json
import os
import pandas as pd
from sklearn.utils import shuffle

chn_file = './train.json'
out_file = './all_copurs.txt'

with open(chn_file, 'r', encoding='utf-8') as f:
    trans_all = ''
    examples = json.load(f)
    count_len = 0
    for line in examples:
        article = line['Content']
        count_len += len(article)
        trans_all += article
        for i in range(len(line['Questions'])):
            question = line['Questions'][i]['Question']
            for j in range(len(line['Questions'][i]['Choices'])):
                option_j = line['Questions'][i]['Choices'][j]
                option_j = ' ' + option_j
                trans_all += option_j
            # try:
            #
            # except IndexError:
            #     print(id)
    print(len(trans_all))
with open(out_file, 'w', encoding="utf-8") as fout:
    fout.write(trans_all + '\n')
