# coding:utf-8
import numpy as np
import json
import os
import pandas as pd
from sklearn.utils import shuffle

chn_file = './validation.json'
out_file = './Task_1_test.jsonl'

with open(chn_file, 'r', encoding='utf-8') as f:
    trans_exs = []
    examples = json.load(f)
    count_len = 0
    for line in examples:
        id = line['ID']
        article = line['Content']
        count_len += len(article)
        for i in range(len(line['Questions'])):
            if len(line['Questions'][i]['Choices']) <= 2:
                print(line['Questions'][i]['Q_id'])
                continue
            question = line['Questions'][i]['Question']
            if '（）' not in question:
                question = question + " （）"
            else:
                question = question.replace("（）", " （）")
            option_1 = line['Questions'][i]['Choices'][0]
            option_2 = line['Questions'][i]['Choices'][1]
            option_3 = line['Questions'][i]['Choices'][2]
            # try:
            #
            # except IndexError:
            #     print(id)
            if len(line['Questions'][i]['Choices']) < 4:
                option_4 = 'X'
            else:
                option_4 = line['Questions'][i]['Choices'][3]
            # answer = line['Questions'][i]['Answer']
            q_id = line['Questions'][i]['Q_id']
            # label = ord(answer) - 65
            ex = {
                "id": id,
                "article": article,
                "option_0": option_1,
                "option_1": option_2,
                "option_2": option_3,
                "option_3": option_4,
                "question": question,
                "q_id": q_id}
            # "label": label}
            trans_exs.append(ex)
    print(len(trans_exs))
    avg_len = count_len / len(trans_exs)
    print(avg_len)
    with open(out_file, 'w', encoding="utf-8") as fout:
        for ex in trans_exs:
            fout.write(json.dumps(ex, ensure_ascii=False) + '\n')