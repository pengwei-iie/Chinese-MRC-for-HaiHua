# coding:utf-8
import numpy as np
import json
import os
import pandas as pd
from sklearn.utils import shuffle

csv_file = './result-final.csv'
output_file = './result-final-label.csv'
with open(csv_file, 'r', encoding='gb18030') as infile:
    df = pd.read_csv(infile)
for i in range(len(df)):

    p1 = df['p1'][i]
    p2 = df['p2'][i]
    p3 = df['p3'][i]
    p4 = df['p4'][i]
    list_p = [p1, p2, p3, p4]
    max_index = (int)(np.argmax(list_p))
    df['label'][i] = chr(max_index+65)
df = df.drop(columns=['p1', 'p2', 'p3', 'p4'])
df.to_csv(output_file, index=False, header=True)
