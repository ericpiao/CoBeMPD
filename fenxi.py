import os

import pandas as pd
import json
import torch
from tqdm import tqdm

from dataloader import CodeDataset
from torch.utils.data import ConcatDataset

result_df = pd.read_csv('./test.csv',index_col=0)
result_df = result_df.drop_duplicates('package')
# result_df = result_df.dropna()
print(result_df)
df = pd.read_csv(f'.\\data\\csv_data\\pypi to pypi-1.csv', names=['package', 'label'])
df = df.drop_duplicates('package')
# print(df)
test_package_name = list(df['package'])
train_dataset = []
test_dataset = []
files = result_df['file'].dropna()
files_list = []
for item in files:
    split_items = [path.strip() for path in item.split(',')]
    files_list.extend(split_items)
print(files_list)
print(len(files_list))
t_df = pd.read_csv(f'.\\label.csv',index_col=0)
t_df = t_df.drop_duplicates('file')
tp,fp,tn,fn,c = 0,0,0,0,t_df.count()
for item in files_list:
    t = t_df[t_df['file'] == item]['label'].values
    if 1 == t:
        tp += 1
    else:
        print(item)
        fp += 1
print(tp)
print(fp)
# dataset = ConcatDataset([malicious_dataset_train, benign_dataset_train,malicious_dataset_lack])
# a = list()
# for item in dataset:
#     if item[0]['name'] in test_package_name:
#         a.append(item[0]['name'])
#     else:
#         pass
# for item in set(df['package'])-set(a):
#     print(item)
# print(set(df['package'])-set(result_df['package']))
# path = r'D:\zzl\MutiHunter\result_data\filelabel'
# tp = 0
# fp = 0
# f_n = 0
# print(len(result_df))
# files = []
# y = []
# for item in df.iterrows():
#     name = item[1]['package']
#     if os.path.exists(path + "\\" + name+'.json'):
#         try:
#             with open(path + "\\" + name+'.json', "r", encoding="utf-8") as file:
#                 project_data = json.load(file)
#                 for item in project_data['file']:
#                     y.append(project_data['file'][item]['label'])
#                     files.append(item)
#         except Exception as e:
#             pass
# a_df = pd.DataFrame({'file': files, 'label': y})
# a_df.to_csv('./label.csv')


# for item_row in result_df.iterrows():
#     name,y = item_row[1]['package'],item_row[1]['y']
#     try:
#         if os.path.exists(path + "\\" + name +'.json'):
#             with open(path + "\\" + name +'.json', "r", encoding="utf-8") as file:
#                 project_data = json.load(file)
#                 f_n += len(project_data['file'])
#                 flag = 0
#                 for item in project_data['file']:
#                     if 1 == project_data['file'][item]['label']:
#                         tp += 1
#                         flag = 1
#                     else:
#                         fp += 1
#                 if flag == 0:
#                     print(project_data['name'])
#     except Exception as e:
#         pass
# print(tp,fp,f_n)
