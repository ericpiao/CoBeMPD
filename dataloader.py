import os
import json

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
import warnings
from tqdm import tqdm
import ast
from itertools import zip_longest
warnings.filterwarnings("ignore")


def lines_to_block_ids(lines,tokenizer, block_num=20, block_contain_nums=5, padding_value=0,word_length = 512):
    block_lines = [list(group) for group in zip_longest(*[iter(lines)]*block_contain_nums, fillvalue='')]
    block_word = [ '\n'.join(str(x) for x in block if x != '')  for block in block_lines]
    word_ids = []
    word_mask = []
    if block_num - len(block_word) > 0:
        block_word.extend([''] * (block_num - len(block_word)))
    else:
        block_word = block_word[:20]
    for word in block_word:
        result = tokenizer.encode_plus(word, max_length=word_length , padding='max_length', truncation=True, return_attention_mask=True)
        word_ids.append(result['input_ids'])
        word_mask.append(result['attention_mask'])
    return torch.tensor(word_ids), torch.tensor(word_mask)

def project_to_file_ids(files_map, tokenizer, files_n=20, word_length=512):
    code_list = []
    code_ids = []
    code_mask = []
    file_list = []
    file_mask = []  # 新增：文件级别掩码，1=真实文件，0=padding

    try:
        for key in files_map:
            code = '\n'.join(files_map[key]['addition']) + '\n' + files_map[key]['content']
            code_list.append(code)
            file_list.append(key)
            file_mask.append(1)  # 真实文件

        # 处理补齐
        pad_len = files_n - len(code_list)
        file_mask.extend([0] * (files_n - len(file_list)))
        if pad_len > 0:
            code_list.extend([''] * pad_len)
        else:
            code_list = code_list[:files_n]
            file_list = file_list[:files_n]
            file_mask = file_mask[:files_n]

        for code in code_list:
            result = tokenizer.encode_plus(
                code,
                max_length=word_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )
            code_ids.append(result['input_ids'])
            code_mask.append(result['attention_mask'])

    except Exception as e:
        print(e)

    return (
        torch.tensor(code_ids),         # shape: (files_n, word_length)
        torch.tensor(code_mask),        # shape: (files_n, word_length)
        torch.tensor(file_mask),        # shape: (files_n,)
        file_list
    )


class CodeDataset(Dataset):
    def __init__(self, path, tokenizer_path):
        super().__init__()
        self.x = []
        self.y = []
        self.path = path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        for json_file in tqdm(os.listdir(path), ncols=100):
            try:
                with open(path + "\\" + json_file, "r", encoding="utf-8") as file:
                    project_data = json.load(file)
                    if len(project_data['file'])<=20:
                        files_ids, code_mask, file_mask,file_name = project_to_file_ids(project_data['file'], tokenizer, files_n = 20)
                        y = torch.tensor(project_data['label'], dtype=torch.float32)
                        features = {'input_ids': files_ids, 'attention_mask': code_mask,
                                    'file_mask': file_mask,
                                    'name': project_data['project'],
                                    'file_name':project_data['file']}
                        self.x.append(features)
                        self.y.append(y)
                    elif project_data['label'] == 0:
                        file_names = list(project_data['file'].keys())
                        for i in range(0, len(project_data['file']), 20):
                            tmp =  {k: project_data['file'][k] for k in file_names[i:i+20]}
                            files_ids, code_mask, file_mask,file_name = project_to_file_ids(tmp, tokenizer, files_n=20)
                            y = torch.tensor(project_data['label'], dtype=torch.float32)
                            features = {'input_ids': files_ids, 'attention_mask':code_mask,
                                        'file_mask': file_mask,
                                        'name': project_data['project'],
                                        'file_name':file_name}
                            self.x.append(features)
                            self.y.append(y)
                    else:
                        file_names = list(project_data['file'].keys())
                        for i in range(0, len(project_data['file']), 20):
                            tmp = {k: project_data['file'][k] for k in file_names[i:i+20]}
                            y = [project_data['file'][item]['label'] for item in file_names[i:i+20]]
                            files_ids, code_mask, file_mask,file_name = project_to_file_ids(tmp, tokenizer, files_n=20)
                            features = {'input_ids': files_ids, 'attention_mask': code_mask,
                                        'file_mask': file_mask,
                                        'name': project_data['project'],
                                        'file_name':file_name}
                            self.x.append(features)
                            self.y.append(max(y))
            except Exception as e:
                print(json_file +':'+ str(e))

    def __len__(self):
        return len(self.x)  # 返回数据集的大小

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



# dataloader = CodeDataset(r'D:\zzl\MutiHunter\result_data\filelabel', r'D:\zzl\PypiHunter\module\codebert\\')
# #
# torch.save(dataloader, 'dataloader1.pth')
#
# dataloader = CodeDataset(r'D:\zzl\PypiHunter\new_data\benign', r'D:\zzl\PypiHunter\module\codebert\\')
# #
# torch.save(dataloader, 'dataloader2.pth')