import argparse

import pandas as pd
from tqdm import tqdm

from MIL.MC_ATT_MIL.MC_MAX_MLPModel import MILConfidenceMaxPool
from model import MILModel
from meanmodel import MeanMILModel
from torch.utils.data import random_split
import warnings
import torch
import numpy as np
from line20_dataloader import CodeDataset
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
warnings.filterwarnings("ignore")
y_list = []
x_list = []
file_list = []

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1999, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--lr', type=float, default=5e-6, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--data_loader_worker_n', type=int, default=0, help='data_loader_worker_n')
parser.add_argument('--transformer_layer_name', type=str, default=r'D:\zzl\PypiHunter\module\codebert')
# 获取参数值
args = parser.parse_args()
epochs = args.epochs
malicious_dataset_train = torch.load(r'./dataloader1.pth')
# malicious_dataset_lack = torch.load(r'./dataloader_lack.pth')
benign_dataset_train = torch.load(r'./dataloader2.pth')
df = pd.read_csv(f'.\\data\\csv_data\\pypi to pypi-1.csv',names=['package','label'])
test_package_name = list(df['package'])
train_dataset = []
test_dataset = []
test_dataset_name = []
dataset = ConcatDataset([malicious_dataset_train,benign_dataset_train])
for item in dataset:
    if item[0]['name'] in test_package_name and item[0]['name'] not in  test_dataset_name:
        test_dataset.append(item)
        test_dataset_name.append(item[0]['name'])
    else:
        train_dataset.append(item)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                          num_workers=args.data_loader_worker_n, shuffle=True, pin_memory=True)
test_progress = tqdm(test_loader)
# model =MIL(bert_model=args.transformer_layer_name, hidden_dim=768, classifier_dims=[768, 256, 1])
# model = MILModel(bert_model=args.transformer_layer_name, hidden_dim=768, classifier_dims=[768, 256, 1])
model = MILConfidenceMaxPool(bert_model=args.transformer_layer_name, hidden_dim=768)
model.load_state_dict(torch.load(r'D:\zzl\MutiHunter\MIL\MIL\MIL_model5.pth'))
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用GPU or CPU
_ = torch.tensor([0.0]).to(device)
torch.cuda.manual_seed_all(args.seed)
model = model.to(device)  # 加载模型
model.eval()
TP, FP, FN, TN = 0, 0, 0, 0  # 计算混淆矩阵
correct_preds = 0  # 统计正确预测的样本数
batch_idx = 1
project = []
with torch.no_grad():
    for x,y in tqdm(test_progress):
        if len(x['input_ids']) == 0:
            pass
        if y == 1:
            input_ids, attention_mask, file_mask, y = (torch.tensor(x['input_ids']).to(device),
                                                       torch.tensor(x['attention_mask']).to(device),
                                                       torch.tensor(x['file_mask']).to(device),
                                                       y.float().to(device))
        if y == 0:
            input_ids, attention_mask, y = (torch.tensor(x['input_ids']).to(device),
                                            torch.tensor(x['attention_mask']).to(device),
                                            y.float().to(device))
            file_mask = [1] * 20
            file_mask = torch.tensor(file_mask).to(device)
        # input_ids, attention_mask, file_mask, y = (torch.tensor(x['input_ids']).to(device),
        #                                 torch.tensor(x['attention_mask']).to(device),
        #                                 file_mask,
        #                                 y.unsqueeze(1).to(device))
        # output, instance_pred = model(input_ids, attention_mask, file_mask)
        output, instance_pred, confidence = model(input_ids, attention_mask, file_mask)
        predict = (output >= 0.5).long().item()
        true_label = y.item()
        y_list.append(predict)
        x_list.append(x['name'][0])
        file = []
        bad_file_pro,bad_file_index = torch.topk(instance_pred, 1, dim=1)
        if predict == 1:
            for i in range(0, 20):
                if i >= len(x['file_name']):
                    break
                if i < len(bad_file_index[0]):
                        index = bad_file_index[0][i]
                        if index < len(x['file_name']):
                            if isinstance(list(x['file_name'])[index],list):
                                file.append(list(x['file_name'])[index][0])
                            else:
                                file.append(list(x['file_name'])[index])
        project.append(x['name'][0])
        try:
            file_list.append(','.join(file))
        except Exception as e:
            pass
        # 计算 TP, FP, FN, TN
        if predict == 1 and true_label == 1:
            TP += 1  # 预测为恶意，实际也是恶意
        elif predict == 1 and true_label == 0:
            print(x['name'])
            FP += 1  # 预测为恶意，实际是良性
        elif predict == 0 and true_label == 1:
            FN += 1  # 预测为良性，实际是恶意
        elif predict == 0 and true_label == 0:
            TN += 1  # 预测为良性，实际也是良性

        correct_preds += (predict == true_label)  # 更新正确预测的样本数
        # 计算评估指标
        accuracy = 100 * correct_preds / batch_idx
        precision = 100 * TP / (TP + FP) if (TP + FP) > 0 else 0  # 精确率
        tpr = 100 * TP / (TP + FN) if (TP + FN) > 0 else 0  # 召回率 (TPR)
        f1_score = (2 * precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0  # F1-score
        fpr = 100 * FP / (FP + TN) if (FP + TN) > 0 else 0  # 假阳性率 (FPR)
        fnr = 100 * FN / (TP + FN) if (TP + FN) > 0 else 0  # 假阴性率 (FNR)
        tnr = 100 * TN / (FP + TN) if (FP + TN) > 0 else 0  # 真阴性率 (TNR)
        batch_idx += 1
        test_progress.set_postfix({
            "Acc": f"{accuracy:.8f}%",
            "Prec": f"{precision:.8f}%",
            "F1": f"{f1_score:.8f}%",
            "TPR": f"{tpr:.8f}%",
            "FPR": f"{fpr:.8f}%",
            "FNR": f"{fnr:.8f}%",
            "TNR": f"{tnr:.8f}%",
            "TP": TP, "FP": FP, "FN": FN, "TN": TN
        })





import pandas as pd
df = pd.DataFrame({'package':x_list,'y':y_list,'file':file_list})
df.to_csv('MIL-TOP1.csv')
