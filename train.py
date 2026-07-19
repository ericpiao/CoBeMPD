import argparse

import pandas as pd
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from line20_dataloader import CodeDataset
from model import MILModel
from torch.utils.data import random_split
import warnings
import torch
import numpy as np
from ATT_MIL import ATT_MILModel
from MLPModel import MLPModel
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
warnings.filterwarnings("ignore")

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1999, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--data_loader_worker_n', type=int, default=0, help='data_loader_worker_n')
    parser.add_argument('--transformer_layer_name', type=str, default=r'D:\zzl\PypiHunter\module\codebert')
    # 获取参数值
    args = parser.parse_args()
    epochs = args.epochs
    malicious_dataset_train = torch.load(r'./dataloader1.pth')
    benign_dataset_train = torch.load(r'./dataloader2.pth')

    # 使用 ConcatDataset 合并数据集
    dataset = ConcatDataset([malicious_dataset_train
                                ,malicious_dataset_train,malicious_dataset_train,
                             malicious_dataset_train,
                             malicious_dataset_train,malicious_dataset_train,malicious_dataset_train,
                             malicious_dataset_train,
                             malicious_dataset_train,malicious_dataset_train,malicious_dataset_train,
                             malicious_dataset_train,
                             malicious_dataset_train, malicious_dataset_train, malicious_dataset_train,
                             malicious_dataset_train,
                             malicious_dataset_train, malicious_dataset_train, malicious_dataset_train,
                             malicious_dataset_train,
                             malicious_dataset_train, malicious_dataset_train, malicious_dataset_train,
                             malicious_dataset_train,
                             benign_dataset_train
                             ])
    df = pd.read_csv(f'.\\data\\csv_data\\pypi to pypi-1.csv',names=['package','label'])
    test_package_name = list(df['package'])
    train_dataset = []
    test_dataset = []
    for item in dataset:
        if item[0]['name'] in test_package_name and item[0]['name'] not in test_dataset:
            test_dataset.append(item)
        else:
            train_dataset.append(item)

    def collate(batch):
        # 分别收集字段
        input_ids = [item[0]['input_ids'] for item in batch]  # tensor
        attention_masks = [item[0]['attention_mask'] for item in batch]  # tensor
        names = [item[0]['name'] for item in batch]  # str
        file_names = [item[0]['file_name'] for item in batch]  # list[int] or int
        labels = [item[1] for item in batch]  # tensor

        # 只堆叠 tensor 类型
        input_ids = torch.stack(input_ids, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)
        labels = torch.stack(labels, dim=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'name': names,
            'file_name': file_names  # 保持为 list，或你可自己转成 tensor
        }, labels
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.data_loader_worker_n, shuffle=True, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            num_workers=args.data_loader_worker_n, shuffle=True, pin_memory=True)
    model = MLPModel(bert_model=args.transformer_layer_name, hidden_dim=768, classifier_dims=[768, 256, 1])
    model.train()  # 训练模式
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用GPU or CPU
    _ = torch.tensor([0.0]).to(device)
    torch.cuda.manual_seed_all(args.seed)
    model = model.to(device)  # 加载模型
    # model.load_state_dict(torch.load('./MIL_model3.pth'))
    nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(0, epochs):
        train_loss = 0.0
        correct_preds = 0
        total_preds = 0
        train_progress = tqdm(train_loader,
                               desc=f"Epoch {epoch+1}/{epochs}",
                              unit="batch")
        model.train()
        for batch_idx, (x, y) in enumerate(train_progress):
            if y == 1:
                input_ids, attention_mask, file_mask, y = (torch.tensor(x['input_ids']).to(device),
                                                          torch.tensor(x['attention_mask']).to(device),
                                                            torch.tensor(x['file_mask']).to(device),
                                                          y.float().to(device))
            if y == 0:
                input_ids, attention_mask, y = (torch.tensor(x['input_ids']).to(device),
                                                           torch.tensor(x['attention_mask']).to(device),
                                                           y.float().to(device))
                file_mask = [1]*20
                file_mask = torch.tensor(file_mask).to(device)
            optimizer.zero_grad()
            output,instance_pred = model(input_ids, attention_mask,file_mask)
            loss = criterion(output, y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predicted_labels = (output >= 0.5).long()
            correct_preds += (predicted_labels == y).sum()
            total_preds += y.size(0)
            avg_loss = train_loss / (batch_idx+1)
            accuracy = 100 * correct_preds.item() / total_preds
            train_progress.set_postfix(
                Loss=avg_loss, Accuracy=str(accuracy) + r"%")

        model.eval()
        val_loss = 0
        correct_preds = 0
        total_preds = 0
        val_accuracy = 0
        # with torch.no_grad():
        #     for batch_idx, (x, y) in enumerate(val_loader):
        #         input_ids, attention_mask, y = (torch.tensor(x['input_ids']).to(device),
        #                                         torch.tensor(x['attention_mask']).to(device),
        #                                         torch.tensor(x['file_mask']).to(device),
        #                                         y.float().to(device))
        #         output, instance_pred = model(input_ids, attention_mask)
        #         loss = criterion(output, y)
        #         val_loss += loss.item()
        #         predicted_labels = (output >= 0.5).long()
        #         correct_preds += (predicted_labels == y).sum()
        #         total_preds += y.size(0)
        #         val_accuracy = 100 * correct_preds.item() / total_preds
        # tqdm.write(f"Epoch {epoch + 1} , Val_Loss={val_loss / (batch_idx+1)}, Val_Accuracy={str(val_accuracy)+r"%"}")
        torch.save(model.state_dict(), rf'./mlp_MIL_model{epoch+1}.pth')


if __name__ == "__main__":
    train()
