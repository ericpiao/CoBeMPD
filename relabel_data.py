import json
import os

import pandas as pd
from torch.utils.data import random_split
import warnings
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

path = r'D:\zzl\MutiHunter\result_data\file_data'
df = pd.read_csv(r'.\\data\\csv_data\\pypi to pypi-1.csv', names=['package', 'label'])
print(df)
map_df = pd.read_csv(r'D:\zzl\MutiHunter\data\supplemental materials\supplemental materials\poisoning-dataset\pypi_benign_package_version.csv')
print(map_df)
map_df = map_df.merge(df, left_on='match_name', right_on='package', how='outer')
map_df = map_df.dropna()
print(map_df.describe())
for json_file in tqdm(os.listdir(path), ncols=100):
    try:
        with open(path + "\\" + json_file, "r", encoding="utf-8") as file:
            project_data = json.load(file)
            count = 0
            for item in project_data['file']:
                if project_data['file'][item]['label'] == 1:
                    count += 1
            if count == 0 and len( project_data['file']) != 0:
                print(project_data['project'])
        # try:
        #     with open(path + "\\" + json_file, "w", encoding="utf-8") as json_file:
        #         json.dump(project_data, json_file, indent=4)
        # except Exception as e:
        #     print(1)
    except Exception as e:
        print(json_file + str(e))

