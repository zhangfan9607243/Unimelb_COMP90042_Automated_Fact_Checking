# Import packages
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model & tokenizer (with suggested code for speeding up model & tokenizer loading from cloud computing service provider)
import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

# Load model & tokenizer
from transformers import AutoTokenizer, AutoModel
distilroberta_token = AutoTokenizer.from_pretrained("distilroberta-base")
distilroberta_model = AutoModel.from_pretrained("distilroberta-base")

# Define dataset & dataloader
class ModelDataset(Dataset):
    
    def __init__(self, data, distilroberta_token, is_tran, max_length=512):
        self.data = data
        self.distilroberta_token = distilroberta_token
        self.is_tran = is_tran
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        claim = row['claim_text']
        evidence = row['evidence_text']
        
        encoded = self.distilroberta_token(
            claim, evidence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        if self.is_tran:
            label = torch.tensor(row['label'], dtype=torch.long)
            return {
                'input_ids': encoded['input_ids'].squeeze(0).to(device),
                'attention_mask': encoded['attention_mask'].squeeze(0).to(device),
                'label': label.float().to(device)
            }
        else:
            return {
                'input_ids': encoded['input_ids'].squeeze(0).to(device),
                'attention_mask': encoded['attention_mask'].squeeze(0).to(device),
            }

data_tran_t2 = pd.read_json('/root/COMP90042/data/data2/data_tran_t2.json', orient='records', lines=True)
data_vald_t2 = pd.read_json('/root/COMP90042/data/data2/data_vald_t2.json', orient='records', lines=True)
data_test_t2 = pd.read_json('/root/COMP90042/data/data2/data_test_t2.json', orient='records', lines=True)
data_evdn    = pd.read_json('/root/COMP90042/data/data2/data_evdn.json', orient='records', lines=True)

dataset_tran = ModelDataset(data_tran_t2, distilroberta_token, is_tran=True)
dataset_vald = ModelDataset(data_vald_t2, distilroberta_token, is_tran=True)
dataset_test = ModelDataset(data_test_t2, distilroberta_token, is_tran=False)

datalod_tran = DataLoader(dataset_tran, batch_size=32, shuffle=True)
datalod_vald = DataLoader(dataset_vald, batch_size=32, shuffle=True)
datalod_test = DataLoader(dataset_test, batch_size=32, shuffle=True)

