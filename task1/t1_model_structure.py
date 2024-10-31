from t1_dataset_datalod import *

# Import models
import torch
import torch.nn as nn
import torch.optim as optim

# Define model & optimizer & loss function
class ModelTask1(nn.Module):
    def __init__(self, distilroberta_model, num_labels):
        super(ModelTask1, self).__init__()

        self.distilroberta_model = distilroberta_model
        
        self.classifier = nn.Sequential(
            nn.Linear(self.distilroberta_model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, num_labels),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids, attention_mask):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        outputs = self.distilroberta_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]
        return self.classifier(hidden_state)