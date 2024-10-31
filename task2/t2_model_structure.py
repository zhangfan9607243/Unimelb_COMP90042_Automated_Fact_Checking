from t2_dataset_datalod import *

# Defeine model structure
class ModelTask2(nn.Module):
    def __init__(self, distilroberta_model, num_labels):
        super(ModelTask2, self).__init__()

        self.distilroberta_model = distilroberta_model
        
        self.classifier = nn.Sequential(
            nn.Linear(self.distilroberta_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, num_labels),
            nn.Softmax(dim=1)
        )
        
    def forward(self, input_ids, attention_mask):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        outputs = self.distilroberta_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]
        return self.classifier(hidden_state)