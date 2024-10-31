from t1_dataset_datalod import *
from t1_model_structure import *

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_task1 = ModelTask1(distilroberta_model, num_labels=1).to(device)
model_task1.load_state_dict(torch.load('/root/COMP90042/model/model_m1t1/model_task1.pth'))
model_task1.to(device) 
model_task1.eval() 

# Prepare test dataset
class TestDataset(Dataset):
    def __init__(self, data_test, data_evdn, tokenizer, max_length=128):

        self.claim = data_test['claim']
        self.claim_id = data_test['claim_id']
        self.data_evdn = data_evdn
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_evdn)

    def __getitem__(self, idx):
        evidence = self.data_evdn['evidence'][idx]
        evidence_id = self.data_evdn['evidence_id'][idx]

        inputs = self.tokenizer(
            self.claim,
            evidence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'claim_id':self.claim_id,
            'evidence_id': evidence_id 
        }

result_dict = {claim_id: [] for claim_id in data_test_t1['claim_id']}

# Predict on test dataset
def predict_model(idx):

    dataset_test = TestDataset(data_test_t1.iloc[idx], data_evdn, distilroberta_token)
    datalod_test = DataLoader(dataset_test, batch_size=32)
    claim_id = data_test_t1.iloc[idx]['claim_id']
    evidence_id_list = []
    prob_list = []

    with torch.no_grad(): 
        for batch in datalod_test:
            evidence_id = batch['evidence_id'].tolist()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model_task1(input_ids, attention_mask).tolist()
            evidence_id_list.extend(evidence_id)
            prob_list.extend(outputs)
    
    result_df = pd.DataFrame({
        'evidence_id': evidence_id_list,
        'prob': prob_list
    })

    result_dict[claim_id] = result_df.sort_values(by='prob', ascending=False).head(5)['evidence_id'].tolist()

from tqdm import tqdm
for i in tqdm(range(153)):
    predict_model(i)

# Save prediction result
result_df = data_test_t1.copy()
result_df['evidences'] = result_df['claim_id'].map(result_dict)
result_df.to_json("/root/COMP90042/data/data3/t1_result.json", orient="records", lines=True)