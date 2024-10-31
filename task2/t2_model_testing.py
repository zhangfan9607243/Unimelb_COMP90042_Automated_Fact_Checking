from t2_dataset_datalod import *
from t2_model_structure import *

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_task2 = ModelTask2(distilroberta_model=distilroberta_model, num_labels=4)
model_task2.load_state_dict(torch.load("/root/COMP90042/model/model_task2/model_task2.pth"))
model_task2.to(device)
model_task2.eval() 

# Predict on test dataset
def predict_model(model, dataloader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            predicted_labels = torch.argmax(outputs, dim=1)
            
            predictions.extend(predicted_labels.cpu().numpy())
    
    label_map = {
        0: "SUPPORTS",
        1: "REFUTES",
        2: "NOT_ENOUGH_INFO",
        3: "DISPUTED"
    }

    return [label_map[label] for label in predictions]

# Save prediction result
result_df = data_test_t2.copy()
result_df['pred_label'] = predict_model(model_task2, datalod_test)
result_df.to_json("/root/COMP90042/data/data3/t2_result.json", orient="records", lines=True)