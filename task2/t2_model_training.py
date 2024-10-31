from t2_dataset_datalod import *
from t2_model_structure import *

# Define mode & optimizer & loss
model_task2 = ModelTask2(distilroberta_model=distilroberta_model, num_labels=4).to(device)
optimizer = optim.AdamW(model_task2.parameters(), lr=1e-5) 
criterion = nn.CrossEntropyLoss()

# Early stop
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, vald_loss):
        if self.best_loss is None or vald_loss < self.best_loss - self.delta:
            self.best_loss = vald_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered")

early_stopping = EarlyStopping(patience=10, delta=0.0001)

# Model evaluation
def evaluate_model(model, datalod, criterion):
    model.eval()
    total_loss, correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for batch in datalod:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            if len(labels.shape) > 1:
                labels = torch.argmax(labels, dim=1)

            outputs = model(input_ids, attention_mask)  
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1) 
            correct += (predictions == labels).sum().item() 
            total_samples += labels.size(0) 

    avg_loss = total_loss / len(datalod)
    accuracy = correct / total_samples

    return accuracy, avg_loss

# Model training
def train_model(model, datalod_tran, datalod_vald, criterion, optimizer, epochs, epochs_eval, save_path="model/model_task2/model_task2.pth"):
    
    best_vald_accu = 0  

    for epoch in range(epochs):
        model.train()

        for batch in datalod_tran:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        if (epoch + 1) % epochs_eval == 0:
            tran_accu, tran_loss = evaluate_model(model, datalod_tran, criterion)
            vald_accu, vald_loss = evaluate_model(model, datalod_vald, criterion)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Training:   Loss: {tran_loss:.4f}, Accuracy: {tran_accu:.4f}")
            print(f"Validation: Loss: {vald_loss:.4f}, Accuracy: {vald_accu:.4f}\n")
            print()

            # 保存最佳验证准确率的模型
            if vald_accu > best_vald_accu:
                best_vald_accu = vald_accu
                torch.save(model.state_dict(), save_path)
                print(f"Model saved with Val Acc: {vald_accu:.4f}\n")
                print()

            # 检查 Early Stopping 条件
            early_stopping(vald_loss)
            if early_stopping.early_stop:
                print("Training stopped early.")
                break

train_model(model_task2, 
            datalod_tran, 
            datalod_vald, 
            criterion, 
            optimizer, 
            epochs = 500, 
            epochs_eval = 10, 
            save_path="/root/COMP90042/model/model_task2/model_task2.pth"
            )