import torch
from model import LSTMModel
from dataset import dataset
from preprocessing import norm

# extract dataset
train_100, test = dataset("needle_passing",None)
# preprocessing 
norm(train_100.data)

# test phase
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel().to(device) 
model.load_state_dict(torch.load("saved_model_parameters/train1009085.pth",map_location=device))

model.eval() 
correct = 0
with torch.no_grad():
    for trial_idx, (inputs, targets) in enumerate(test):
            inputs = torch.tensor(inputs, dtype=torch.float32)
            targets = torch.tensor(targets, dtype=torch.long)
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            values, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
    acc = correct / test.total_timestamps
    print(f"accuracy: {acc:.4f}")