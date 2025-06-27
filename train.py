import torch
import torch.nn as nn
from model import LSTMModel
from torch.utils.data import Dataset, DataLoader
from dataset import dataset
import os

class Trial(Dataset):
    def __init__(self, X, y):
        self.X = X # torch tensor: [539, 2660, 76]
        self.y = y  # torch tensor: [539]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# extract data
train, test = dataset("Suturing","G")   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel().to(device) 

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 4

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0

    # train = list of trials
    for trial_idx, (inputs, targets) in enumerate(train):
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.long)
        inputs, targets = inputs.to(device), targets.to(device)

        trial = Trial(inputs,targets)
        trial_loader = DataLoader(trial,batch_size=16,shuffle=False)

        for inputs, targets in trial_loader:


            optimizer.zero_grad()
            
            outputs = model(inputs) 
            loss = criterion(outputs, targets)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

     
    loss_value = total_loss /  train.total_timestamps
    print(f"Epoch {epoch+1}: Loss = {loss_value:.4f}, loss : {total_loss}")

    

     

            # _, preds = torch.max(outputs, 1)
            # correct += (preds == labels).sum().item()

    # acc = correct / len(dataset)

exit(0)
# save model parameters
save_path = os.path.join(os.path.dirname(__file__), "saved_model_parameters", "model.pth")
torch.save(model.state_dict(), save_path)
print("\nmodel parameters saved in: ")
print(os.path.abspath("model.pth"))