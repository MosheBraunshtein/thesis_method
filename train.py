import torch
import torch.nn as nn
from model import LSTMModel
from torch.utils.data import Dataset, DataLoader
from dataset import dataset
import os
import copy 
import numpy as np
from preprocessing import norm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def method(train,per):
    n_fetures = int((76/100)*per)
    cols_to_zeros = np.random.randint(low=0,high=76,size=n_fetures)
    for trial in train.data: 
        # trial shape (timestamps,features)
        trial[:,cols_to_zeros] = 0
    return train

class Trial(Dataset):
    def __init__(self, X, y):
        self.X = X # torch tensor: [539, 2660, 76]
        self.y = y  # torch tensor: [539]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# extract data
train_100, test = dataset("Suturing","G")
# preprocessing 
norm(train_100.data)
# split
train_70 = method(copy.deepcopy(train_100),per=30)
train_50 = method(copy.deepcopy(train_100),per=50)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel().to(device) 

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train phase
log_dir='log/train1007050/loss'
writer = SummaryWriter(log_dir=log_dir)
# writer = SummaryWriter(log_dir='log/train1007050/loss')

for index,train in enumerate([train_100,train_70,train_50]):
    index = index*5
    num_epochs = 5
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
        print(f"Epoch {epoch}: Loss = {loss_value:.4f}")
        writer.add_scalar("val", loss_value , epoch+index)

writer.close()
print(f"log to : {log_dir}")


# save model
PROJECT_ROOT = Path(__file__).resolve().parent
saved_parameters_path = PROJECT_ROOT/"saved_model_parameters"/"train1007050.pth"
torch.save(model.state_dict(), saved_parameters_path)
print(f"model saved : {saved_parameters_path} ")
    
