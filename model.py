import torch.nn as nn


KINEMATICS_FEATURES_SIZE = 76
GESTURES_SIZE = 15


class LSTMModel(nn.Module):
    def __init__(self, input_size=KINEMATICS_FEATURES_SIZE, hidden_size=50, output_size=GESTURES_SIZE, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        lstm_out, _ = self.lstm(x) # out = contains the hidden state output for each time step in the input sequence.
        out = self.fc(lstm_out) # tensor(15)  
        return out

    def print_parameters_data_type(self):
        """ should be float32 """
        for name, param in self.named_parameters():
            print(f"Parameter: {name}, Data Type: {param.dtype}")
        
    
# The model outputs a tensor of shape [batch_size, num_classes] where:
# - batch_size: The number of samples in the current mini-batch.
# - num_classes: The number of classes (15 in this case), representing the labels that the model predicts.
# 
# Each element of the output tensor is a raw score (logit) for each class. These logits are unnormalized, 
# and represent the model's confidence for each class. Higher values indicate higher confidence for that class.
#
# The model's output tensor will be used with the CrossEntropyLoss function, which will internally 
# apply the softmax function to convert the logits into probabilities, and then calculate the loss based on 
# the correct labels.
#
# The predicted class for each sample can be determined by selecting the index of the highest logit, 
# which corresponds to the model's predicted class for that sample.
