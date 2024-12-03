import torch
import torch.nn as nn

class chatModel(torch.nn.Module):
    def __init__(self, input_feature, hidden_feature, out_feature):
        super(chatModel, self).__init__()
        
        # Layers
        self.layer1 = nn.Linear(input_feature, hidden_feature)
        self.layer2 = nn.Linear(hidden_feature, hidden_feature)
        self.layer3 = nn.Linear(hidden_feature, out_feature)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Optional: Batch Normalization and Dropout for better generalization
        self.batchnorm1 = nn.BatchNorm1d(hidden_feature)
        self.batchnorm2 = nn.BatchNorm1d(hidden_feature)
        
        self.dropout = nn.Dropout(0.5)  # Dropout probability of 50%

    def forward(self, x):
        # First layer
        x = self.layer1(x)
        x = self.relu(x)
        # x = self.batchnorm1(x)  # Apply batch norm
        
        # Second layer
        x = self.layer2(x)
        x = self.relu(x)
        # x = self.batchnorm2(x)  # Apply batch norm
        
        # # Dropout layer (regularization)
        # x = self.dropout(x)
        
        # Output layer
        x = self.layer3(x)

        return x
