import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """Convolutional neural network for MNIST digit classification"""
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Conv1 + ReLU + Pool: 28x28 -> 14x14
        x = self.pool(F.relu(self.conv1(x)))
        # Conv2 + ReLU + Pool: 14x14 -> 7x7
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
