import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, num_classes=3):

        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers (assuming input image size 64x64, after 3 poolings: 64 -> 32 -> 16 -> 8)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 16, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv3(x)))  # -> [B, 64, 8, 8]

        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
