import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    
    def __init__(self, num_classes=3):

        super(CNN, self).__init__()
        
        # Updated architecture with BatchNorm and Dropout
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.dropout = nn.Dropout(0.5)
        
        # After two poolings, feature map size: 64x64 -> 32x32 -> 16x16
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)
    

    def forward(self, x):

        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # [B, 32, 32, 32]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # [B, 64, 16, 16]

        x = F.relu(self.bn3(self.conv3(x)))              # [B, 128, 16, 16]
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
