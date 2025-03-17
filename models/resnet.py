import torch
import torch.nn as nn
from torchvision import models

class ResNetTransfer(nn.Module):

    def __init__(self, num_classes=3, freeze_layers=True):

        super(ResNetTransfer, self).__init__()

        self.model = models.resnet18(pretrained=True)

        if freeze_layers:
            for param in self.model.parameters():
                param.requires_grad = False

        num_features = self.model.fc.in_features

        self.model.fc = nn.Linear(num_features, num_classes)
    
    
    def forward(self, x):
        return self.model(x)
