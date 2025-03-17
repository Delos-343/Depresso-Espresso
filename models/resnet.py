import torch
import torch.nn as nn
from torchvision import models

class ResNetTransfer(nn.Module):
    
    def __init__(self, num_classes=3):

        super(ResNetTransfer, self).__init__()

        # Load pre-trained ResNet18
        self.model = models.resnet18(pretrained=True)

        # Optionally freeze early layers:
        # for param in self.model.parameters():
        #     param.requires_grad = False
        num_features = self.model.fc.in_features
        
        self.model.fc = nn.Linear(num_features, num_classes)
    

    def forward(self, x):
        return self.model(x)
