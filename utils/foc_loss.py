import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        
        super(FocalLoss, self).__init__()

        self.alpha = alpha

        self.gamma = gamma

        self.reduction = reduction


    def forward(self, inputs, targets):

        # Compute the standard cross entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)  # prevents nans when probability 0

        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
