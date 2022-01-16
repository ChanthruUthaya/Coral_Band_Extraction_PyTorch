import torch
import numpy as np
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = 0 if alpha is None else alpha
        self.eps = 1e-8
        self.reduction = reduction

    
    def forward(self, input, target):

        probs = torch.sigmoid(input)

        #target = target.unsqueeze(dim=1)

        loss_tmp = -torch.pow((1. - probs), self.gamma) * target * torch.log(probs + self.eps) -torch.pow(probs, self.gamma) * (1. - target) * torch.log(1. - probs + self.eps) #first line when target is positive class, second line when negative class

        loss_tmp = loss_tmp.squeeze(dim=1)

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        
        return loss