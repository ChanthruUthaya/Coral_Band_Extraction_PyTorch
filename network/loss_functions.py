import torch
import torch.backends.cudnn
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from multiprocessing import cpu_count
from pathlib import Path
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class WeightedCrossEntropyLoss(nn.BCEWithLogitsLoss):
    def __init__(self, reduction: str):
        super(WeightedCrossEntropyLoss, self).__init__(reduction=reduction)
    
    def forward(self, inputs: torch.Tensor, target: torch.Tensor, weights:torch.Tensor) -> torch.Tensor:

        assert self.weight is None or isinstance(self.weight, Tensor)
        assert self.pos_weight is None or isinstance(self.pos_weight, Tensor)

        # loss = torch.Tensor([])

        # for i, _ in enumerate(inputs):

        loss_val = F.binary_cross_entropy_with_logits(inputs, target,
                                                self.weight,
                                                pos_weight=weights,
                                                reduction=self.reduction)
            # print(loss_val)
            # loss = torch.cat([loss, loss_val])
        
        # loss = torch.mean(loss)
        # loss = torch.div(loss, 256*256)

        print(loss_val)

        return loss_val
    

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

def binary_focal_loss_with_logits(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float = .25,
        gamma: float = 2.0,
        eps: float = 1e-8) -> torch.Tensor:

    probs = torch.sigmoid(input)
    target = target.unsqueeze(dim=1)
    loss_tmp = - alpha * torch.pow((1. - probs + eps), gamma) * target * torch.log(probs + eps) \
               - (1 - alpha) * torch.pow(probs + eps, gamma) * (1. - target) * torch.log(1. - probs + eps)

    loss_tmp = loss_tmp.squeeze(dim=1)

    return torch.mean(loss_tmp)


class BinaryFocalLossWithLogits(nn.Module):

    def __init__(self, alpha: float, gamma: float = 2.0) -> None:
        super(BinaryFocalLossWithLogits, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.eps: float = 1e-8

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return binary_focal_loss_with_logits(
            input, target, self.alpha, self.gamma, self.eps)

class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=3, gamma=2, ignore_index=None, reduction='mean',**kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6 # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

        # if self.alpha is None:
        #     self.alpha = torch.ones(2)
        # elif isinstance(self.alpha, (list, np.ndarray)):
        #     self.alpha = np.asarray(self.alpha)
        #     self.alpha = np.reshape(self.alpha, (2))
        #     assert self.alpha.shape[0] == 2, \
        #         'the `alpha` shape is not match the number of class'
        # elif isinstance(self.alpha, (float, int)):
        #     self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        # else:
        #     raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -torch.sum(pos_weight * torch.log(prob)) / (torch.sum(pos_weight) + 1e-4)
        
        
        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * torch.sum(neg_weight * F.logsigmoid(-output)) / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss

        return loss

class HausdorfLoss(nn.Module):

    def __init__(self, reduction="mean",gamma=1, theta=1.0, sigma=1.0):
        super(HausdorfLoss, self).__init__()
        self.eps = 1e-8
        self.reduction = reduction
        self.theta = theta
        self.sigma = sigma
        self.gamma =gamma

    
    def forward(self, input, target):

        probs = torch.sigmoid(input)

        pred_clone = probs.detach().cpu().numpy()
        label_clone = target.detach().cpu().numpy()

        weight_maps = []

        for batch_ind in range(pred_clone.shape[0]):

            image_boundaries, label_boundaries = get_boundaries(pred_clone[batch_ind], label_clone[batch_ind])

            c_image_boundaries = (ctypes.c_int * len(image_boundaries))(*image_boundaries)
            c_label_boundaries = (ctypes.c_int * len(label_boundaries))(*label_boundaries)
            
            C.one_value_euclidean.restype = ctypes.c_double

            weight_map = np.ones((pred_clone.shape[1], pred_clone.shape[2]))

            for i in range(0,len(label_boundaries),2):
                x = ctypes.c_int(label_boundaries[i])
                y = ctypes.c_int(label_boundaries[i+1])

                distance = C.one_value_euclidean(x,y,c_image_boundaries, len(image_boundaries))

                weight_map[label_boundaries[i]][label_boundaries[i+1]] += self.theta*math.exp(-distance/self.sigma)
            
            
            for i in range(0,len(image_boundaries),2):
                x = ctypes.c_int(image_boundaries[i])
                y = ctypes.c_int(image_boundaries[i+1])

                distance = C.one_value_euclidean(x,y,c_label_boundaries, len(c_image_boundaries))

                weight_map[image_boundaries[i]][image_boundaries[i+1]] += self.theta*math.exp(-distance/self.sigma)

            weight_maps.append(weight_map)
        
        weight_maps = torch.tensor(np.stack(weight_maps, axis=0)).to(DEVICE)

        loss_tmp = weight_maps*(-torch.pow((1. - probs), self.gamma) * target * torch.log(probs + self.eps) -torch.pow(probs, self.gamma) * (1. - target) * torch.log(1. - probs + self.eps)) #first line when target is positive class, second line when negative class

        loss_tmp = loss_tmp.squeeze(dim=1)

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        
        return loss

# self.alpha (1 - self.alpha) * 