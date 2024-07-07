import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn
import pdb
import logging


class Dice_Loss_val(nn.Module):
    def __init__(self):
        super(Dice_Loss_val, self).__init__()
        self.smooth = 1e-5

    def forward(self, pred, label):
        K = pred.shape[1]
        one_hot_y = torch.zeros(pred.shape).cuda()
        one_hot_y = one_hot_y.scatter_(1, label, 1.0)

        dice_score = 0.0
        
        for class_idx in range(K):   
            inter = (pred[:,class_idx,...] * one_hot_y[:,class_idx,...]).sum()
            union = pred[:,class_idx,...].sum() + one_hot_y[:,class_idx,...].sum()

            dice_score += (2*inter + self.smooth) / (union + self.smooth)

        loss = 1 - dice_score/K

        return loss
    

def kl_divergence(alpha):
    shape = list(alpha.shape)
    shape[0] = 1
    ones = torch.ones(tuple(shape)).cuda()

    S = torch.sum(alpha, dim=1, keepdim=True) 
    # pdb.set_trace()
    first_term = (
        torch.lgamma(S)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(S))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl.mean()    


class EDL_Dice_Loss(nn.Module):
    def __init__(self, kl_weight=0.01, annealing_step=10):
        super(EDL_Dice_Loss, self).__init__()
        self.smooth = 1e-5
        self.kl_weight = kl_weight
        self.annealing_step = annealing_step

    def forward(self, logit, label, epoch_num):
        K = logit.shape[1]
        logit = torch.clamp_max(logit, 80)
        alpha = torch.exp(logit) + 1
        S = torch.sum(alpha, dim=1, keepdim=True) 
        pred = alpha / S

        one_hot_y = torch.zeros(pred.shape).cuda()
        one_hot_y = one_hot_y.scatter_(1, label, 1.0)
        one_hot_y.requires_grad = False

        dice_score = 0
        for class_idx in range(K):   
            inter = (pred[:,class_idx,...] * one_hot_y[:,class_idx,...]).sum()
            union = (pred[:,class_idx,...] **2).sum() + one_hot_y[:,class_idx,...].sum() 
            dice_score += (2*inter + self.smooth) / (union + self.smooth)
            
        dice_score = dice_score/K
        loss_dice = 1 - dice_score

        annealing_coef = torch.min(
                    torch.tensor(1.0, dtype=torch.float32),
                    torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32),
                ) 

        kl_alpha = (alpha - 1) * (1 - one_hot_y) + 1
        loss_kl = annealing_coef * kl_divergence(kl_alpha)


        return loss_dice + self.kl_weight * loss_kl 