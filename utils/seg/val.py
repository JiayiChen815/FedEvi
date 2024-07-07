import logging
import torch
from medpy.metric.binary import hd95
from scipy import ndimage
import numpy as np
import pdb
from utils.loss_func import Dice_Loss_val
import torch.nn as nn
import torch.nn.functional as F

def val(model, dataloader, args):
    model.eval()

    loss = 0.0
    seg_loss = Dice_Loss_val()

    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):

            image, label = data['image'], data['label']

            image = image.cuda()
            label = label.cuda()

            if args.fl_method == 'FedEvi':
                logit = model(image)[0]
                logit = torch.clamp_max(logit, 80)
                alpha = torch.exp(logit)+1
                S = torch.sum(alpha, dim=1, keepdim=True) 
                loss += seg_loss(alpha/S, label)
            else:
                pred = model(image)[1]
                loss += seg_loss(pred, label)

        return loss / len(dataloader)