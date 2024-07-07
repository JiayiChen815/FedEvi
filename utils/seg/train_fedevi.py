import logging
import numpy as np
import torch.nn as nn
import pdb
import torch
import torch.nn.functional as F
from utils.loss_func import EDL_Dice_Loss, Dice_Loss
import copy
import matplotlib.pyplot as plt

def train(round_idx, client_idx, model, dataloader, optimizer, args):
    model.train()

    max_epoch = args.max_epoch

    seg_loss = EDL_Dice_Loss(kl_weight=args.kl_weight, annealing_step=args.annealing_step)


    for epoch in range(max_epoch):
        for iters, (_, data) in enumerate(dataloader):
            optimizer.zero_grad()

            image, label = data['image'], data['label']
            image = image.cuda()
            label = label.cuda()

            logit = model(image)[0]
            loss = seg_loss(logit, label, round_idx)

            loss.backward()
            optimizer.step()

    return model
            
