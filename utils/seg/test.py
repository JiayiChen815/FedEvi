import logging
import torch
from medpy.metric.binary import hd95
from scipy import ndimage
import numpy as np
import pdb
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as F
import cv2

def cal_dice_hd95(pred, label):
    dice_score = np.zeros(pred.shape[1])
    hd95_score = np.zeros(pred.shape[1])

    smooth = 1e-5
    for data_idx in range(pred.shape[0]):
        avg_sample_dice = 0
        for class_idx in range(pred.shape[1]): 
            label_i = (label>class_idx).astype('float')  # polyp: label>0; fundus: label>0 -> optic disc; label>1 -> optic cup
        
            inter = (pred[data_idx, class_idx, ...] * label_i[data_idx, 0, ...]).sum()
            pred_sum = pred[data_idx, class_idx, ...].sum()
            label_sum = label_i[data_idx, 0, ...].sum()

            dice_idx = (2*inter + smooth) / (pred_sum + label_sum + smooth)
            dice_score[class_idx] += dice_idx

            hd95_idx = hd95(label_i[data_idx, 0, ...], pred[data_idx, class_idx, ...])
            hd95_score[class_idx] += hd95_idx

            avg_sample_dice += dice_idx
        avg_sample_dice /= pred.shape[1]
            
    return dice_score / pred.shape[0], hd95_score / pred.shape[0]


def connectivity_region_analysis(mask):    
    mask_np = mask.cpu().numpy()
    label_im, nb_labels = ndimage.label(mask_np) 
    sizes = ndimage.sum(mask_np, label_im, range(nb_labels + 1))

    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return torch.from_numpy(label_im).cuda()


def test(dataset, model, dataloader, client_idx, args, savefig=False):
    model.eval()

    if not os.path.exists('visualization/{}/{}/seed{}/client{}/gt/'.format(args.dataset, args.fl_method, args.seed, client_idx+1)):
        os.makedirs('visualization/{}/{}/seed{}/client{}/gt/'.format(args.dataset, args.fl_method, args.seed, client_idx+1))
    if not os.path.exists('visualization/{}/{}/seed{}/client{}/pred/'.format(args.dataset, args.fl_method, args.seed, client_idx+1)):
        os.makedirs('visualization/{}/{}/seed{}/client{}/pred/'.format(args.dataset, args.fl_method, args.seed, client_idx+1))

    pred_list = torch.tensor([]).cuda()
    label_list = torch.tensor([]).cuda()

    iters = 0
    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):
            image, label = data['image'], data['label']

            image = image.cuda()
            label = label.cuda()
            label_list = torch.cat((label_list, label), 0)

            logit = model(image)[0]

            pred_class = torch.argmax(logit, dim=1, keepdim=True)    

            for i in range(image.shape[0]):
                iters += 1
                if savefig == True:
                    plt.imsave('./visualization/{}/{}/seed{}/client{}/pred/pred_{}.png'.format(args.dataset, args.fl_method, args.seed, client_idx+1,iters), pred_class[i,0,...].cpu().numpy(), cmap='gray')
                    plt.imsave('./visualization/{}/{}/seed{}/client{}/gt/gt_{}.png'.format(args.dataset, args.fl_method, args.seed, client_idx+1,iters), label[i,0,...].cpu().numpy(), cmap='gray')

                processed_pred = torch.tensor([]).cuda()
                
                for class_idx in range(1, logit.shape[1]):   # 正类
                    processed_pred = torch.cat((processed_pred, connectivity_region_analysis(pred_class[i:i+1] > class_idx-1)), 1)
                
                pred_list = torch.cat((pred_list, processed_pred), 0)

        pred_list = pred_list.cpu().numpy()
        label_list = label_list.cpu().numpy()

        dice_score, hd95_score = cal_dice_hd95(pred_list, label_list)

        return dice_score, hd95_score