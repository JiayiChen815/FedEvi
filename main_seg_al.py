import numpy as np
import argparse
import os
import time
import random
import logging
import sys
import glob
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from model.unet2d import Unet2D as Model
from data.dataset import generate_dataset
from utils.fed_merge import FedAvg, FedUpdate
from utils.utils import scoring_func
from utils.seg.val import val
from utils.seg.test import test

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str,  default='Polyp', help='dataset')

parser.add_argument('--fl_method', type=str,  default='FedEvi', help='federated method')
parser.add_argument('--max_round', type=int,  default=200, help='maximum round number to train')
parser.add_argument('--max_epoch', type=int,  default=2, help='maximum epoch number to train')
parser.add_argument('--norm', type=str,  default='bn', help='normalization type')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=1e-3, help='learning rate')
parser.add_argument('--deterministic', type=bool,  default=False, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=3, help='random seed')

parser.add_argument('--kl_weight', type=float, default=0.01, help='edl kl weight')
parser.add_argument('--ratio', type=float, default=1.0, help='ratio')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma')
parser.add_argument('--annealing_step', type=int, default=10, help='annealing_step')

parser.add_argument('--num_classes', type=int, default=2, help='class num')


args = parser.parse_args()


def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)


if __name__ == '__main__':
    # log
    localtime = time.localtime(time.time())
    ticks = '{:>02d}{:>02d}{:>02d}{:>02d}{:>02d}'.format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min, localtime.tm_sec)

    snapshot_path = "UNet_{}/{}_{}_{}/".format(args.dataset.lower(), args.dataset, args.fl_method, ticks)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if not os.path.exists(snapshot_path + '/model'):
        os.makedirs(snapshot_path + '/model')

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    

    # init
    dataset = args.dataset
    assert dataset in ['Polyp']
    fl_method = args.fl_method
    assert fl_method in ['FedEvi']

    batch_size = args.batch_size
    base_lr = args.base_lr
    max_round = args.max_round
    norm = args.norm
    

    if fl_method == 'FedEvi':
        from utils.seg.train_fedevi import train
        bn = False
        norm = 'in'
        val_batch_size = 1
   
    
    if dataset == 'Polyp':
        c = 3
        client_num = 4

        
    logging.info(str(args))

    # random seed
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    

    # local dataloader, model, optimizer
    local_models = []

    local_train_loaders = []
    local_val_loaders = []
    local_test_loaders = []

    train_num = []

    for client_idx in range(client_num):
        # data
        data_train, data_val, data_test = generate_dataset(dataset=dataset, fl_method=fl_method, client_idx=client_idx)
        train_num.append(len(data_train))

        # dataloader
        train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(dataset=data_val, batch_size=val_batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
                
        local_train_loaders.append(train_loader)
        local_val_loaders.append(val_loader)
        local_test_loaders.append(test_loader)

        # model
        model = Model(c=c, num_classes=args.num_classes, norm=norm).cuda()
        local_models.append(model)

    writer = SummaryWriter(snapshot_path+'/log')

    result_after_avg = np.zeros(client_num)

    best_val = 9999 # val loss
    best_dice = 0.0

    # global model
    global_model = Model(c=c, num_classes=args.num_classes, norm=norm).cuda()

    local_optimizers = []
    local_schedulers = []
    for client_idx in range(client_num):
        optimizer = torch.optim.Adam(local_models[client_idx].parameters(), lr=args.base_lr, betas=(0.9, 0.99), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

        local_optimizers.append(optimizer)
        local_schedulers.append(scheduler)

    client_weight = train_num / np.sum(train_num)

    with open(os.path.join(snapshot_path, 'global_test_result.txt'), 'a') as f:
        print('train num: {}'.format(train_num), file=f)
        print('init weight: {}'.format(client_weight), file=f)


    u_dis = np.zeros((client_num, max_round))
    u_data = np.zeros((client_num, max_round))
    
    for round_idx in range(max_round):
        local_models = FedUpdate(global_model, local_models, bn=bn)   # distribute

        for client_idx in range(client_num):
            print(client_idx)
            local_models[client_idx] = train(round_idx=round_idx+1,
                                                client_idx=client_idx, 
                                                model=local_models[client_idx], 
                                                dataloader=local_train_loaders[client_idx], 
                                                optimizer=local_optimizers[client_idx], 
                                                args=args) 
            
            local_schedulers[client_idx].step()

        # update the global model, while the local models haven't been updated
        global_model = FedAvg(global_model, local_models, client_weight, bn=bn)   

        if fl_method == 'FedEvi':
            for client_idx in range(client_num):
                u_dis[client_idx, round_idx], u_data[client_idx, round_idx] = scoring_func(global_model, local_models[client_idx], local_val_loaders[client_idx], client_idx=client_idx, args=args)


            client_weight += args.ratio * u_dis[:, round_idx] / u_data[:, round_idx]
            client_weight /= client_weight.sum()
            client_weight = np.clip(client_weight, a_min=1e-3, a_max=None)
            global_model = FedAvg(global_model, local_models, client_weight, bn=bn)

        for client_idx in range(client_num):
            result_after_avg[client_idx] = val(model=global_model, dataloader=local_val_loaders[client_idx], args=args)

        avg_val = result_after_avg.mean()   # val_loss
        print(avg_val)
        writer.add_scalar('val_loss', avg_val, round_idx)

        if avg_val < best_val:
            best_val = avg_val

            with open(os.path.join(snapshot_path, 'global_test_result.txt'), 'a') as f:
                print('FL round {}'.format(round_idx+1), file=f)
                print('weight: {}'.format(client_weight), file=f)

            avg_dice = 0
            avg_hd95 = 0
            for client_idx in range(client_num):
                dice_score, hd95_score = test(dataset=dataset, model=global_model, dataloader=local_test_loaders[client_idx], client_idx=client_idx, args=args)
                
                avg_dice += dice_score.mean()
                avg_hd95 += hd95_score.mean()
                with open(os.path.join(snapshot_path, 'global_test_result.txt'), 'a') as f:
                    print('client {}.\tDice\t{:.5f}\tHD95\t{:.3f}'.format(client_idx, dice_score[0], hd95_score[0]), file=f)

            avg_dice /= client_num
            avg_hd95 /= client_num
            writer.add_scalar('avg_dice', avg_dice, round_idx)
            writer.add_scalar('avg_hd95', avg_hd95, round_idx)

            if avg_dice > best_dice:
                best_dice = avg_dice
                
                save_model_path = os.path.join(snapshot_path + '/model/best_seed{}_global.pth'.format(args.seed))
                torch.save(global_model.state_dict(), save_model_path)                

    writer.close()


    

