import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.ST_Former import GenerateModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from dataloader.dataset_DFEW import train_data_loader, test_data_loader

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--data_set', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()



def main():
    print('The training set: set ' + str(args.data_set))


    # Data loading code
    train_data = train_data_loader(data_set=args.data_set)
    test_data = test_data_loader(data_set=args.data_set)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        inf = '********************' + str(epoch) + '********************'

        # train for one epoch
        train(train_loader)
        # evaluate on validation set
        validate(val_loader)




def train(train_loader):

    for i, images in enumerate(train_loader):
    # print(images)
        continue
    print("train done")



def validate(val_loader):
    for i, images in enumerate(val_loader):
        # print(images)
        continue
    print("val done")



if __name__ == '__main__':
    main()
