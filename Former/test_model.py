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
from models2.ST_Former import GenerateModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from dataloader.dataset_DFEW import train_data_loader, test_data_loader
from sklearn.metrics import accuracy_score,confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default='checkpoint/model_CE.pth', type=str, metavar='checkpoint/model_CE.pth', help='path to latest checkpoint')
parser.add_argument('--data_set', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")
project_path = './'
log_txt_path = project_path + 'log/' + time_str + 'set' + str(args.data_set) + '-log.txt'
log_curve_path = project_path + 'log/' + time_str + 'set' + str(args.data_set) + '-log.png'
checkpoint_path = project_path + 'checkpoint/' + time_str + 'set' + str(args.data_set) + '-model.pth'
best_checkpoint_path = project_path + 'checkpoint/' + time_str + 'set' + str(args.data_set) + '-model_best.pth'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def main():


    model = GenerateModel()
    # model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint

    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load('checkpoint/3.pth')
    model.load_state_dict(checkpoint['state_dict'])


    cudnn.benchmark = True

    # Data loading code
    test_data = test_data_loader(data_set=args.data_set)


    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)


    validate(val_loader, model)



def validate(val_loader, model):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        pres_tr, trues_tr = [], []
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 5))

            _, pre_tr = torch.max(output, 1)
            pres_tr += pre_tr.cpu().numpy().tolist()
            trues_tr += target.cpu().numpy().tolist()


        print('Current WAR: '+ str(get_WAR(trues_tr,pres_tr)))
        print('Current UAR: '+ str(get_UAR(trues_tr,pres_tr)))



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_WAR(trues_te, pres_te):
    WAR  = accuracy_score(trues_te, pres_te)
    return WAR

def get_UAR(trues_te, pres_te):
    cm = confusion_matrix(trues_te, pres_te)
    acc_per_cls = [ cm[i,i]/sum(cm[i]) for i in range(len(cm))]
    UAR = sum(acc_per_cls)/len(acc_per_cls)
    return UAR





class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)





if __name__ == '__main__':
    main()
