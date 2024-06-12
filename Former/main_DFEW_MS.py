import argparse
import os
import time
import shutil
from models2.ST_Former_MS import GenerateModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from dataloader.dataset_DFEW_MS import train_data_loader, test_data_loader
import random
import  numpy
from sklearn.metrics import accuracy_score,confusion_matrix

from mindspore import dtype as mstype
import mindspore.numpy as mnp
from mindspore.common import set_seed
import mindspore.context as context
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.experimental import optim
from mindspore.nn import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.dataset as ds
import mindspore.ops.operations as P


parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--data_set', type=int, default=1)
parser.add_argument('--gpu', type=str, default='1')

args = parser.parse_args()
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")
project_path = './'
log_txt_path = project_path + 'log/' + time_str + 'set' + str(args.data_set) + '-log.txt'
log_curve_path = project_path + 'log/' + time_str + 'set' + str(args.data_set) + '-log.png'
checkpoint_path = project_path + 'checkpoint/' + time_str + 'set' + str(args.data_set) + '-model.pth'
best_checkpoint_path = project_path + 'checkpoint/' + time_str + 'set' + str(args.data_set) + '-model_best.pth'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def reset_seeds(universal_seed):
    set_seed(universal_seed)
    np.random.seed(universal_seed)

def main():


    mindspore.set_context(mode=context.GRAPH_MODE,
                    device_target="GPU",
                    device_id=0)

    if context.get_context("device_target") == "GPU":
        mindspore.set_seed(42)
        # context.set_context(seed=42)
    else:
        print(context.get_context("device_target"))
        print("Cuda not available. You need cuda to run this.")
        exit(0)
    # 
    best_acc = 0
    recorder = RecorderMeter(args.epochs)
    print('The training time: ' + now.strftime("%m-%d %H:%M"))
    print('The training set: set ' + str(args.data_set))
    with open(log_txt_path, 'a') as f:
        f.write('The training set: set ' + str(args.data_set) + '\n')


    model = GenerateModel()

    criterion = ASLSingleLabel(gamma_pos=1, gamma_neg=2)


    optimizer = optim.SGD(model.trainable_params(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.1)


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = load_checkpoint(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            best_acc = best_acc
            load_param_into_net(model, checkpoint)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # cudnn.benchmark = True
    

    # Data loading code
    train_data = train_data_loader(data_set=args.data_set)
    test_data = test_data_loader(data_set=args.data_set)



    train_loader = ds.GeneratorDataset(train_data, ["data", "label"], num_parallel_workers=args.workers)
    train_loader = train_loader.batch(args.batch_size, drop_remainder=True)
    train_loader = train_loader.shuffle(buffer_size=train_loader.get_dataset_size())

    
    val_loader = ds.GeneratorDataset(test_data, ["data", "label"], num_parallel_workers=args.workers)
    val_loader = val_loader.batch(args.batch_size, drop_remainder=False)



    for epoch in range(args.start_epoch, args.epochs):
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()

        print(inf)

        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args)

        val_acc, val_los = validate(val_loader, model, criterion, args)

        scheduler.step()

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch + 1,
                        'state_dict': model.get_parameters(),
                        'best_acc': best_acc,
                        'optimizer': optimizer.get_lr(),
                        'recorder': recorder}, is_best)

        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder.plot_curve(log_curve_path)

        print('The best accuracy: {:.3f}'.format(best_acc.item()))
        print('An epoch time: {:.1f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
            f.write('An epoch time: {:.1f}s' + str(epoch_time) + '\n')


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.set_train()

    for i, (images, target) in enumerate(train_loader):

        images = images.numpy()
        target = target.numpy()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.shape[0])
        top1.update(acc1[0], images.shape[0])

        # compute gradient and do SGD step
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg




def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.set_eval()

    pres_tr, trues_tr = [], []
    for i, (images, target) in enumerate(val_loader):
        images = images.numpy()
        target = target.numpy()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))

        _, pre_tr = np.argmax(output, 1)
        pres_tr += pre_tr.tolist()
        trues_tr += target.tolist()

        losses.update(loss.item(), images.shape[0])
        top1.update(acc1[0], images.shape[0])

        if i % args.print_freq == 0:
            progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    print('Current Accuracy: {:.3f}'.format(top1.avg))
    print('Current UAR: '+ str(get_UAR(trues_tr,pres_tr)))
    with open(log_txt_path, 'a') as f:
        f.write('Current Accuracy: {:.3f}'.format(top1.avg) + '\n')
        f.write('Current UAR: '+ str(get_UAR(trues_tr,pres_tr)) + '\n')
    return top1.avg, losses.avg


def save_checkpoint(state, is_best):
    mindspore.save_checkpoint(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]
    pred = output.argsort(axis=1)[:, ::-1][:, :maxk]
    pred = pred.transpose()
    correct = pred.equal(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).astype(np.float32).sum(0).astype(np.float32)
        res.append(correct_k * 100.0 / batch_size)
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



class LabelSmoothingCrossEntropy(nn.Cell):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.log_softmax = P.LogSoftmax(axis=-1)

    def construct(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = self.log_softmax(x)
        nll_loss = -logprobs.gather(target.unsqueeze(1), axis=-1)
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(axis=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

    
class ASLSingleLabel(nn.Cell):
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(axis=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
        self.scatter = ops.ScatterNd()

    def construct(self, inputs, target):
        num_classes = inputs.shape[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = ops.ZerosLike()(inputs)
        target = target.astype(np.int32)
        indices = np.arange(target.shape[0]).reshape(-1, 1)
        self.targets_classes = self.scatter(self.targets_classes, Tensor(indices), Tensor(target.reshape(-1, 1)), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = ops.Exp()(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = ops.Pow()(1 - xs_pos - xs_neg, self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes * (1 - self.eps) + self.eps / num_classes

        # loss calculation
        loss = - self.targets_classes * log_preds

        loss = ops.ReduceSum(keep_dims=False)(loss, -1)
        if self.reduction == 'mean':
            loss = ops.ReduceMean(keep_dims=False)(loss)

        return loss
if __name__ == '__main__':
    main()
