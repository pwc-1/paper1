import os
import time
import argparse
import socket
import numpy as np

import mindspore
from mindspore import nn
from mindspore.experimental import optim
from mindspore import ops
from utils.dataset import get_cifar100_dataloaders, get_tinyImagenet_dataloaders, get_cifar10_dataloaders

from utils import *

from models import build_mskd_backbone, build_resnetv1_backbone


def parse_option():
    parser = argparse.ArgumentParser(description='ReviewKD for Self Fusion')

    # basic
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=240, help='number of epochs to train (default: 240)')
    parser.add_argument('--deterministic', type=bool, default=True, help='Make results reproducible')
    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu is used for training')
    parser.add_argument('--suffix', type=str, default='', help='the experiment label')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')

    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar100', 'cifar10', 'tiny-imagenet'])
    # optimization
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='learning rate decay ratio')
    parser.add_argument('--lr_decay_epochs', default=[150, 180, 210, 240], type=int, nargs='+',
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')

    # models / student
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet8x4', 'wrn-16-2', 'wrn-40-1', 'shufflev1',
                                 'shufflev2', 'resnet34', 'resnet50', 'resnet18', 'mobile', 'vgg8'])
    parser.add_argument('--teacher', type=str, default='resnet34', help='teacher models')
    parser.add_argument('--teacher_weight', type=str, default='./ResNet34_ckpt/resnet34-f297d27e.ckpt',
                        help='teacher models weight path')

    # distill
    parser.add_argument('--kd_loss_weight', type=float, default=1.0, help='review kd loss weight')
    parser.add_argument('--kd_warm_up', type=float, default=20.0,
                        help='feature knowledge distillation loss weight warm up epochs')

    parser.add_argument('--distill', type=str, default='relation', choices=['kd', 'relation', 'reviewkd'],
                        help='which type of distll for using')
    parser.add_argument('--kl_loss_weight', type=float, default=1.0, help='kl konwledge distillation loss weight')
    parser.add_argument('-T', type=float, default=4.0, help='knowledge distillation loss temperature')

    parser.add_argument('--ce_loss_weight', type=float, default=1.0, help='cross entropy loss weight')

    args = parser.parse_args()
    print(args)

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if not os.path.exists('logs/'):
        os.mkdir('logs/')
    if not os.path.exists('checkpoints/'):
        os.mkdir('checkpoints/')
    if not os.path.isfile('checkpoints/exp_record.txt'):
        open('checkpoints/exp_record.txt', 'w')

    if args.deterministic:
        mindspore.set_seed(12345)
        mindspore.set_context(mode=mindspore.PYNATIVE_MODE,
                              device_target="GPU",
                              device_id=int(args.gpu_id),
                              gpu_config={"conv_fprop_algo": "normal"})

    if 'shuffle' in args.model or 'mobile' in args.model:
        args.lr = 0.02

    return args


def validate(val_loader, model):
    """validation"""
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.set_train(False)
    for idx, (image, target) in enumerate(val_loader):
        # compute output
        _, output = model(image, is_feat=True)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.shape[0])
        top5.update(acc5[0], image.shape[0])

    return top1.avg


def main():
    args = parse_option()

    if args.teacher == '':
        test_id = 'baseline/{}_{}_{}'.format(args.model, args.dataset, args.suffix)
    else:
        test_id = 'student/S_{}_T_{}_{}_ce_{}_kl_{}_kd_{}_{}_{}'.format(
            args.model, args.teacher, args.dataset, args.ce_loss_weight,
            args.kl_loss_weight, args.kd_loss_weight, args.distill, args.suffix)

    filename = 'logs/' + test_id + '.txt'
    logger = Logger(args=args, filename=filename)

    # dataset
    if args.dataset == 'cifar10':
        num_classes = 10
        train_loader, test_loader = get_cifar10_dataloaders(args)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_loader, test_loader = get_cifar100_dataloaders(args)
    elif args.dataset == 'tiny-imagenet200':
        num_classes = 200
        train_loader, test_loader = get_tinyImagenet_dataloaders(args)
    else:
        raise RuntimeError("the specified dataset is not exist")

    # teacher models
    if args.teacher in ['resnet18', 'resnet34', 'resnet50']:
        teacher = build_resnetv1_backbone(depth=int(args.teacher[6:]), num_classes=num_classes)
        teacher.set_train(False)
    elif args.teacher == '':
        teacher = None
    else:
        raise NameError('The specified teacher is not support')

    # load teacher dict
    if teacher is not None:
        load_teacher_weight(teacher, args.teacher_weight)

    # defining student models or baseline
    if teacher is not None:
        cnn = build_mskd_backbone(args, num_classes=num_classes)
    else:
        raise NameError('The specified model does not support')
    cnn.set_train()

    cls_criterion = nn.CrossEntropyLoss()
    if args.distill == 'kd':
        kl_criterion = DistillKL(args.T)
    elif args.distill == 'relation':                   # method in this paper
        kl_criterion = ReLoss(num_classes)
    else:
        kl_criterion = None

    def forward_fn(images, labels):
        losses = {}
        if teacher is not None:
            s_features, pred = cnn(images, is_feat=True)
            t_features, t_pred = teacher(images, is_feat=True)  # preact=True
            # t_features = [t_fea.detach() for t_fea in t_features[1:]]

            feature_kd_loss = hcl(s_features, t_features)

            losses['fkd_loss'] = feature_kd_loss * min(1, epoch / args.kd_warm_up) * args.kd_loss_weight  # Lfd

            if args.distill == 'kd':
                losses['lkd_loss'] = kl_criterion(pred, t_pred) * args.kl_loss_weight
            elif args.distill == 'relation':
                losses['lkd_loss'] = kl_criterion(labels, pred, t_pred) * args.kl_loss_weight  # Lsd
        else:
            pred = cnn(images)

        ce_loss = cls_criterion(pred, labels)  # Lce
        losses['cls_loss'] = ce_loss * args.ce_loss_weight

        loss = sum(losses.values())  # overall loss
        return loss, pred

    optimizer = optim.SGD(params=cnn.trainable_params(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_epochs, 0.1, last_epoch=-1)
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    if teacher is not None:
        val_acc = validate(test_loader, teacher)
        print("teacher acc is: {}".format(val_acc))

    # Train ############################################################################################################
    best_acc = 0.0
    st_time = time.time()
    for epoch in range(args.epochs):
        loss_avg = {}
        top1 = AverageMeter()

        for i, (images, labels) in enumerate(train_loader):  # loop once is a batch

            labels = labels.astype(mindspore.int32)
            (loss, pred), grads = grad_fn(images, labels)
            optimizer(grads)

            # for key in losses:
            #     if key not in loss_avg:
            #         loss_avg[key] = AverageMeter()
            #     else:
            #         loss_avg[key].update(losses[key])   # calculate average of loss

            # calculate running average of accuracy
            acc1 = accuracy(pred, labels, topk=(1, ))
            top1.update(acc1[0], labels.shape[0])

        test_acc = validate(test_loader, cnn)

        if test_acc > best_acc:
            best_acc = test_acc
            mindspore.save_checkpoint(cnn, 'checkpoints/'+test_id+'_best.pt')   # save parameters

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        # loss_avg = {k: loss_avg[k].val for k in loss_avg}   # average loss of the last batch

        row = {'epoch': str(epoch),
               'train_acc': '%.2f' % top1.avg,
               'test_acc': '%.2f' % test_acc,
               'best_acc': '%.2f' % best_acc,
               'lr': '%.5f' % lr,
               'loss': '%.5f' % (loss),
               }
        # loss_avg = {k: '%.5f' % loss_avg[k] for k in loss_avg}   # format as string
        # row.update(loss_avg)

        row.update({
            'time': format_time(time.time() - st_time),
            'eta': format_time((time.time() - st_time) / (epoch + 1) * (args.epochs - epoch - 1)),
        })   # used time, remaining time
        print(row)
        logger.writerow(row)

    logger.close()

    # save every exp result
    exp_record = open('./checkpoints/exp_record.txt', 'a+')
    exp_record.write('Model Name:{} \t Best Acc:{} \t Hostname:{} \n'.format(test_id, best_acc, socket.gethostname()))
    exp_record.close()


if __name__ == '__main__':
    main()
