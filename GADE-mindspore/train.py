import logging
import os
import time
import random
import numpy as np
from numpy import mean
import argparse
# from utils.dataset_generation import *
from logger import set_logger
from utils import *
import mindspore
from mindspore import nn
from mindspore.dataset import GeneratorDataset
from GADE_framework.GADE import GADE
import math

mindspore.set_context(device_target='GPU', device_id=1)


f1_list = []

def calculate_f1_(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    pred = (scores > 0.5).astype('int')

    TP = np.sum((pred == 1) * (labels == 1))
    TN = np.sum((pred == 0) * (labels == 0))
    FP = np.sum((pred == 1) * (labels == 0))
    FN = np.sum((pred == 0) * (labels == 1))
    acc = (TP + TN) * 1.0 / (TP + TN + FN + FP)
    if TP == 0:
        p = r = f1 = 0.0
    else:
        p = TP * 1.0 / (TP + FP)
        r = TP * 1.0 / (TP + FN)
        f1 = 2 * p * r / (p + r)

    return p, r, f1, acc

def test(iter, logger, model, batch_size, criterion, test_step=None, prefix='Test'):
    model.set_train(False)

    scores = []
    labels = []
    
    batch_count = math.ceil((len(iter)) / batch_size)
    batch_iter = []
    for bb in range(batch_count):
        batch_iter.append(iter[bb*batch_size:(bb+1)*batch_size])

    for j, batch in enumerate(batch_iter):
        
        pred, label, masks = model(batch)
        label = label.view(-1)
        loss = criterion(pred, label)
        pred = mindspore.ops.softmax(pred, 1)
        p, r, acc = accuracy(pred, label)

        pred_results = mindspore.ops.argmax(pred, 1).long()
        logger.info('entity:\t{}'.format(batch[0]["entity"]))
        for k in range(len(label)):
            s1 = "targeted document" if pred_results[k] == 1 else "non-targeted document"
            s2 = "targeted document" if label[k] == 1 else "non-targeted document"
            logger.info('{}\t[{:d}/{:d}]\tPrediction result: {}\tLabel: {}'.format(prefix, k+1, len(label), s1, s2))

        scores += list(pred[:,1].asnumpy())
        labels += list(label.asnumpy())

    p, r, f1, acc = calculate_f1_(scores, labels)
    
    return f1


def train(iter, checkpoint_path, logger, fold, model, optimizer, criterion, epoch_num, batch_size,
          start_epoch=0, test_iter=None, val_iter=None, log_freq=1, start_f1=None):

    step = 0
    if start_f1 is None:
        best_f1 = 0.0
    else:
        best_f1 = start_f1
    
    for i in range(start_epoch, epoch_num):
        model.set_train(True)

        batch_count = math.ceil((len(iter)) / batch_size)
        batch_iter = []
        for bb in range(batch_count):
            batch_iter.append(iter[bb*batch_size:(bb+1)*batch_size])
        
        for j, batch in enumerate(batch_iter):
            
            step += 1
        
            def forward_fn():
                pred, label, masks = model(batch)
                label = label.view(-1)
                loss = criterion(pred, label)
                return loss, pred, label

            grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
            (loss, pred, label), grads = grad_fn()
            grads = mindspore.ops.clip_by_value(grads, clip_value_max=1.0)
            loss = mindspore.ops.depend(loss, optimizer(grads))
            p, r, acc = accuracy(pred, label)
            
            if (j + 1) % log_freq == 0:
                logger.info(
                    'Train\tEpoch:[{:d}][{:d}/{:d}]\tLoss {:.3f}\tAccuracy {:.3f}\tPrecison {:.3f}\tRecall {:.3f}'.format(
                        i, j + 1, len(batch_iter), loss.asnumpy(), acc, p, r))

        if val_iter:
            f1_score = test(iter=val_iter, logger=logger, model=model, batch_size=batch_size, prefix='Val',
                      criterion=criterion, test_step=i + 1)
            if f1_score > best_f1:
                best_f1 = f1_score
                mindspore.save_checkpoint(model, os.path.join(checkpoint_path, "{}_best.ckpt".format(fold)))


    if test_iter:
        checkpoint = mindspore.load_checkpoint(os.path.join(checkpoint_path, "{}_best.ckpt".format(fold)))
        mindspore.load_param_into_net(model, checkpoint)
        f1_score = test(iter=test_iter, logger=logger, model=model, batch_size=batch_size, prefix='Test',
                      criterion=criterion, test_step=i + 1)
        f1_list.append(f1_score)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--seed', default=28, type=int)
    parser.add_argument('--exp_dir', default=".", type=str)
    parser.add_argument('--log_freq', default=5, type=int)
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--max_node', type=int, default=165)

    # Optimization args
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--embed_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float,default=0.4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)

    # Data path args
    parser.add_argument('--checkpoint_path', default="./saved_ckpt", type=str)
    parser.add_argument('--data_type', type=str, default='Wiki300')
    parser.add_argument('--model_name', default='GADE_300', type=str)

    # Device
    parser.add_argument('--gpu', type=int, default=[0,1], nargs='+')
    parser.add_argument('--gcn_layer', default=1, type=int)

    args = parser.parse_args()

    # mindspore.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    gcn_dim = 768
    kfold = args.kfold
    args.gcn_dim = gcn_dim

    params = args.__dict__

    args.entity_path = 'datasets/' + args.data_type + '/target_entities.txt'
    args.data_path = 'datasets/' + args.data_type + '/TDD_dataset.json'
    args.description_path = 'datasets/' + args.data_type + '/entity_desc.json'
    ent_list = load_entity_list(args.entity_path)

    for i in range(kfold):
        model = GADE(args)
        tokenizer = model.lrm.bert_tokenizer
        input_tokens, label_inputs, desc_tokens = generate_data(ent_list, args.description_path,
                                                                args.data_path, tokenizer,
                                                                args.max_seq_length)
        train_ent, val_ent, test_ent = get_kfold_data(ent_list, kfold, i)

        train_dataset = yield_example(train_ent, input_tokens, label_inputs, desc_tokens)
        val_dataset = yield_example(val_ent, input_tokens, label_inputs, desc_tokens)
        test_dataset = yield_example(test_ent, input_tokens, label_inputs, desc_tokens)

        criterion = nn.CrossEntropyLoss()
        
        no_decay = ['bias', 'LayerNorm.weight']
        
        lrm_params = list(filter(lambda x: 'lrm' in x.name, model.trainable_params()))
        gim_params = list(filter(lambda x: 'gim' in x.name, model.trainable_params()))
        optimizer_grouped_parameters = [
            {'params': lrm_params, 'weight_decay': 0.0, 'lr': args.embed_lr},
            {'params': gim_params, 'weight_decay': 0.0, 'lr': args.lr}
        ]
        opt = nn.Adam(optimizer_grouped_parameters, learning_rate=args.embed_lr)
        
        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
        # model_dir = args.exp_dir
        checkpoint_path = args.checkpoint_path + '/' + args.model_name
        log_dir = os.path.join(args.exp_dir, "logs")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        logger = set_logger(os.path.join(log_dir, str(time.time()) + "_" + args.model_name + ".log"))
        logger.info("The {}-th fold training begins!".format(i))

        start_epoch = 0
        start_f1 = 0.0
        
        train(train_dataset, checkpoint_path, logger, i, model, opt, criterion, args.epochs, args.batch_size, start_epoch, test_dataset, val_dataset, args.log_freq, start_f1)

