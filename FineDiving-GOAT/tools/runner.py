import os

import numpy as np

from scipy import stats
from tools import builder, helper
from tools.trainer import Trainer
from utils import misc
import time
import pickle
from models.cnn_model import GCNnet_artisticswimming
from models.cnn_simplified import GCNnet_artisticswimming_simplified
from models.group_aware_attention import Encoder_Blocks
from utils.goat_utils import *
from models.linear_for_bp import Linear_For_Backbone
from thop import profile

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from utils.misc import segment_iou, cal_tiou, seg_pool_1d, seg_pool_3d


def train_net(args, logger):
    # build dataset
    train_dataset, test_dataset = builder.dataset_builder(args)
    if args.use_multi_gpu:
        train_dataloader = build_dataloader(train_dataset,
                                            batch_size=args.bs_train,
                                            shuffle=True,
                                            num_workers=args.workers,
                                            persistent_workers=True,
                                            seed=set_seed(args.seed))
    else:
        train_dataloader = ms.dataset.GeneratorDataset(train_dataset,
                                                       column_names=["data", "target"],
                                                       num_parallel_workers=int(args.workers),
                                                       shuffle=False).batch(batch_size=args.bs_train)

    column_names = ["data"]
    for i in range(args.voter_number):
        column_names.append("target" + str(i))
    test_dataloader = ms.dataset.GeneratorDataset(test_dataset,
                                                  column_names=column_names,
                                                  num_parallel_workers=int(args.workers),
                                                  shuffle=False).batch(batch_size=args.bs_test)

    # Set data position
    # if torch.cuda.is_available():
    #     device = get_device()
    # else:
    #     device = torch.device('cpu')


    # build model
    base_model, psnet_model, decoder, regressor_delta = builder.model_builder(args)

    # input1 = ops.randn(2, 9, 1024)
    # input2 = ops.randn(1, 15, 64)
    # input3 = ops.randn(1, 15, 64)
    # input4 = ops.randn(1, 15, 64)
    # flops, params = profile(psnet_model, inputs=(input1, ))
    # print(f'[psnet_model]flops: ', flops, 'params: ', params)
    # flops, params = profile(decoder, inputs=(input2, input3))
    # print(f'[decoder]flops: ', flops, 'params: ', params)
    # flops, params = profile(regressor_delta, inputs=(input4, ))
    # print(f'[regressor_delta]flops: ', flops, 'params: ', params)

    # Set models and optimizer(depend on whether to use goat)
    if args.use_goat:
        if args.use_cnn_features:
            gcn = GCNnet_artisticswimming_simplified(args)
        else:
            gcn = GCNnet_artisticswimming(args)
            gcn.loadmodel(args.stage1_model_path)
        attn_encoder = Encoder_Blocks(args.qk_dim, 1024, args.linear_dim, args.num_heads, args.num_layers, args.attn_drop)
        linear_bp = Linear_For_Backbone(args)
        optimizer = nn.Adam([
            {'params': gcn.trainable_params()},
            {'params': attn_encoder.trainable_params()},
            {'params': psnet_model.trainable_params()},
            {'params': decoder.trainable_params()},
            {'params': linear_bp.trainable_params()},
            {'params': regressor_delta.trainable_params()}
        ], learning_rate=args.lr, weight_decay=args.weight_decay)
        scheduler = None
        if args.use_multi_gpu:
            wrap_model(gcn, distributed=args.distributed)
            wrap_model(attn_encoder, distributed=args.distributed)
            wrap_model(psnet_model, distributed=args.distributed)
            wrap_model(decoder, distributed=args.distributed)
            wrap_model(linear_bp, distributed=args.distributed)
            wrap_model(regressor_delta, distributed=args.distributed)
        else:
            gcn = gcn
            attn_encoder = attn_encoder
            psnet_model = psnet_model
            decoder = decoder
            linear_bp = linear_bp
            regressor_delta = regressor_delta
    else:
        gcn = None
        attn_encoder = None
        linear_bp = Linear_For_Backbone(args)
        optimizer = nn.Adam([
            {'params': psnet_model.trainable_params()},
            {'params': decoder.trainable_params()},
            {'params': linear_bp.trainable_params()},
            {'params': regressor_delta.trainable_params()}
        ], learning_rate=args.lr, weight_decay=args.weight_decay)
        scheduler = None
        if args.use_multi_gpu:
            wrap_model(psnet_model, distributed=args.distributed)
            wrap_model(decoder, distributed=args.distributed)
            wrap_model(regressor_delta, distributed=args.distributed)
            wrap_model(linear_bp, distributed=args.distributed)
        else:
            psnet_model = psnet_model
            decoder = decoder
            regressor_delta = regressor_delta
            linear_bp = linear_bp

    if args.warmup:
        num_steps = len(train_dataloader) * args.max_epoch
        lr_scheduler = nn.cosine_decay_lr(total_step=num_steps, min_lr=0, decay_epoch=-1, max_lr=0.1, step_per_epoch=2)

    start_epoch = 0
    global epoch_best_tas, pred_tious_best_5, pred_tious_best_75, epoch_best_aqa, rho_best, L2_min, RL2_min
    epoch_best_tas = 0
    pred_tious_best_5 = 0
    pred_tious_best_75 = 0
    epoch_best_aqa = 0
    rho_best = 0
    L2_min = 1000
    RL2_min = 1000

    # resume ckpts
    if args.resume:
        start_epoch, epoch_best_aqa, rho_best, L2_min, RL2_min = builder.resume_train(base_model, psnet_model, decoder,
                                                                                      regressor_delta, optimizer, args)
        print('resume ckpts @ %d epoch(rho = %.4f, L2 = %.4f , RL2 = %.4f)'
              % (start_epoch - 1, rho_best, L2_min, RL2_min))

    # DP
    # base_model = nn.DataParallel(base_model)
    # psnet_model = nn.DataParallel(psnet_model)
    # decoder = nn.DataParallel(decoder)
    # regressor_delta = nn.DataParallel(regressor_delta)

    # loss
    mse = nn.MSELoss()
    bce = nn.BCELoss()
    trainer = Trainer(base_model, psnet_model, decoder, regressor_delta, mse, optimizer, args, bce, gcn, attn_encoder, linear_bp)

    if is_main_process():
        print('Trainer start ... ')
    # training phase
    for epoch in range(start_epoch, args.max_epoch):
        if args.use_multi_gpu:
            train_dataloader.sampler.set_epoch(epoch)
        pred_tious_5 = []
        pred_tious_75 = []
        true_scores = []
        pred_scores = []

        trainer.set_train()

        # if args.fix_bn:
        #     base_model.apply(misc.fix_bn)
        for idx, (data, target) in enumerate(train_dataloader):
            if args.bs_train == 1 or data['final_score'].shape == ():
                data = {k: v.unsqueeze(0) for k, v in data.items() if k != 'key'}
                target = {k: v.unsqueeze(0) for k, v in target.items() if k != 'key'}
            # num_iter += 1
            opti_flag = True

            # video_1 is query and video_2 is exemplar
            feature_1 = data['feature'].float()
            feature_2 = target['feature'].float()
            feamap_1 = data['feamap'].float()
            feamap_2 = target['feamap'].float()
            label_1_tas = data['transits'].float() + 1
            label_2_tas = target['transits'].float() + 1
            label_1_score = data['final_score'].float().reshape(-1, 1)
            label_2_score = target['final_score'].float().reshape(-1, 1)

            # forward
            # helper.network_forward_train(base_model, psnet_model, decoder, regressor_delta, pred_scores,
            #                              feature_1, label_1_score, feature_2, label_2_score, mse, optimizer,
            #                              opti_flag, epoch, idx + 1, len(train_dataloader),
            #                              args, label_1_tas, label_2_tas, bce,
            #                              pred_tious_5, pred_tious_75, feamap_1, feamap_2, data, target, gcn,
            #                              attn_encoder, linear_bp)
            start = time.time()
            loss, score, Batch_tIoU_5, Batch_tIoU_75 = trainer.train_epoch(feature_1, label_1_score, feature_2, label_2_score, epoch, label_1_tas, label_2_tas, feamap_1, feamap_2, data, target)
            end = time.time()
            batch_time = end - start
            pred_scores.extend(score.numpy())
            pred_tious_5.extend([Batch_tIoU_5])
            pred_tious_75.extend([Batch_tIoU_75])

            batch_idx = idx + 1
            if batch_idx % args.print_freq == 0:
                msg = '[Training][%d/%d][%d/%d] \t Batch_time: %.2f \t Batch_loss: %.4f \t '% (epoch, args.max_epoch, batch_idx, len(train_dataloader), batch_time, loss.item())
                print(msg)
                logger.info(msg)
            true_scores.extend(data['final_score'].numpy())

        # evaluation results
        pred_scores = np.array(pred_scores).squeeze()
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
              true_scores.shape[0]
        pred_tious_mean_5 = sum(pred_tious_5) / len(train_dataset)
        pred_tious_mean_75 = sum(pred_tious_75) / len(train_dataset)

        if is_main_process():
            msg = '[Training] EPOCH: %d, tIoU_5: %.4f, tIoU_75: %.4f' % (epoch, pred_tious_mean_5, pred_tious_mean_75)
            print(msg)
            logger.info(msg)

            msg = '[Training] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f' % (epoch, rho, L2, RL2)
            print(msg)
            logger.info(msg)

            trainer.set_test()
            validate(trainer.base_model, trainer.psnet_model, trainer.decoder, trainer.regressor_delta, test_dataloader, epoch, trainer.optimizer, args, trainer.gcn,
                 trainer.attn_encoder, trainer.linear_bp, logger)

            msg = '[TEST] EPOCH: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f' % (epoch_best_aqa, rho_best, L2_min, RL2_min)
            print(msg)
            logger.info(msg)

            msg = '[TEST] EPOCH: %d, best tIoU_5: %.6f, best tIoU_75: %.6f' % (epoch_best_tas, pred_tious_best_5, pred_tious_best_75)
            print(msg)
            logger.info(msg)

        # scheduler lr
        if scheduler is not None:
            scheduler.step()


def validate(base_model, psnet_model, decoder, regressor_delta, test_dataloader, epoch, optimizer, args, gcn,
             attn_encoder, linear_bp, logger):
    print("Start validating epoch {}".format(epoch))
    global use_gpu
    global epoch_best_aqa, rho_best, L2_min, RL2_min, epoch_best_tas, pred_tious_best_5, pred_tious_best_75

    true_scores = []
    pred_scores = []
    pred_tious_test_5 = []
    pred_tious_test_75 = []

    # base_model.eval()
    # psnet_model.set_train(False)
    # decoder.set_train(False)
    # regressor_delta.set_train(False)
    # linear_bp.set_train(False)
    # if args.use_goat:
    #     gcn.set_train(False)
    #     attn_encoder.set_train(False)

    batch_num = len(test_dataloader)
    # with torch.no_grad():
    datatime_start = time.time()

    for batch_idx, data_list in enumerate(test_dataloader, 0):
        data = data_list[0]
        target = data_list[1:]
        if args.bs_test == 1 or data['final_score'].shape == ():
            data = {k: v.unsqueeze(0) for k, v in data.items() if k != 'key'}
            for i in range(len(target)):
                target[i] = {k: v.unsqueeze(0) for k, v in target[i].items() if k != 'key'}
        datatime = time.time() - datatime_start
        start = time.time()

        # video_1 = data['video'].float().cuda()
        feature_1 = data['feature'].float()
        feamap_1 = data['feamap'].float()
        # video_2_list = [item['video'].float().cuda() for item in target]
        feature_2_list = [item['feature'].float() for item in target]
        feamap_2_list = [item['feamap'].float() for item in target]
        label_1_tas = data['transits'].float() + 1
        label_2_tas_list = [item['transits'].float() + 1 for item in target]
        label_2_score_list = [item['final_score'].float().reshape(-1, 1) for item in target]

        helper.network_forward_test(base_model, psnet_model, decoder, regressor_delta, pred_scores,
                                    feature_1, feature_2_list, label_2_score_list,
                                    args, label_1_tas, label_2_tas_list,
                                    pred_tious_test_5, pred_tious_test_75, feamap_1, feamap_2_list, data, target,
                                    gcn, attn_encoder, linear_bp)

        batch_time = time.time() - start
        if batch_idx % args.print_freq == 0:
            msg = '[TEST][%d/%d][%d/%d] \t Batch_time %.6f \t Data_time %.6f' % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, datatime)
            print(msg)
            logger.info(msg)
        datatime_start = time.time()
        true_scores.extend(data['final_score'].numpy())

    # evaluation results
    pred_scores = np.array(pred_scores).squeeze()
    true_scores = np.array(true_scores)
    rho, p = stats.spearmanr(pred_scores, true_scores)
    L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
    RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
          true_scores.shape[0]
    pred_tious_test_mean_5 = sum(pred_tious_test_5) / (len(test_dataloader) * args.bs_test)
    pred_tious_test_mean_75 = sum(pred_tious_test_75) / (len(test_dataloader) * args.bs_test)

    if pred_tious_test_mean_5 > pred_tious_best_5:
        pred_tious_best_5 = pred_tious_test_mean_5
    if pred_tious_test_mean_75 > pred_tious_best_75:
        pred_tious_best_75 = pred_tious_test_mean_75
        epoch_best_tas = epoch
    msg = '[TEST] EPOCH: %d, tIoU_5: %.6f, tIoU_75: %.6f' % (epoch, pred_tious_best_5, pred_tious_best_75)
    print(msg)
    logger.info(msg)

    if L2_min > L2:
        L2_min = L2
    if RL2_min > RL2:
        RL2_min = RL2
    if rho > 0.4 and RL2 < 0.06:
        os.makedirs(f'ckpts/{"I3D" if args.use_i3d_bb else "SWIN"}-lr{args.lr}-rho{rho:.4f}-rl{RL2 * 100:.4f}',
                    exist_ok=True)
        ms.save_checkpoint(gcn,
                           f'ckpts/{"I3D" if args.use_i3d_bb else "SWIN"}-lr{args.lr}-rho{rho:.4f}-rl{RL2 * 100:.4f}/gcn.ckpt')
        ms.save_checkpoint(attn_encoder,
                           f'ckpts/{"I3D" if args.use_i3d_bb else "SWIN"}-lr{args.lr}-rho{rho:.4f}-rl{RL2 * 100:.4f}/attn_encoder.ckpt')
        ms.save_checkpoint(linear_bp,
                           f'ckpts/{"I3D" if args.use_i3d_bb else "SWIN"}-lr{args.lr}-rho{rho:.4f}-rl{RL2 * 100:.4f}/linear_bp.ckpt')
        ms.save_checkpoint(regressor_delta,
                           f'ckpts/{"I3D" if args.use_i3d_bb else "SWIN"}-lr{args.lr}-rho{rho:.4f}-rl{RL2 * 100:.4f}/regressor_delta.ckpt')
        ms.save_checkpoint(decoder,
                           f'ckpts/{"I3D" if args.use_i3d_bb else "SWIN"}-lr{args.lr}-rho{rho:.4f}-rl{RL2 * 100:.4f}/decoder.ckpt')
        ms.save_checkpoint(psnet_model,
                           f'ckpts/{"I3D" if args.use_i3d_bb else "SWIN"}-lr{args.lr}-rho{rho:.4f}-rl{RL2 * 100:.4f}/psnet_model.ckpt')
    if rho > rho_best:
        rho_best = rho
        epoch_best_aqa = epoch
        msg = '-----New best found!-----'
        print(msg)
        logger.info(msg)
        # helper.save_outputs(pred_scores, true_scores, args)
        # helper.save_checkpoint(base_model, psnet_model, decoder, regressor_delta, optimizer, epoch, epoch_best_aqa,
        #                        rho_best, L2_min, RL2_min, 'last', args)
    if epoch == args.max_epoch - 1:
        log_best(rho_best, RL2_min, epoch_best_aqa, args)

    msg = '[TEST] EPOCH: %d, correlation: %.6f, L2: %.6f, RL2: %.6f' % (epoch, rho, L2, RL2)
    print(msg)
    logger.info(msg)


def test_net(args):
    print('Tester start ... ')

    train_dataset, test_dataset = builder.dataset_builder(args)
    column_names = ["data"]
    for i in range(args.voter_number):
        column_names.append("target" + str(i))
    test_dataloader = ms.dataset.GeneratorDataset(test_dataset,
                                                  column_names=column_names,
                                                  num_parallel_workers=int(args.workers),
                                                  shuffle=False).batch(batch_size=args.bs_test)

    # Set data position
    # if torch.cuda.is_available():
    #     device = get_device()
    # else:
    #     device = torch.device('cpu')
    ms.set_context(device_target='GPU', device_id=0)

    # build model
    base_model, psnet_model, decoder, regressor_delta = builder.model_builder(args)
    if args.use_cnn_features:
        gcn = GCNnet_artisticswimming_simplified(args)
    else:
        gcn = GCNnet_artisticswimming(args)
        gcn.loadmodel(args.stage1_model_path)
    attn_encoder = Encoder_Blocks(args.qk_dim, 1024, args.linear_dim, args.num_heads, args.num_layers, args.attn_drop)
    linear_bp = Linear_For_Backbone(args)

    # load checkpoints
    ms.load_checkpoint(args.ckpts + '/attn_encoder.ckpt', attn_encoder)
    ms.load_checkpoint(args.ckpts + '/gcn.ckpt', gcn)
    ms.load_checkpoint(args.ckpts + '/linear_bp.ckpt', linear_bp)
    ms.load_checkpoint(args.ckpts + '/regressor_delta.ckpt', regressor_delta)
    ms.load_checkpoint(args.ckpts + '/decoder.ckpt', decoder)
    ms.load_checkpoint(args.ckpts + '/psnet_model.ckpt', psnet_model)

    # CUDA
    # global use_gpu
    # use_gpu = torch.cuda.is_available()
    # if use_gpu:
    #     # base_model = base_model.cuda()
    #     psnet_model = psnet_model.cuda()
    #     decoder = decoder.cuda()
    #     regressor_delta = regressor_delta.cuda()
    #     torch.backends.cudnn.benchmark = True

    # DP
    # base_model = nn.DataParallel(base_model)
    # psnet_model = nn.DataParallel(psnet_model)
    # decoder = nn.DataParallel(decoder)
    # regressor_delta = nn.DataParallel(regressor_delta)

    test(base_model, psnet_model, decoder, regressor_delta, test_dataloader, args, gcn, attn_encoder, linear_bp)


def test(base_model, psnet_model, decoder, regressor_delta, test_dataloader, args, gcn, attn_encoder, linear_bp):
    global use_gpu
    global epoch_best_aqa, rho_best, L2_min, RL2_min
    global epoch_best_tas, pred_tious_best_5, pred_tious_best_75

    true_scores = []
    pred_scores = []
    pred_tious_test_5 = []
    pred_tious_test_75 = []

    # base_model.eval()
    psnet_model.set_train(False)
    decoder.set_train(False)
    regressor_delta.set_train(False)
    if args.use_goat:
        gcn.set_train(False)
        attn_encoder.set_train(False)
        linear_bp.set_train(False)

    batch_num = len(test_dataloader)
    # with torch.no_grad():
    datatime_start = time.time()

    for batch_idx, data_list in enumerate(test_dataloader, 0):
        data = data_list[0]
        target = data_list[1:]
        if args.bs_test == 1 or data['final_score'].shape == ():
            data = {k: v.unsqueeze(0) for k, v in data.items() if k != 'key'}
            for i in range(len(target)):
                target[i] = {k: v.unsqueeze(0) for k, v in target[i].items() if k != 'key'}
        datatime = time.time() - datatime_start
        start = time.time()

        # video_1 = data['video'].float().cuda()
        feature_1 = data['feature'].float()
        feamap_1 = data['feamap'].float()
        # video_2_list = [item['video'].float().cuda() for item in target]
        feature_2_list = [item['feature'].float() for item in target]
        feamap_2_list = [item['feamap'].float() for item in target]
        label_1_tas = data['transits'].float() + 1
        label_2_tas_list = [item['transits'].float() + 1 for item in target]
        label_2_score_list = [item['final_score'].float().reshape(-1, 1) for item in target]

        helper.network_forward_test(base_model, psnet_model, decoder, regressor_delta, pred_scores,
                                    feature_1, feature_2_list, label_2_score_list,
                                    args, label_1_tas, label_2_tas_list,
                                    pred_tious_test_5, pred_tious_test_75, feamap_1, feamap_2_list, data, target,
                                    gcn, attn_encoder, linear_bp)

        batch_time = time.time() - start
        if batch_idx % args.print_freq == 0:
            print('[TEST][%d/%d] \t Batch_time %.2f \t Data_time %.2f'
                  % (batch_idx, batch_num, batch_time, datatime))
        datatime_start = time.time()
        true_scores.extend(data['final_score'].numpy())

    # evaluation results
    pred_scores = np.array(pred_scores).squeeze()
    true_scores = np.array(true_scores)
    rho, p = stats.spearmanr(pred_scores, true_scores)
    L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
    RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
          true_scores.shape[0]
    pred_tious_test_mean_5 = sum(pred_tious_test_5) / (len(test_dataloader) * args.bs_test)
    pred_tious_test_mean_75 = sum(pred_tious_test_75) / (len(test_dataloader) * args.bs_test)

    print('[TEST] tIoU_5: %.6f, tIoU_75: %.6f' % (pred_tious_test_mean_5, pred_tious_test_mean_75))
    print('[TEST] correlation: %.6f, L2: %.6f, RL2: %.6f' % (rho, L2, RL2))
