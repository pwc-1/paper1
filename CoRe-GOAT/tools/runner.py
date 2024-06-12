import os

from scipy import stats
from tools import builder, helper
from tools.trainer import Trainer
import time
from models.cnn_model import GCNnet_artisticswimming
from models.group_aware_attention import Encoder_Blocks
from utils.multi_gpu import *
from models.cnn_simplified import GCNnet_artisticswimming_simplified
from models.linear_for_bp import Linear_For_Backbone
from thop import profile

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

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
    base_model, regressor = builder.model_builder(args)
    # load checkpoints
    ms.load_checkpoint(args.ckpts + '/regressor.ckpt', regressor)
    # if using RT, build a group
    group = builder.build_group(train_dataset, args)
    gcn = GCNnet_artisticswimming_simplified(args)
    attn_encoder = Encoder_Blocks(args.qk_dim, 1024, args.linear_dim, args.num_heads, args.num_layers, args.attn_drop)
    linear_bp = Linear_For_Backbone(args)
    # load checkpoints
    ms.load_checkpoint(args.ckpts + '/gcn.ckpt', gcn)
    ms.load_checkpoint(args.ckpts + '/attn_encoder.ckpt', attn_encoder)
    ms.load_checkpoint(args.ckpts + '/linear_bp.ckpt', linear_bp)

    regressor.set_train(False)
    linear_bp.set_train(False)
    if args.use_goat:
        gcn.set_train(False)
        attn_encoder.set_train(False)

    # CUDA
    global use_gpu
    use_gpu = True

    #  DP
    # base_model = nn.DataParallel(base_model)
    # regressor = nn.DataParallel(regressor)

    test(base_model, regressor, test_dataloader, group, args, gcn, attn_encoder, linear_bp)


def run_net(args, logger):

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
                                                       num_parallel_workers=args.workers,
                                                       shuffle=False).batch(batch_size=args.bs_train)
    column_names = ["data"]
    for i in range(args.voter_number):
        column_names.append("target" + str(i))
    test_dataloader = ms.dataset.GeneratorDataset(test_dataset,
                                                  column_names=column_names,
                                                  num_parallel_workers=args.workers,
                                                  shuffle=False).batch(batch_size=args.bs_test)

    # Set data position
    device = get_device()

    # build model
    base_model, regressor = builder.model_builder(args)

    # input1 = ops.randn(2, 2049)
    # flops, params = profile(regressor, inputs=(input1, ))
    # print(f'[regressor]flops: ', flops, 'params: ', params)

    if args.warmup:
        num_steps = len(train_dataloader) * args.max_epoch
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        lr_scheduler = nn.cosine_decay_lr(1e-8, 0.01, num_steps, len(train_dataloader), args.max_epoch // 2)
    
    # Set models and optimizer(depend on whether to use goat)
    if args.use_goat:
        if args.use_cnn_features:
            gcn = GCNnet_artisticswimming_simplified(args)

            # input1 = ops.randn(1, 540, 8, 1024)
            # input2 = ops.randn(1, 540, 8, 4)
            # flops, params = profile(gcn, inputs=(input1, input2))
            # print(f'[GCNnet_artisticswimming_simplified]flops: ', flops, 'params: ', params)
        else:
            gcn = GCNnet_artisticswimming(args)
            gcn.loadmodel(args.stage1_model_path)
        attn_encoder = Encoder_Blocks(args.qk_dim, 1024, args.linear_dim, args.num_heads, args.num_layers, args.attn_drop)
        linear_bp = Linear_For_Backbone(args)
        optimizer = nn.Adam([
            {'params': gcn.trainable_params(), 'lr': args.lr * args.lr_factor},
            {'params': regressor.trainable_params()},
            {'params': linear_bp.trainable_params()},
            {'params': attn_encoder.trainable_params()}
        ], learning_rate=lr_scheduler if args.warmup else args.lr, weight_decay=args.weight_decay)
        scheduler = None
        if args.use_multi_gpu:
            wrap_model(gcn, distributed=args.distributed)
            wrap_model(attn_encoder, distributed=args.distributed)
            wrap_model(linear_bp, distributed=args.distributed)
            wrap_model(regressor, distributed=args.distributed)
        else:
            gcn = gcn
            attn_encoder = attn_encoder
            linear_bp = linear_bp
            regressor = regressor
    else:
        gcn = None
        attn_encoder = None
        linear_bp = Linear_For_Backbone(args)
        optimizer = nn.Adam([{'params': regressor.trainable_params()}, {'params': linear_bp.trainable_params()}], learning_rate=lr_scheduler if args.warmup else args.lr, weight_decay=args.weight_decay)
        scheduler = None
        if args.use_multi_gpu:
            wrap_model(regressor, distributed=args.distributed)
            wrap_model(linear_bp, distributed=args.distributed)
        else:
            regressor = regressor
            linear_bp = linear_bp


    # if using RT, build a group
    group = builder.build_group(train_dataset, args)
    # CUDA
    # global use_gpu
    # use_gpu = torch.cuda.is_available()
    # if use_gpu:
    #     torch.backends.cudnn.benchmark = True


    # parameter setting
    start_epoch = 1
    global epoch_best, rho_best, L2_min, RL2_min
    epoch_best = 0
    rho_best = 0
    L2_min = 1000
    RL2_min = 1000

    # resume ckpts
    if args.resume:
        start_epoch, epoch_best, rho_best, L2_min, RL2_min = \
            builder.resume_train(base_model, regressor, optimizer, args)
        print('resume ckpts @ %d epoch( rho = %.4f, L2 = %.4f , RL2 = %.4f)' % (
            start_epoch - 1, rho_best, L2_min, RL2_min))

    #  DP
    # regressor = nn.DataParallel(regressor)
    # if args.use_goat:
    #     gcn = nn.DataParallel(gcn)
    #     attn_encoder = nn.DataParallel(attn_encoder)

    # loss
    mse = nn.MSELoss()
    nll = nn.NLLLoss()
    
    trainer = Trainer(base_model, regressor, group, mse, nll, optimizer, args, gcn, attn_encoder, linear_bp)
    if is_main_process():
        print('Trainer start ... ')
    # trainval

    # training
    for epoch in range(start_epoch, args.max_epoch + 1):
        if args.use_multi_gpu:
            train_dataloader.sampler.set_epoch(epoch)
        true_scores = []
        pred_scores = []
        num_iter = 0
        # base_model.train()  # set model to training mode
        trainer.set_train()
        # if args.fix_bn:
        #     base_model.apply(misc.fix_bn)  # fix bn
        for idx, (data, target) in enumerate(train_dataloader):
            start = time.time()
            if args.bs_train == 1 or data['final_score'].shape == ():
                data = {k: v.unsqueeze(0) for k, v in data.items() if k != 'key'}
                target = {k: v.unsqueeze(0) for k, v in target.items() if k != 'key'}

            # break
            num_iter += 1
            opti_flag = False

            true_scores.extend(data['final_score'].numpy())
            # data preparing
            # featue_1 is the test video ; video_2 is exemplar
            if args.benchmark == 'MTL':
                feature_1 = data['feature'].float()  # N, C, T, H, W
                if args.usingDD:
                    label_1 = data['completeness'].float().reshape(-1, 1)
                    label_2 = target['completeness'].float().reshape(-1, 1)
                else:
                    label_1 = data['final_score'].float().reshape(-1, 1)
                    label_2 = target['final_score'].float().reshape(-1, 1)
                if not args.dive_number_choosing and args.usingDD:
                    assert (data['difficulty'] == target['difficulty']).all()
                diff = data['difficulty'].float().reshape(-1, 1)
                feature_2 = target['feature'].float()  # N, C, T, H, W

            else:
                raise NotImplementedError()

            # forward
            if num_iter == args.step_per_update:
                num_iter = 0
                opti_flag = True

            loss, leaf_probs_2, delta_2 = trainer.train_epoch(feature_1, label_1, feature_2, label_2, data, target, opti_flag)
            end = time.time()
            batch_time = end - start
            batch_idx = idx + 1
            if batch_idx % args.print_freq == 0:
                msg = '[Training][%d/%d][%d/%d] \t Batch_time %.2f \t Batch_loss: %.4f \t' % (epoch, args.max_epoch, batch_idx, len(train_dataloader), batch_time, loss.item())
                logger.info(msg)
                print(msg)

            # evaluate result of training phase
            relative_scores = group.inference(leaf_probs_2.numpy(), delta_2.numpy())
            if args.benchmark == 'MTL':
                if args.usingDD:
                    score = (relative_scores + label_2) * diff
                else:
                    score = relative_scores + label_2
            elif args.benchmark == 'Seven':
                score = relative_scores + label_2
            else:
                raise NotImplementedError()
            pred_scores.extend(score.numpy())

        # analysis on results
        pred_scores = np.array(pred_scores).squeeze()
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
              true_scores.shape[0]
        if is_main_process():
            msg = '[Training] EPOCH: %d, correlation: %.4f, L2: %.4f, RL2: %.4f' % (epoch, rho, L2, RL2)
            logger.info(msg)
            print(msg)


        if is_main_process():
            trainer.set_test()
            validate(trainer.base_model, trainer.regressor, test_dataloader, epoch, trainer.group, args, trainer.gcn, trainer.attn_encoder, trainer.linear_bp, logger)
            # helper.save_checkpoint(base_model, regressor, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min,
            #                        'last',
            #                        args)
            msg = '[TEST] EPOCH: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f' % (epoch, rho_best, L2_min, RL2_min)
            logger.info(msg)
            print(msg)
        # scheduler lr
        if scheduler is not None:
            scheduler.step()


# TODO: 修改以下所有;修改['difficulty'].float
def validate(base_model, regressor, test_dataloader, epoch, group, args, gcn, attn_encoder, linear_bp, logger):
    print("Start validating epoch {}".format(epoch))
    global use_gpu
    global epoch_best, rho_best, L2_min, RL2_min
    true_scores = []
    pred_scores = []
    # base_model.eval()  # set model to eval mode
    batch_num = len(test_dataloader)

    datatime_start = time.time()
    for batch_idx, data_list in enumerate(test_dataloader, 1):
        data = data_list[0]
        target = data_list[1:]
        if args.bs_test == 1 or data['final_score'].shape == ():
            data = {k: v.unsqueeze(0) for k, v in data.items() if k != 'key'}
            for i in range(len(target)):
                target[i] = {k: v.unsqueeze(0) for k, v in target[i].items() if k != 'key'}
        datatime = time.time() - datatime_start
        start = time.time()
        true_scores.extend(data['final_score'].numpy())
        # data prepare
        if args.benchmark == 'MTL':
            feature_1 = data['feature'].float()  # N, C, T, H, W
            if args.usingDD:
                label_2_list = [item['completeness'].float().reshape(-1, 1) for item in target]
            else:
                label_2_list = [item['final_score'].float().reshape(-1, 1) for item in target]
            diff = data['difficulty'].float().reshape(-1, 1)
            feature_2_list = [item['feature'].float() for item in target]
            # check
            if not args.dive_number_choosing and args.usingDD:
                for item in target:
                    assert (diff == item['difficulty'].reshape(-1, 1)).all()
        else:
            raise NotImplementedError()
        helper.network_forward_test(base_model, regressor, pred_scores, feature_1, feature_2_list, label_2_list,
                                    diff, group, args, data, target, gcn, attn_encoder, linear_bp)
        batch_time = time.time() - start
        if batch_idx % args.print_freq == 0:
            msg = '[TEST][%d/%d][%d/%d] \t Batch_time %.6f \t Data_time %.6f ' % (epoch, args.max_epoch, batch_idx, batch_num, batch_time, datatime)
            logger.info(msg)
            print(msg)
        datatime_start = time.time()

    # analysis on results
    pred_scores = np.array(pred_scores).squeeze()
    true_scores = np.array(true_scores)
    rho, p = stats.spearmanr(pred_scores, true_scores)
    L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
    RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
          true_scores.shape[0]
    if rho > 0.4 and RL2 < 0.06:
        os.makedirs(f'ckpts/{"I3D" if args.use_i3d_bb else "SWIN"}-lr{args.lr}-rho{rho:.4f}-rl{RL2 * 100:.4f}', exist_ok=True)
        ms.save_checkpoint(gcn,
                            f'ckpts/{"I3D" if args.use_i3d_bb else "SWIN"}-lr{args.lr}-rho{rho:.4f}-rl{RL2 * 100:.4f}/gcn.ckpt')
        ms.save_checkpoint(attn_encoder,
                            f'ckpts/{"I3D" if args.use_i3d_bb else "SWIN"}-lr{args.lr}-rho{rho:.4f}-rl{RL2 * 100:.4f}/attn_encoder.ckpt')
        ms.save_checkpoint(linear_bp,
                            f'ckpts/{"I3D" if args.use_i3d_bb else "SWIN"}-lr{args.lr}-rho{rho:.4f}-rl{RL2 * 100:.4f}/linear_bp.ckpt')
        ms.save_checkpoint(regressor,
                            f'ckpts/{"I3D" if args.use_i3d_bb else "SWIN"}-lr{args.lr}-rho{rho:.4f}-rl{RL2 * 100:.4f}/regressor.ckpt')
    if L2_min > L2:
        L2_min = L2
    if RL2_min > RL2:
        RL2_min = RL2
    if rho > rho_best:
        rho_best = rho
        epoch_best = epoch
        msg = '-----New best found!-----'
        logger.info(msg)
        print(msg)
        # helper.save_outputs(pred_scores, true_scores, args)
        # helper.save_checkpoint(base_model, regressor, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min,
        #                        'best', args)
    if epoch == args.max_epoch - 1:
        log_best(rho_best, RL2_min, epoch_best, args)

    msg = '[TEST] EPOCH: %d, correlation: %.6f, L2: %.6f, RL2: %.6f' % (epoch, rho, L2, RL2)
    logger.info(msg)
    print(msg)


def test(base_model, regressor, test_dataloader, group, args, gcn, attn_encoder, linear_bp):
    global use_gpu
    global epoch_best, rho_best, L2_min, RL2_min
    true_scores = []
    pred_scores = []
    # base_model.eval()  # set model to eval mode
    batch_num = len(test_dataloader)

    datatime_start = time.time()
    for batch_idx, data_list in enumerate(test_dataloader, 1):
        data = data_list[0]
        target = data_list[1:]
        if args.bs_test == 1 or data['final_score'].shape == ():
            data = {k: v.unsqueeze(0) for k, v in data.items() if k != 'key'}
            for i in range(len(target)):
                target[i] = {k: v.unsqueeze(0) for k, v in target[i].items() if k != 'key'}
        datatime = time.time() - datatime_start
        start = time.time()
        true_scores.extend(data['final_score'].numpy())
        # data prepare
        if args.benchmark == 'MTL':
            feature_1 = data['feature'].float()  # N, C, T, H, W
            if args.usingDD:
                label_2_list = [item['completeness'].float().reshape(-1, 1) for item in target]
            else:
                label_2_list = [item['final_score'].float().reshape(-1, 1) for item in target]
            diff = data['difficulty'].float().reshape(-1, 1)
            feature_2_list = [item['feature'].float() for item in target]
            # check
            if not args.dive_number_choosing and args.usingDD:
                for item in target:
                    assert (diff == item['difficulty'].reshape(-1, 1)).all()
        else:
            raise NotImplementedError()
        helper.network_forward_test(base_model, regressor, pred_scores, feature_1, feature_2_list, label_2_list,
                                    diff, group, args, data, target, gcn, attn_encoder, linear_bp)
        batch_time = time.time() - start
        if batch_idx % args.print_freq == 0:
            msg = '[TEST][%d/%d] \t Batch_time %.6f \t Data_time %.6f ' % (batch_idx, batch_num, batch_time, datatime)
            print(msg)
        datatime_start = time.time()

    # analysis on results
    pred_scores = np.array(pred_scores).squeeze()
    true_scores = np.array(true_scores)
    rho, p = stats.spearmanr(pred_scores, true_scores)
    L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
    RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
          true_scores.shape[0]

    msg = '[TEST] Ecorrelation: %.6f, L2: %.6f, RL2: %.6f' % (rho, L2, RL2)
    print(msg)
