import pickle
import sys
# from thop import profile

sys.path.append('../')

from opts import *
from scipy import stats
from dataset import VideoDataset
from models.evaluator import Evaluator
from config import get_parser
import time
from models.cnn_model import GCNnet_artisticswimming
from models.cnn_simplified import GCNnet_artisticswimming_simplified
from models.group_aware_attention import Encoder_Blocks
from utils import *
from models.linear_for_bp import Linear_For_Backbone

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = True


def get_models(args):
    """
    Get the i3d backbone and the evaluator with parameters moved to GPU.
    """
    # i3d = InceptionI3d().cuda()
    # i3d.load_state_dict(torch.load(i3d_pretrained_path))
    i3d = 0

    if args.type == 'USDL':
        evaluator = Evaluator(output_dim=output_dim['USDL'], model_type='USDL')
    else:
        evaluator = Evaluator(output_dim=output_dim['MUSDL'], model_type='MUSDL', num_judges=num_judges)

    # if len(args.gpu.split(',')) > 1:
    #     # i3d = nn.DataParallel(i3d)
    #     evaluator = nn.DataParallel(evaluator)
    return i3d, evaluator


def compute_score(model_type, probs, data):
    if model_type == 'USDL':
        pred = probs.argmax(axis=-1) * (label_max / (output_dim['USDL'] - 1))
    else:
        # calculate expectation & denormalize & sort
        judge_scores_pred = ops.stack([prob.argmax(axis=-1) * judge_max / (output_dim['MUSDL'] - 1)
                                       for prob in probs], axis=1).sort()[0]  # N, 7

        # keep the median 3 scores to get final score according to the rule of diving
        pred = ops.sum(judge_scores_pred[:, 2:5], dim=1) * data['difficulty']
    return pred


def compute_loss(model_type, criterion, probs, data):
    if model_type == 'USDL':
        loss = criterion(ops.log(probs), data['soft_label'])
    else:
        loss = sum([criterion(ops.log(probs[i]), data['soft_judge_scores'][:, i]) for i in range(num_judges)])
    return loss


def get_dataloaders(args):
    dataloaders = {}
    boxes_dict = pickle.load(open(args.boxes_path, 'rb'))
    # dataloaders['test'] = torch.utils.data.DataLoader(VideoDataset('test', args),
    #                                                   batch_size=args.test_batch_size,
    #                                                   num_workers=args.num_workers,
    #                                                   shuffle=False,
    #                                                   pin_memory=True,
    #                                                   worker_init_fn=worker_init_fn)
    dataloaders['test'] = ms.dataset.GeneratorDataset(VideoDataset('test', args, boxes_dict),
                                                      column_names=["data"],
                                                      num_parallel_workers=args.num_workers,
                                                      shuffle=False).batch(batch_size=args.test_batch_size)

    if args.use_multi_gpu:
        dataloaders['train'] = build_dataloader(VideoDataset('train', args, boxes_dict),
                                                batch_size=args.train_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                persistent_workers=True,
                                                seed=set_seed(args.seed))
    else:
        # dataloaders['train'] = torch.utils.data.DataLoader(VideoDataset('train', args),
        #                                                    batch_size=args.train_batch_size,
        #                                                    num_workers=args.num_workers,
        #                                                    shuffle=False,
        #                                                    pin_memory=False,
        #                                                    worker_init_fn=worker_init_fn)
        dataloaders['train'] = ms.dataset.GeneratorDataset(VideoDataset('train', args, boxes_dict),
                                                           column_names=["data"],
                                                           num_parallel_workers=args.num_workers,
                                                           shuffle=False).batch(batch_size=args.train_batch_size)
    return dataloaders


# def flops_params(model, model_name: str, input_size: tuple):
#     input = ops.randn(*input_size)
#     flops, params = profile(model, inputs=(input,))
#     print(f'[{model_name}]flops: ', flops, 'params: ', params)


def main(dataloaders, i3d, evaluator, base_logger, args):
    # Print configuration
    if is_main_process():
        print('=' * 40)
        for k, v in vars(args).items():
            print(f'{k}: {v}')
        print('=' * 40)
    if args.use_bp:
        args.qk_dim = 768
    else:
        args.qk_dim = 1024

    # Set loss function
    criterion = nn.KLDivLoss()

    # Set models and optimizer(depend on whether to use goat)
    gcn = GCNnet_artisticswimming_simplified(args)
    attn_encoder = Encoder_Blocks(args.qk_dim, 1024, args.linear_dim, args.num_heads, args.num_layers,
                                      args.attn_drop)
    linear_bp = Linear_For_Backbone(args)

    param_dict = ms.load_checkpoint(args.ckpts + '/gcn.ckpt')
    param_not_load, _ = ms.load_param_into_net(gcn, param_dict)

    param_dict = ms.load_checkpoint(args.ckpts + '/attn_encoder.ckpt')
    param_not_load, _ = ms.load_param_into_net(attn_encoder, param_dict)

    param_dict = ms.load_checkpoint(args.ckpts + '/linear_bp.ckpt')
    param_not_load, _ = ms.load_param_into_net(linear_bp, param_dict)

    param_dict = ms.load_checkpoint(args.ckpts + '/evaluator.ckpt')
    param_not_load, _ = ms.load_param_into_net(evaluator, param_dict)

    def forward_fn(clip_feats, data):
        if not args.use_i3d_bb:
            clip_feats = linear_bp(clip_feats)  # B,540,1024
        attn = None
        if args.use_goat:
            # Use formation features
            if args.use_formation:
                q = data['formation_features']  # B,540,1024
                k = q
                # clip_feats = attn_encoder(q, k, clip_feats).to(device)
                output = attn_encoder(q, k, clip_feats)
                clip_feats = output[0]
                attn = output[1]
            # Use bridge-prompt features
            elif args.use_bp:
                q = data['bp_features']  # B,540,768
                k = q
                # clip_feats = attn_encoder(q, k, clip_feats).to(device)
                output = attn_encoder(q, k, clip_feats)
                clip_feats = output[0]
                attn = output[1]
            # Use self-attention
            elif args.use_self:
                q = clip_feats
                k = q
                # clip_feats = attn_encoder(q, k, clip_feats).to(device)
                output = attn_encoder(q, k, clip_feats)
                clip_feats = output[0]
                attn = output[1]
            # Use group features
            else:
                if args.use_cnn_features:
                    boxes_features = data['cnn_features']
                    boxes_in = data['boxes']  # B,T,N,4
                    q = gcn(boxes_features, boxes_in)  # B,540,1024
                    k = q
                    # clip_feats = attn_encoder(q, k, clip_feats).to(device)
                    output = attn_encoder(q, k, clip_feats)
                    clip_feats = output[0]
                    attn = output[1]
                else:
                    images_in = data['video']  # B,T,C,H,W
                    boxes_in = data['boxes']  # B,T,N,4
                    q = gcn(images_in, boxes_in)  # B,540,1024
                    k = q
                    # clip_feats = attn_encoder(q, k, clip_feats).to(device)
                    output = attn_encoder(q, k, clip_feats)
                    clip_feats = output[0]
                    attn = output[1]
        #########  GOAT END  ##########
        # print(clip_feats.mean(1))
        probs = evaluator(clip_feats.mean(1))
        loss = compute_loss(args.type, criterion, probs, data)
        return loss, probs, attn

    # grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    # DP
    # evaluator = nn.DataParallel(evaluator)
    # if args.use_goat:
    #     gcn = nn.DataParallel(gcn)
    #     attn_encoder = nn.DataParallel(attn_encoder)

    # train or test
    epoch_best = 0
    rho_best = 0
    RL2_best = 100
    rho = -1
    RL2 = 100
    for epoch in range(1):
        if args.use_multi_gpu:
            dataloaders['train'].sampler.set_epoch(epoch)
        if is_main_process():
            log_and_print(base_logger,
                          f'Epoch: {epoch}  Current Best rho: {rho_best} at epoch {epoch_best}, Current Best RL2: {RL2_best * 100}')

        for split in ['test']:
            true_scores = []
            pred_scores = []

            if split == 'train':
                # i3d.train()
                if args.use_goat:
                    gcn.set_train()
                    attn_encoder.set_train()
                evaluator.set_train()
                linear_bp.set_train()

                # torch.set_grad_enabled(True)
            else:
                # i3d.eval()
                if args.use_goat:
                    gcn.set_train(False)
                    attn_encoder.set_train(False)
                evaluator.set_train(False)
                linear_bp.set_train(False)
                # torch.set_grad_enabled(False)
            # visual
            attn_list = []
            key_list = []
            loss_list = []
            if split == 'train' or (split == 'test' and is_main_process()):
                start = time.time()
                for data in dataloaders[split]:
                    data = data[0]
                    key = data['key']
                    if split == 'train' and args.train_batch_size == 1:
                        data = {k: v.unsqueeze(0) for k, v in data.items() if k != 'key'}
                    if split == 'test' and args.test_batch_size == 1:
                        data = {k: v.unsqueeze(0) for k, v in data.items() if k != 'key'}
                    # print(data)
                    true_scores.extend(data['final_score'].numpy())
                    clip_feats = data['feature'] # B,540,1024
                    if split == 'train':
                        (loss, probs, attn), grads = grad_fn(clip_feats, data)
                        optimizer(grads)
                    elif split == 'test':
                        loss, probs, attn = forward_fn(clip_feats, data)
                        if args.use_goat:
                            attn_list.append(attn)
                            key_list.append(key)
                    loss_list.append(loss.numpy())
                    preds = compute_score(args.type, probs, data)
                    pred_scores.extend(preds.numpy())
                infer_time = time.time() - start
                # print("pred_scores: ", pred_scores)
                # print("true_scores: ", true_scores)
                rho, p = stats.spearmanr(pred_scores, true_scores)
                pred_scores = np.array(pred_scores)
                true_scores = np.array(true_scores)
                loss_avg = np.array(loss_list).mean()
                RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / \
                      true_scores.shape[0]

                if is_main_process():
                    log_and_print(base_logger,
                                  f'epoch:{epoch}, {split} loss: {loss_avg}, correlation: {rho}, Rl2: {RL2 * 100}, Infer_Time: {infer_time:.6f}')

        if rho > rho_best and split == 'test' and is_main_process():
            if args.use_goat:
                attn_list_log = attn_list
                key_list_log = key_list
            rho_best = rho
            epoch_best = epoch
            if is_main_process():
                log_and_print(base_logger, '-----New best rho found!-----')
        if RL2 < RL2_best and split == 'test' and is_main_process():
            RL2_best = RL2
            if is_main_process():
                log_and_print(base_logger, '-----New best RL2 found!-----')


if __name__ == '__main__':

    args = get_parser()

    if not os.path.exists('exp'):
        os.mkdir('exp')

    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True

    ms.set_context(device_target='GPU', device_id=0)

    setup_env(args.launcher, distributed=args.distributed)

    init_seed(args)

    localtime = time.asctime(time.localtime(time.time()))
    base_logger = get_logger(f'exp/{args.type}_full_split{args.split}_{args.lr}_{localtime}.log', args.log_info)
    i3d, evaluator = get_models(args)
    dataloaders = get_dataloaders(args)

    main(dataloaders, i3d, evaluator, base_logger, args)
