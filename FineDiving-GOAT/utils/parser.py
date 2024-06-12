import os
import yaml
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archs', type=str, choices=['TSA'], help='our approach')
    parser.add_argument('--benchmark', type=str, choices=['FineDiving'], help='dataset')
    parser.add_argument('--prefix', type=str, default='default', help='experiment name')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer name')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training (interrupted by accident)')
    parser.add_argument('--sync_bn', type=bool, default=False)
    parser.add_argument('--fix_bn', type=bool, default=True)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--ckpts', type=str, default='ckpts/', help='test used ckpt path')
    parser.add_argument('--voter_number', type=int, help='voter_number', default=10)
    parser.add_argument('--print_freq', type=int, help='print_freq', default=40)
    parser.add_argument('--fix_size', type=int, help='fix_size', default=5)
    parser.add_argument('--step_num', type=int, help='step_num', default=3)
    parser.add_argument('--prob_tas_threshold', type=float, help='prob_tas_threshold', default=0.25)

    # basic
    parser.add_argument('--max_epoch', type=int, help='number of training epochs', default=50)
    parser.add_argument('--bs_train', type=int, help='batch size for training phase', default=2)
    parser.add_argument('--bs_test', type=int, help='batch size for test phase', default=1)
    parser.add_argument('--seed', type=int, help='manual seed', default=42)
    parser.add_argument('--workers', type=int, help='number of subprocesses for dataloader', default=1)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-5)
    parser.add_argument('--lr_factor', type=float, help='learning rate factor', default=0.01)
    parser.add_argument('--weight_decay', type=float, help='L2 weight decay', default=1e-4)
    parser.add_argument('--train_dropout_prob', type=float, default=0.3, help='train_dropout_prob')

    # goat setting below
    # cnn
    parser.add_argument('--length', type=int, help='length of videos', default=96)
    parser.add_argument('--img_size', type=tuple, help='input image size', default=(224, 224))
    parser.add_argument('--out_size', type=tuple, help='output image size', default=(25, 25))
    parser.add_argument('--crop_size', type=tuple, help='RoiAlign image size', default=(5, 5))

    # gcn
    parser.add_argument('--num_boxes', type=int, help='boxes number of each frames', default=8)
    parser.add_argument('--num_selected_frames', type=int, help='number of selected frames per 16 frames', default=1)

    # path
    parser.add_argument('--data_root', type=str, help='root of dataset', default='/mnt/f/University/2023Summer/LOGO/Dataset/Video_result')
    parser.add_argument('--label_path', type=str, help='path of annotation file', default='/mnt/f/University/2023Summer/LOGO/Dataset/anno_dict.pkl')
    parser.add_argument('--boxes_path', type=str, help='path of boxes annotation file', default='/mnt/f/University/2023Summer/LOGO/Dataset/ob_result_new.pkl')
    # backbone features path
    parser.add_argument('--i3d_feature_path', type=str, help='path of i3d feature dict', default='/mnt/f/University/2023Summer/LOGO/Dataset/video_feature_dict.pkl')
    parser.add_argument('--swin_feature_path', type=str, help='path of swin feature dict', default='/mnt/f/University/2023Summer/LOGO/Dataset/swin_features_dict_new.pkl')
    parser.add_argument('--bpbb_feature_path', type=str, help='path of bridge-prompt feature dict', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/bpbb_features_540.pkl')
    # attention features path
    parser.add_argument('--feamap_root', type=str, help='path of feature dict', default='/mnt/f/University/2023Summer/LOGO/Dataset/video_feamap_dict.pkl')
    parser.add_argument('--train_split', type=str, help='', default='/mnt/f/University/2023Summer/LOGO/Dataset/train_split3.pkl')
    parser.add_argument('--test_split', type=str, help='', default='/mnt/f/University/2023Summer/LOGO/Dataset/test_split3.pkl')
    parser.add_argument('--cnn_feature_path', type=str, help='path of cnn feature dict', default='/mnt/f/University/2023Summer/LOGO/Dataset/inception_feature_dict.pkl')
    parser.add_argument('--stage1_model_path', type=str, default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/Group-AQA-Distributed/ckpts/STAGE1_256frames_rho0.3257707338254451_(224, 224)_(25, 25)_loss82.48323059082031.pth', help='stage1_model_path')
    parser.add_argument('--bp_feature_path', type=str, default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/bp_features', help='bridge prompt feature path')
    parser.add_argument('--formation_feature_path', type=str, default='/mnt/f/University/2023Summer/LOGO/Dataset/formation_features_middle_1.pkl', help='formation feature path')

    # [BOOL]
    # bool for attention mode[GOAT / BP / FORMATION / SELF]
    parser.add_argument('--use_goat', type=int, help='whether to use group-aware-attention', default=1)
    parser.add_argument('--use_bp', type=int, help='whether to use bridge prompt features', default=0)
    parser.add_argument('--use_formation', type=int, help='whether to use formation features', default=0)
    parser.add_argument('--use_self', type=int, help='whether to use self attention', default=0)
    # bool for backbone[I3D / SWIN / BP]
    parser.add_argument('--use_i3d_bb', type=int, help='whether to use i3d as backbone', default=1)
    parser.add_argument('--use_swin_bb', type=int, help='whether to use swin as backbone', default=0)
    parser.add_argument('--use_bp_bb', type=int, help='whether to use bridge-prompt as backbone', default=0)
    # others
    parser.add_argument('--train_backbone', type=int, help='whether to train backbone', default=0)
    parser.add_argument('--use_gcn', type=int, help='whether to use gcn', default=1)
    parser.add_argument('--warmup', type=int, help='whether to warm up', default=0)
    parser.add_argument('--random_select_frames', type=int, help='whether to select frames randomly', default=0)
    parser.add_argument('--use_multi_gpu', type=int, help='whether to use multi gpus', default=0)
    parser.add_argument('--gcn_temporal_fuse', type=int, help='whether to fuse temporal node before gcn', default=0)
    parser.add_argument('--use_cnn_features', type=int, help='whether to use pretrained cnn features', default=1)

    # log
    parser.add_argument('--exp_name', type=str, default='goat', help='experiment name')
    parser.add_argument('--result_path', type=str, default='result/result.csv', help='result log path')

    # attention
    parser.add_argument('--num_heads', type=int, default=8, help='number of self-attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='number of encoder layers')
    parser.add_argument('--linear_dim', type=int, default=1024, help='dimension of query and key')
    parser.add_argument('--attn_drop', type=float, default=0., help='drop prob of attention layer')

    # fixed parameters
    parser.add_argument('--emb_features', type=int, default=1056, help='output feature map channel of backbone')
    parser.add_argument('--num_features_boxes', type=int, default=1024, help='dimension of features of each box')
    parser.add_argument('--num_features_relation', type=int, default=256, help='dimension of embedding phi(x) and theta(x) [Embedded Dot-Product]')
    parser.add_argument('--num_features_gcn', type=int, default=1024, help='dimension of features of each node')
    parser.add_argument('--num_graph', type=int, default=16, help='number of graphs')
    parser.add_argument('--gcn_layers', type=int, default=1, help='number of gcn layers')
    parser.add_argument('--pos_threshold', type=float, default=0.2, help='threshold for distance mask')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--random_choosing', type=int, help=' ', default=0)
    parser.add_argument('--action_number_choosing', type=int, help=' ', default=1)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    # if 'RANK' not in os.environ:
    #     os.environ['RANK'] = str(0)

    if args.test:
        if args.ckpts is None:
            raise RuntimeError('--ckpts should not be None when --test is activate')

    return args


def setup(args):
    args.config = '{}_TSA.yaml'.format(args.benchmark)
    args.experiment_path = os.path.join('./experiments', args.archs, args.benchmark, args.prefix)
    if args.resume:
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path):
            print("Failed to resume")
            args.resume = False
            setup(args)
            return

        print('Resume yaml from %s' % cfg_path)
        with open(cfg_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        merge_config(config, args)
        args.resume = True
    else:
        config = get_config(args)
        merge_config(config, args)
        create_experiment_dir(args)
        save_experiment_config(args)


def get_config(args):
    try:
        print('Load config yaml from %s' % args.config)
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
    except:
        raise NotImplementedError('%s arch is not supported' % args.archs)
    return config


def merge_config(config, args):
    for k, v in config.items():
        setattr(args, k, v)


def create_experiment_dir(args):
    try:
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    except:
        pass


def save_experiment_config(args):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(args.__dict__, f)
        print('Save the Config file at %s' % config_path)
