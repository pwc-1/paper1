from typing import Union, Optional
import csv
import mindspore as ms
import mindspore.nn as nn
from mindspore.dataset import GeneratorDataset
import numpy as np
import random
from mmengine.runner import set_random_seed
from mmengine.device import get_device
from mmengine.dataset import DefaultSampler
from mmengine.utils.dl_utils import set_multi_processing
from mmengine.model import is_model_wrapper
from mmengine.dist import (is_main_process, get_rank, init_dist,
                           is_distributed, sync_random_seed)


def setup_env(
        launcher: str,
        distributed: bool,
        cudnn_benchmark: bool = False,
        backend: str = 'nccl') -> None:
    # if cudnn_benchmark:
    #     # Whether to use `cudnn.benchmark` to accelerate training.
    #     torch.backends.cudnn.benchmark = True
    set_multi_processing(distributed=distributed)

    if distributed and not is_distributed():
        init_dist(launcher, backend=backend)


def wrap_model(model: nn.Cell,
               distributed: bool):
    # Set `export CUDA_VISIBLE_DEVICES=-1` to enable CPU training.
    # model = model.to(get_device())
    #
    # if not distributed:
    #     return model
    #
    # model = DistributedDataParallel(
    #     module=model,
    #     device_ids=[int(os.environ['LOCAL_RANK'])],
    #     broadcast_buffers=False,
    #     find_unused_parameters=False)
    # return model
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)


def build_dataloader(
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        persistent_workers: bool = True,
        seed: Optional[int] = None
) -> GeneratorDataset:
    sampler = DefaultSampler(dataset, shuffle=shuffle, seed=seed)
    dataloader = GeneratorDataset(
        source=dataset,
        sampler=sampler,
        num_parallel_workers=num_workers)
    # persistent_workers: create_tuple_iterator.num_epoch > 1
    dataloader = dataloader.batch(batch_size)
    return dataloader


def set_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = sync_random_seed()
    else:
        set_random_seed(seed=seed)

    print(f"Set seed as: {seed}")

    # if get_rank() == 0:
    #     # Only master rank will print msg
    #     print(f"Set seed as: {seed}")

    return seed


def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args:
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    B = X.shape[0]

    rx = X.pow(2).sum(axis=2).reshape((B, -1, 1))
    ry = Y.pow(2).sum(axis=2).reshape((B, -1, 1))

    dist = rx - 2.0 * X.matmul(Y.swapaxes(1, 2)) + ry.swapaxes(1, 2)

    return ms.ops.sqrt(dist)


def log_best(rho_best, RL2_min, epoch_best, args):
    # log for best
    with open(args.result_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if args.use_goat:
            if args.use_formation:
                mode = 'Formation'
            elif args.use_bp:
                mode = 'BP'
            elif args.use_self:
                mode = 'SELF'
            else:
                mode = 'GOAT'
        else:
            mode = 'Ori'

        if args.use_i3d_bb:
            backbone = 'I3D'
        elif args.use_swin_bb:
            backbone = 'SWIN'
        else:
            backbone = 'BP_BB'

        log_list = [format(rho_best, '.4f'), epoch_best, args.use_goat, args.lr, args.max_epoch, args.warmup,
                    args.seed, args.train_backbone, args.num_selected_frames, args.num_heads,
                    args.num_layers, args.random_select_frames, args.bs_train, args.bs_test, args.linear_dim, RL2_min, mode, backbone]
        writer.writerow(log_list)
