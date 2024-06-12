import logging
import time

from tools import run_net
from tools import test_net
from utils import parser
from utils.multi_gpu import setup_env
import mindspore as ms
from mmengine.dist import (is_main_process, get_rank, init_dist,
                           is_distributed, sync_random_seed)


def get_logger(filepath):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 30 + 'CoRe+GOAT' + '-' * 30)
    return logger


def main():
    # config
    # print(torch.cuda.device_count())
    # torch.backends.cudnn.enabled = False
    args = parser.get_args()
    if args.benchmark == 'MTL':
        if not args.usingDD:
            args.score_range = 100

    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
    if args.use_bp:
        args.qk_dim = 768
    else:
        args.qk_dim = 1024

    setup_env(args.launcher, distributed=args.distributed)
    localtime = time.asctime(time.localtime(time.time()))
    base_logger = get_logger(f'logs/train_{args.lr}_{localtime}.log')
    ms.set_context(device_target='GPU', device_id=0)

    if is_main_process():
        print(args)
        base_logger.info(args)

    # run
    if args.test:
        test_net(args)
    else:
        run_net(args, base_logger)


if __name__ == '__main__':
    main()
