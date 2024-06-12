import logging
import time

import mindspore as ms
from tools import train_net, test_net
from utils.parser import get_args
from utils.goat_utils import setup_env, init_seed
from mmengine.dist import is_main_process


def get_logger(filepath):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 30 + 'TSA+GOAT' + '-' * 30)
    return logger


def main():
    # print(torch.cuda.device_count())
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = True
    args = get_args()
    if args.use_bp:
        args.qk_dim = 768
    else:
        args.qk_dim = 1024
    args.benchmark = 'FineDiving'

    localtime = time.asctime(time.localtime(time.time()))
    base_logger = get_logger(f'logs/train_{args.lr}_{localtime}.log')

    if is_main_process():
        print(args)
        base_logger.info(args)

    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True

    ms.set_context(device_target='GPU', device_id=0)

    setup_env(args.launcher, distributed=args.distributed)
    init_seed(args)

    if args.test:
        test_net(args)
    else:
        train_net(args, base_logger)


if __name__ == '__main__':
    main()
