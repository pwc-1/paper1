# Copyright 2021-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Data operations, will be used in train.py and eval.py
"""
import os
import multiprocessing
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision
from mindspore.dataset.vision.utils import Inter
#import mindvision.classification.dataset as datat
from src.data.augment.auto_augment import _pil_interp, rand_augment_transform
from src.data.augment.mixup import Mixup
from src.data.augment.random_erasing import RandomErasing
from .data_utils.moxing_adapter import sync_data
import pickle
import numpy as np

class Cifar100:
    """ImageNet Define"""

    def __init__(self, args, training=True):
        if args.run_modelarts:
            print('Download data.')
            local_data_path = '/cache/data'
            sync_data(args.data_url, local_data_path, threads=128)
            print('Create train and evaluate dataset.')
            train_dir = os.path.join(local_data_path)
            val_ir = os.path.join(local_data_path)
            self.train_dataset = create_dataset_imagenet(train_dir, training=True, args=args)
            self.val_dataset = create_dataset_imagenet(val_ir, training=False, args=args)
        else:
            self.train_dataset = create_dataset_imagenet(args.data_url , usage='train', training=True, args=args)
            self.val_dataset = create_dataset_imagenet(args.data_url, training=False, args=args, usage='test')




def create_dataset_imagenet(dataset_dir, args, repeat_num=1, training=True,usage='Train'):
    """
    create a train or eval imagenet2012 dataset for SwinTransformer

    Args:
        dataset_dir(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1

    Returns:
        dataset
    """

    device_num, rank_id = _get_rank_info()
    shuffle = bool(training)
    if device_num == 1 :
        data_set = ds.Cifar100Dataset(dataset_dir, num_parallel_workers=get_num_parallel_workers(12), shuffle=shuffle,usage=usage)
    else:
        data_set = ds.Cifar100Dataset(dataset_dir, num_parallel_workers=get_num_parallel_workers(12), shuffle=shuffle,
                                     num_shards=device_num, shard_id=rank_id,usage=usage)
    #data = data_set.create_dict_iterator()
    #for image in data_set:
            #print(image )
    print(data_set.get_dataset_size())

    input=["image","fine_label"]
    data_set=data_set.project(columns=input)
    output=["image","label"]
    data_set=data_set.rename(input_columns=input,output_columns=output)


    image_size = args.image_size
 
    if training:
        mean = [0.507, 0.487, 0.441]
        std = [0.267, 0.256, 0.276]
        aa_params = dict(
            translate_const=int(image_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        interpolation = args.interpolation
        auto_augment = args.auto_augment
        assert auto_augment.startswith('rand')
        aa_params['interpolation'] = _pil_interp(interpolation)

        transform_img = [
            #vision.ToTensor(), 
            #vision.HWC2CHW(),
            #vision.RandomCrop((32, 32), (4, 4, 4, 4)), 
            vision.Resize((image_size,image_size),interpolation=Inter.BICUBIC),
            vision.RandomHorizontalFlip(prob=0.5),
            #vision.HWC2CHW(),
            vision.ToPIL()
        ]
    
        transform_img += [rand_augment_transform(auto_augment, aa_params)]
        transform_img += [
            vision.ToTensor(),
            vision.Normalize(mean=mean, std=std, is_hwc=False),
            RandomErasing(args.re_prob, mode=args.re_mode, max_count=args.re_count),
            #vision.HWC2CHW()
        ]
    else:
        mean = [0.507 * 255, 0.487 * 255, 0.441 * 255]
        std = [0.267 * 255, 0.256 * 255, 0.276 * 255]
        # test transform complete
        if args.crop:
            transform_img = [
                #vision.Decode(),
                vision.Resize(int(256 / 224 * image_size), interpolation=Inter.BICUBIC),
                vision.CenterCrop(image_size),
                vision.Normalize(mean=mean, std=std, is_hwc=True),
                vision.HWC2CHW()
            ]
        else:
            transform_img = [
                #vision.Decode(),
                vision.Resize(int(image_size), interpolation=Inter.BICUBIC),
                vision.Normalize(mean=mean, std=std, is_hwc=True),
                vision.HWC2CHW()
            ]

    transform_label = C.TypeCast(mstype.int32)
    data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_label)
    data_set = data_set.map(input_columns="image", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_img)
    
   
    if (args.mix_up > 0. or args.cutmix > 0.)  and not training:
        # if use mixup and not training(False), one hot val data label
        one_hot = C.OneHot(num_classes=args.num_classes)
        data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                                operations=one_hot)


    # apply batch operations

    data_set = data_set.batch(args.batch_size, drop_remainder=True,
                              num_parallel_workers=args.num_parallel_workers)

    if (args.mix_up > 0. or args.cutmix > 0.) and training:
        mixup_fn = Mixup(
            mixup_alpha=args.mix_up, cutmix_alpha=args.cutmix, cutmix_minmax=None,
            prob=args.mixup_prob, switch_prob=args.switch_prob, mode=args.mixup_mode,
            label_smoothing=args.label_smoothing, num_classes=args.num_classes)

        data_set = data_set.map(operations=mixup_fn, input_columns=["image", "label"],
                                num_parallel_workers=args.num_parallel_workers)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)
    ds.config.set_prefetch_size(4)
    return data_set


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id


def get_num_parallel_workers(num_parallel_workers):
    """
    Get num_parallel_workers used in dataset operations.
    If num_parallel_workers > the real CPU cores number, set num_parallel_workers = the real CPU cores number.
    """
    cores = multiprocessing.cpu_count()
    if isinstance(num_parallel_workers, int):
        if cores < num_parallel_workers:
            print("The num_parallel_workers {} is set too large, now set it {}".format(num_parallel_workers, cores))
            num_parallel_workers = cores
    else:
        print("The num_parallel_workers {} is invalid, now set it {}".format(num_parallel_workers, min(cores, 8)))
        num_parallel_workers = min(cores, 8)
    return num_parallel_workers
