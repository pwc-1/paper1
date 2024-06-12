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
from mindspore.dataset import GeneratorDataset
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
import pandas as pd
import cv2
class Nabirds:
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
            self.train_dataset = create_dataset_imagenet(args.data_url , training=True, args=args)
            self.val_dataset = create_dataset_imagenet(args.data_url, training=False, args=args)

class NabirdsDataset(object):
    base_folder = 'nabirds/images'
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(NabirdsDataset, self).__init__()
        dataset_path = os.path.join(root, 'nabirds')
        #self.loader = default_loader
        self.root=root
        self.train = train
        self.transform = transform
        if download:
            self._download()
        image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
                                  sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        # Since the raw labels are non-continuous, map them to new ones
        self.label_map = get_continuous_class_map(image_class_labels['target'])
        train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = image_paths.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        # Load in the train / test split
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        # Load in the class data
        self.class_names = load_class_names(dataset_path)
        self.class_hierarchy = load_hierarchy(dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = self.label_map[sample.target]
        img =cv2.imread(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

def get_continuous_class_map(class_labels):
    label_set = set(class_labels)
    return {k: i for i, k in enumerate(label_set)}

def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])

    return names

def load_hierarchy(dataset_path=''):
    parents = {}

    with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            parents[child_id] = parent_id

    return parents



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
        data_set = NabirdsDataset(dataset_dir,training)
        data_set= GeneratorDataset(data_set,column_names=["image", "label"])
    else:
        data_set = NabirdsDataset(dataset_dir,training)
        data_set= GeneratorDataset(data_set,column_names=["image", "label"],num_parallel_workers=get_num_parallel_workers(3), shuffle=shuffle,
                                     num_shards=device_num, shard_id=rank_id)


    image_size = args.image_size


    image_size = args.image_size

#[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    if training:
        mean =[0.5*255 ,0.5*255,0.5*255]
        std = [0.5*255 ,0.5*255,0.5*255]
        transform_img = [

            vision.RandomResizedCrop((image_size,image_size),interpolation=Inter.BICUBIC),
            vision.RandomHorizontalFlip(prob=0.5),
            #vision.ToTensor(),
            vision.Normalize(mean=mean, std=std, is_hwc=True),
            vision.HWC2CHW(),
            #vision.ToPIL()
        ]
    else:
        mean = [0.5*255 ,0.5*255,0.5*255]
        std = [0.5*255 ,0.5*255,0.5*255]
        # test transform complete
        if args.crop:
            transform_img = [
                #vision.Decode(),
                vision.Resize(int(256 / 224 * image_size), interpolation=Inter.BICUBIC),
                vision.CenterCrop(image_size),
                #vision.ToTensor(),
                vision.Normalize(mean=mean, std=std, is_hwc=True),
                vision.HWC2CHW()
            ]
        else:
            transform_img = [
                #vision.Decode(),
                vision.Resize(int(image_size), interpolation=Inter.BICUBIC),
                #vision.ToTensor(),
                vision.Normalize(mean=mean, std=std, is_hwc=True),
                vision.HWC2CHW()
            ]


    data_set = data_set.map(input_columns="image", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_img)
    one_hot = C.OneHot(num_classes=args.num_classes)
    data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                                operations=one_hot)
    #data=data_set.create_dict_iterator()
    #for img in data:
    #    print(img['label'])
    transform_label = C.TypeCast(mstype.float32)
    data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_label)

    # apply batch operations

    data_set = data_set.batch(args.batch_size, drop_remainder=True,
                              num_parallel_workers=args.num_parallel_workers)

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
