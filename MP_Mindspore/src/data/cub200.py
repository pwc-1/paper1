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
import pandas as pd
import os
import multiprocessing
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore.dataset import MappableDataset, VisionBaseDataset,GeneratorDataset
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision
from mindspore.dataset.vision.utils import Inter
#import mindvision.classification.dataset as datat
from src.data.augment.auto_augment import _pil_interp, rand_augment_transform
from src.data.augment.mixup import Mixup
from src.data.augment.random_erasing import RandomErasing
from .data_utils.moxing_adapter import sync_data
import pickle
import cv2 
import numpy as np
class CUB200Dataset(object):
    #base_folder = 'CUB_200_2011/images'
    base_folder = 'CUB_200_2011/images'
    file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(CUB200Dataset, self).__init__()
        self.root = root
        self.train = train
        self.transform=transform
        self.target_transform=target_transform
        self.imgs=[]
        self.targets=[]
        #if download:
        #    self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted. You can  download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                  sep=' ', names=['class_name'], usecols=[1])
        self.class_names = class_names['class_name'].to_list()
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]
        for idx in range(len(self.data)):            
            sample = self.data.iloc[idx]
            path = os.path.join(self.root, self.base_folder, sample.filepath)
            target = sample.target - 1  # Targets start at 1 by default, so shift to 0
            img = cv2.imread(path)
            self.imgs.append(img)
            self.targets.append(target)



    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True



    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        img, target = self.imgs[index], self.targets[index]
#        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)              
        return img, target
    

class CUB200:
    """ImageNet Define"""

    def __init__(self, args, training=True):
        if args.run_modelarts:
            print('Download data.')
            local_data_path = '/cache/data'
            sync_data(args.data_url, local_data_path, threads=128)
            print('Create train and evaluate dataset.')
            train_dir = os.path.join(local_data_path)
            val_ir = os.path.join(local_data_path)
            self.train_dataset = create_dataset_imagenet(train_dir, training=True, args=args,usage='train')
            self.val_dataset = create_dataset_imagenet(val_ir, training=False, args=args,usage='test')
        else:

            self.train_dataset = create_dataset_imagenet(args.data_url,args , training=True)
            self.val_dataset = create_dataset_imagenet(args.data_url,args, training=False)



def create_dataset_imagenet(dataset_dir, args, repeat_num=1, training=True,usage='train'):
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
        data_set = CUB200Dataset(dataset_dir,training)

        data_set= GeneratorDataset(data_set,column_names=["image", "label"])
    else:
        data_set = CUB200Dataset(dataset_dir,training)
        data_set= GeneratorDataset(data_set,column_names=["image", "label"],num_parallel_workers=get_num_parallel_workers(1), shuffle=shuffle,
                                     num_shards=device_num, shard_id=rank_id)

    print(data_set.get_dataset_size())

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
