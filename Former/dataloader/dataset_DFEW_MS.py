import os.path
from numpy.random import randint
import glob
import os
from dataloader.Video_transform_MS import *
import numpy as np

from PIL import Image
from mindspore.dataset import GeneratorDataset


import mindspore.dataset as ds
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype
import mindspore.dataset.vision.py_transforms as transforms
import mindspore.dataset as ds

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoDataset(GeneratorDataset):
    def __init__(self, root_path, list_file, num_segments, duration, mode, transform, image_size, train_flag = 0):
        super().__init__(source=self._generate_data(), column_names=['image', 'label'])
        self.root_path = root_path
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.train_flag = train_flag
        self._parse_list()

    def _generate_data(self):
        for index in range(len(self)):
            yield self.__getitem__(index)

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1]) >= 16]
        self.video_list = [VideoRecord(item) for item in tmp]
        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record):
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + np.random.randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(np.random.randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        video_name = record.path.split('/')[-1]
        tem_path = '.' + os.path.join(record.path, '*.jpg')[26:]
        video_frames_path = glob.glob(tem_path)
        video_frames_path.sort()

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):

                seg_imgs = [Image.open(os.path.join(video_frames_path[p])).convert('RGB')]
                images.extend(seg_imgs)
                if p < record.num_frames - 1:
                    p += 1
        if self.train_flag == 1:
            GroupRandomSizedCrop1 = GroupRandomSizedCrop(112)
            images = GroupRandomSizedCrop1(images)
            GroupRandomHorizontalFlip1 = GroupRandomHorizontalFlip()
            images = GroupRandomHorizontalFlip1(images)
            Stack1 = Stack()
            images = Stack1(images)
            ToMindSporeFormatTensor1 = ToMindSporeFormatTensor()
            images = ToMindSporeFormatTensor1(images)
        else:
            GroupResize1 = GroupResize(112)
            images = GroupResize1(images)
            Stack2 = Stack()
            images = Stack2(images)
            ToMindSporeFormatTensor2 = ToMindSporeFormatTensor()
            images = ToMindSporeFormatTensor2(images)
        

        # images = self.transform(images)
        images = np.reshape(images, (-1, 3, self.image_size, self.image_size))

        return {'image': images.astype(np.float32), 'label': record.label}
    
    def __len__(self):
        return len(self.video_list)




#without GroupRandomSizedCrop
def train_data_loader(data_set):
    image_size = 112

    # Define transformations
    transforms_list = [
        GroupRandomSizedCrop(image_size),
        GroupRandomHorizontalFlip(),
        Stack(),
        ToMindSporeFormatTensor()
    ]


    train_transforms = ds.transforms.Compose(transforms_list)


    # Create MindSpore dataset
    train_data = VideoDataset(root_path="/home/datasets/DFER_Face/",
                              list_file="./annotation/ma_set_"+str(data_set)+"_train.txt",
                              num_segments=8,
                              duration=2,
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size,
                              train_flag = 1)

    return train_data

def test_data_loader(data_set):

    image_size = 112

    # Define transformations
    transforms_list = [
        GroupResize(image_size),
        Stack(),
        ToMindSporeFormatTensor()
    ]



    test_transforms = ds.transforms.Compose(transforms_list)

    # Create MindSpore dataset
    test_data = VideoDataset(root_path="/home/datasets/DFER_Face/",
                              list_file="./annotation/ma_set_"+str(data_set)+"_test.txt",
                              num_segments=8,
                              duration=2,
                              mode='test',
                              transform=test_transforms,
                              image_size=image_size,
                              train_flag = 0)

    return test_data

