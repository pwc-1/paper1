import os
import shutil
import socket

import mindspore
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Border


def get_data_folder(args):
    """
    return server-dependent path to store the data
    """
    if args.dataset == 'cifar10':
        data_folder = './data/cifar-10-batches-bin'
    elif args.dataset == 'cifar100':
        data_folder = './data/cifar-100-binary'

    return data_folder


def get_cifar100_dataloaders(args):
    dataset_path = get_data_folder(args)

    normalize = vision.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]], is_hwc=False)  # for others
    # normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # for ResNet50 and VGG13

    transform_train = transforms.Compose([
        vision.Pad(4, padding_mode=Border.REFLECT),
        vision.RandomHorizontalFlip(),
        vision.RandomCrop(32),
        vision.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        vision.ToTensor(),
        normalize
    ])
    train_set = ds.Cifar100Dataset(dataset_dir=dataset_path,
                                   usage='train',
                                   shuffle=True,
                                   num_parallel_workers=args.num_workers)
    train_set = train_set.map(operations=transform_train, input_columns="image")
    train_set = train_set.batch(batch_size=args.batch_size, num_parallel_workers=args.num_workers)
    train_loader = train_set.create_tuple_iterator()

    test_set = ds.Cifar100Dataset(dataset_dir=dataset_path,
                                  usage='test',
                                  shuffle=False,
                                  num_parallel_workers=args.num_workers)
    test_set = test_set.map(operations=transform_test, input_columns="image")
    test_set = test_set.batch(batch_size=args.batch_size, num_parallel_workers=args.num_workers)
    test_loader = test_set.create_tuple_iterator()

    return train_loader, test_loader


def get_cifar10_dataloaders(args):
    dataset_path = get_data_folder(args)

    normalize = vision.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]], is_hwc=False)

    transform_train = transforms.Compose([
        vision.Pad(4, padding_mode=Border.REFLECT),
        vision.RandomHorizontalFlip(),
        vision.RandomCrop(32),
        vision.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        vision.ToTensor(),
        normalize
    ])
    train_set = ds.Cifar10Dataset(dataset_dir=dataset_path,
                                  usage='train',
                                  shuffle=True,
                                  num_parallel_workers=args.num_workers)
    train_set = train_set.map(operations=transform_train, input_columns="image")
    train_set = train_set.batch(batch_size=args.batch_size, num_parallel_workers=args.num_workers)
    train_loader = train_set.create_tuple_iterator()

    test_set = ds.Cifar10Dataset(dataset_dir=dataset_path,
                                 usage='test',
                                 shuffle=False,
                                 num_parallel_workers=args.num_workers)
    test_set = test_set.map(operations=transform_test, input_columns="image")
    test_set = test_set.batch(batch_size=args.batch_size, num_parallel_workers=args.num_workers)
    test_loader = test_set.create_tuple_iterator()

    return train_loader, test_loader


def get_tinyImagenet_dataloaders(args):
    normalize = vision.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821])

    transform_train = transforms.Compose([
        vision.RandomResizedCrop(32),
        vision.RandomHorizontalFlip(),
        vision.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        vision.Resize(32),
        vision.ToTensor(),
        normalize
    ])
    trainset = ds.ImageFolderDataset(dataset_dir='',
                                     shuffle=True,
                                     num_parallel_workers=args.num_workers)
    trainset = trainset.map(operations=transform_train, input_columns="image")
    trainset = trainset.batch(batch_size=args.batch_size, num_parallel_workers=args.num_workers)
    train_loader = trainset.create_tuple_iterator()

    testset = ds.ImageFolderDataset(dataset_dir='',
                                    shuffle=False,
                                    num_parallel_workers=args.num_workers)
    testset = testset.map(operations=transform_test,input_columns="image")
    testset = testset.batch(batch_size=args.batch_size, num_parallel_workers=args.num_workers)
    test_loader = testset.create_tuple_iterator()

    return train_loader, test_loader
