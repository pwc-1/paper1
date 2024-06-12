import os
import time
import argparse
import socket


from models import build_resnetv1_backbone

import mindspore
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision


parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10')
parser.add_argument('--model', '-a', default='resnet18')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')

parser.add_argument('--gpu_id', type=str, default='0',
                    help='which gpu is used for training')
parser.add_argument('--ckt_path', type=str,
default='./checkpoints/student/S_resnet18_T_resnet34_cifar10_ce_1.0_kl_1.0_kd_1.0_relation__best.pt.ckpt',
                    help='resume')


args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
mindspore.set_context(device_target="GPU",
                      device_id=int(args.gpu_id))


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    data_folder = './data/cifar-10-batches-bin'

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    return data_folder


# Image Preprocessing
normalize = vision.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]], is_hwc=False)
train_transform = transforms.Compose([
    vision.RandomCrop(32, padding=4),
    vision.RandomHorizontalFlip(),
    vision.ToTensor(),
    normalize
])
test_transform = transforms.Compose([vision.ToTensor(), normalize])

# dataset
dataset_path = get_data_folder()
if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = ds.Cifar10Dataset(dataset_dir=dataset_path,
                                      usage='train',
                                      shuffle=True,
                                      num_parallel_workers=args.num_workers)
    train_dataset = train_dataset.map(operations=train_transform, input_columns="image")
    test_dataset = ds.Cifar10Dataset(dataset_dir=dataset_path,
                                     usage='test',
                                     shuffle=False,
                                     num_parallel_workers=args.num_workers)
    test_dataset = test_dataset.map(operations=test_transform, input_columns="image")
elif args.dataset == 'cifar100':
    num_classes = 100
    train_dataset = ds.Cifar100Dataset(dataset_dir=dataset_path,
                                       usage='train',
                                       shuffle=True,
                                       num_parallel_workers=args.num_workers)
    train_dataset = train_dataset.map(operations=train_transform, input_columns="image")
    test_dataset = ds.Cifar100Dataset(dataset_dir=dataset_path,
                                      usage='test',
                                      shuffle=False,
                                      num_parallel_workers=args.num_workers)
    test_dataset = test_dataset.map(operations=test_transform, input_columns="image")
else:
    assert False
train_dataset = train_dataset.batch(batch_size=args.batch_size, num_parallel_workers=args.num_workers)
train_loader = train_dataset.create_tuple_iterator()
test_dataset = test_dataset.batch(batch_size=args.batch_size, num_parallel_workers=args.num_workers)
test_loader = test_dataset.create_tuple_iterator()

# models
if args.model in ['resnet18', 'resnet34', 'resnet50']:
    model = build_resnetv1_backbone(depth=int(args.model[6:]), num_classes=num_classes)
    model.set_train(False)
elif args.model == '':
    model = None
else:
    raise NameError('The specified model is not support')

# load model weight
weight = mindspore.load_checkpoint(args.ckt_path)
param_not_load, _ = mindspore.load_param_into_net(model, weight)


def validate(val_loader, model):
    model.set_train(False)
    correct = 0.
    total = 0.
    for images, labels in val_loader:
        labels = labels.astype(mindspore.int32)
        _, logits = model(images, is_feat=True)
        logits = ops.max(logits, 1)[1]
        total += labels.shape[0]
        correct += (logits == labels).sum().item()

    val_acc = correct / total
    return val_acc


if __name__ == '__main__':
    val_acc = validate(test_loader, model)
    print(val_acc)
