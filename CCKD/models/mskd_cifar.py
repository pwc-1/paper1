import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
import mindspore as ms
import math

from models.resnet_cifar import build_resnetv1_backbone


class ANN(nn.Cell):
    def __init__(self, in_channels, key_channels):
        super(ANN, self).__init__()
        self.conv_key = nn.SequentialCell([
            nn.Conv2d(in_channels, key_channels, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU()
        ])
        self.conv_query = nn.SequentialCell([
            nn.Conv2d(in_channels, key_channels, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU()
        ])
        self.transpose = ops.Transpose()
        self.matmul = nn.MatMul()
        self.softmax = nn.Softmax()

    def construct(self, x_l, x_h):
        bsz, c, h, w = x_h.shape

        value = self.transpose(x_l.view(bsz, c, -1), (0, 2, 1))  # bsz, hw, c
        key = self.conv_key(x_l).view(bsz, -1, h * w)

        query = self.conv_query(x_h).view(bsz, -1, h * w)
        query = self.transpose(query, (0, 2, 1))
        sim_map = self.matmul(query, key)
        sim_map = (c ** -.5) * sim_map
        sim_map = self.softmax(sim_map)

        context = self.matmul(sim_map, value)
        context = self.transpose(context, (0, 2, 1))
        context = context.view(bsz, c, *x_h.shape[2:])

        return context


class ABF(nn.Cell):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.SequentialCell([
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(mid_channel),
        ])

        self.conv2 = nn.SequentialCell([
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, has_bias=False, pad_mode="pad"),
            nn.BatchNorm2d(out_channel),
        ])

        if fuse:
            self.att_conv = ANN(in_channels=mid_channel, key_channels=mid_channel // 2)

        else:
            self.att_conv = None

        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))
            elif isinstance(cell, (nn.BatchNorm2d, nn.GroupNorm)):
                cell.gamma.set_data(ms.common.initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(ms.common.initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, (nn.Dense)):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeUniform(negative_slope=math.sqrt(5)),
                    cell.weight.shape, cell.weight.dtype))
                cell.bias.set_data(ms.common.initializer.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def construct(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        x = self.conv1(x)  # transform student features

        if self.att_conv is not None:
            y = ops.interpolate(y, (shape, shape), mode='nearest')
            x = self.att_conv(y, x)

        if x.shape[-1] != out_shape:
            x = ops.interpolate(x, (out_shape, out_shape), mode='nearest')
        y = self.conv2(x)
        return y, x


class ReviewKD(nn.Cell):
    def __init__(self, student, in_channels, out_channels, shapes, out_shapes):
        super(ReviewKD, self).__init__()
        self.student = student
        self.shapes = shapes
        self.out_shapes = shapes if out_shapes is None else out_shapes

        abfs = nn.CellList()

        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels) - 1))   # last one False
        self.abfs = abfs[::-1]

    def construct(self, x, is_feat=True, preact=False):
        student_features, student_logits = self.student(x, is_feat=is_feat)
        x = student_features[::-1]

        results = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
        results.append(out_features)
        for features, abf, shape, out_shape in zip(x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)

        return results, student_logits


def build_mskd_backbone(args, num_classes):
    out_shapes = None
    if args.model in ['resnet18', 'resnet34']:
        student = build_resnetv1_backbone(depth=int(args.model[6:]), num_classes=num_classes)
        in_channels = [64, 128, 256, 512]
        out_channels = [64, 128, 256, 512]
        shapes = [1, 2, 4, 8]
    elif args.model in ['resnet50']:
        student = build_resnetv1_backbone(depth=int(args.model[6:]), num_classes=num_classes)
        in_channels = [256, 512, 1024, 2048]
        out_channels = [256, 512, 1024, 2048]
        shapes = [1, 2, 4, 8]
    else:
        raise NameError("The specified ReviewKD backbone is not support")

    backbone = ReviewKD(
        student=student,
        in_channels=in_channels,
        out_channels=out_channels,
        shapes=shapes,
        out_shapes=out_shapes
    )
    return backbone


if __name__ == '__main__':
    import argparse
    import mindspore
    from mindspore import Tensor
    import numpy as np

    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--teacher', type=str, default='resnet50')
    parser.add_argument('--num_classes', type=int, default=7)

    args = parser.parse_args()

    net = build_mskd_backbone(args, num_classes=args.num_classes)

    inputs = Tensor(np.random.randn(2, 3, 32, 32).astype(np.float32))
    feas, logit = net(inputs, is_feat=True)
    for fea in feas:
        print(fea.shape)
    print(logit.shape)

