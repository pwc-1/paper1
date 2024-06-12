import mindspore.nn as nn
from mindspore.common.initializer import Normal

LAYER1_NODE = 13375

class ImgModuleNus(nn.Cell):
    def __init__(self, y_dim, bit):
        super(ImgModuleNus, self).__init__()
        self.module_name = "img_module_nus"

        # full-conv layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=LAYER1_NODE, kernel_size=(y_dim, 1), stride=(1, 1), pad_mode='valid')
        self.conv2 = nn.Conv2d(in_channels=LAYER1_NODE, out_channels=bit, kernel_size=1, stride=(1, 1), pad_mode='valid')

        self.conv1.weight = Normal(0.0, 0.01)
        self.conv1.bias = Normal(0.0, 0.01)
        self.conv2.weight = Normal(0.0, 0.01)
        self.conv2.bias = Normal(0.0, 0.01)

    def construct(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.Squeeze(axis=2)(x)
        return x
