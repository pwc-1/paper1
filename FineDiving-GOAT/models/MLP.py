import mindspore.nn as nn


class MLP_score(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(MLP_score, self).__init__()
        self.activation_1 = nn.ReLU()
        self.layer1 = nn.Dense(in_channel, 256)
        self.layer2 = nn.Dense(256, 64)
        self.layer3 = nn.Dense(64, out_channel)

    def construct(self, x):
        x = self.activation_1(self.layer1(x))
        x = self.activation_1(self.layer2(x))
        output = self.layer3(x)
        return output



