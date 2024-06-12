import mindspore.nn as nn


class double_conv(nn.Cell):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.SequentialCell(
            nn.Conv1d(in_ch, out_ch, 3, padding=1, pad_mode='pad', has_bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, pad_mode='pad', has_bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def construct(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Cell):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def construct(self, x):
        x = self.conv(x)
        return x


class down(nn.Cell):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.SequentialCell(
            nn.MaxPool1d(2, stride=2, pad_mode='pad'),
            double_conv(in_ch, out_ch)
        )

    def construct(self, x):
        x = self.mpconv(x)
        return x


class MLP_tas(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(MLP_tas, self).__init__()

        self.activation_1 = nn.ReLU()
        self.layer1 = nn.Dense(in_channel, 128)
        self.layer2 = nn.Dense(128, 64)
        self.layer3 = nn.Dense(64, out_channel)
        self.activation_2 = nn.Sigmoid()

    def construct(self, x):
        x = self.activation_1(self.layer1(x))
        x = self.activation_1(self.layer2(x))
        output = self.activation_2(self.layer3(x))
        return output
