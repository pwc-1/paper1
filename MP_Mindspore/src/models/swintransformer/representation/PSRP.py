from mindspore.nn import Dense,ReLU
import mindspore.nn as nn
class PSRP(nn.Cell):
    def __init__(self, channel=768, reduction=16):
        super(PSRP, self).__init__()
        
        self.down = Dense(channel, reduction, has_bias=False)
        self.relu = ReLU()
        self.up_weight = Dense(reduction, channel, has_bias=False)
        self.up_bias = Dense(reduction, channel, has_bias=False)

    def construct(self, x):

        weight = self.up_weight(self.relu(self.down(x)))
        
        bias = self.up_bias(self.relu(self.down(x)))

        return weight, bias