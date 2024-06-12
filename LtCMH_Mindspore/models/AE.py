import mindspore.nn as nn
from mindspore.common.initializer import Normal

class AutoEncoder(nn.Cell):
    def __init__(self, bit):
        super(AutoEncoder, self).__init__()
        self.bit = bit

        self.imgencoder = nn.SequentialCell([
            nn.Dense(bit),
            nn.Tanh(),
            nn.Dense(bit),
            nn.Tanh(),
            nn.Dense(bit)
        ])

        self.imgdecoder = nn.SequentialCell([
            nn.Dense(2 * bit),
            nn.Tanh(),
            nn.Dense(bit),
            nn.Tanh(),
            nn.Dense(bit),
            nn.Tanh(),
            nn.Dense(bit),
            nn.Tanh()
        ])

        self.txtencoder = nn.SequentialCell([
            nn.Dense(bit),
            nn.Tanh(),
            nn.Dense(bit),
            nn.Tanh(),
            nn.Dense(bit)
        ])

        self.txtdecoder = nn.SequentialCell([
            nn.Dense(2 * bit),
            nn.Tanh(),
            nn.Dense(bit),
            nn.Tanh(),
            nn.Dense(bit),
            nn.Tanh(),
            nn.Dense(bit),
            nn.Tanh()
        ])

        self.commonencoder = nn.SequentialCell([
            nn.Dense(2 * bit),
            nn.Tanh(),
            nn.Dense(bit),
            nn.Tanh(),
            nn.Dense(bit),
            nn.Tanh(),
            nn.Dense(bit)
        ])

    def construct(self, inputsx, inputsy):
        imgindi = self.imgencoder(inputsx)
        txtindi = self.txtencoder(inputsy)
        common = self.commonencoder(imgindi.concat(txtindi, 1))
        imgcon = mnp.concatenate((imgindi, common), axis=1)
        txtcon = mnp.concatenate((txtindi, common), axis=1)
        imgout = self.imgdecoder(imgcon)
        txtout = self.txtdecoder(txtcon)
        return imgindi, txtindi, common, imgout, txtout
