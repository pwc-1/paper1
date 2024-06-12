import mindspore as ms
import mindspore.nn as nn
from opts import *


class MLP_block(nn.Cell):

    def __init__(self, output_dim):
        super(MLP_block, self).__init__()
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(axis=-1)
        self.layer1 = nn.Dense(feature_dim, 256)
        self.layer2 = nn.Dense(256, 128)
        self.layer3 = nn.Dense(128, output_dim)

        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeNormal(), cell.weight.shape, cell.weight.dtype))
                if cell.has_bias:
                    cell.bias.set_data(ms.common.initializer.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def construct(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        output = self.softmax(self.layer3(x))
        return output


class Evaluator(nn.Cell):

    def __init__(self, output_dim, model_type='USDL', num_judges=None):
        super(Evaluator, self).__init__()

        self.model_type = model_type

        if model_type == 'USDL':
            self.evaluator = MLP_block(output_dim=output_dim)
        else:
            assert num_judges is not None, 'num_judges is required in MUSDL'
            self.evaluator = nn.CellList([MLP_block(output_dim=output_dim) for _ in range(num_judges)])

    def construct(self, feats_avg):  # data: NCTHW

        if self.model_type == 'USDL':
            probs = self.evaluator(feats_avg)  # Nxoutput_dim
        else:
            probs = [evaluator(feats_avg) for evaluator in self.evaluator]  # len=num_judges
        return probs

