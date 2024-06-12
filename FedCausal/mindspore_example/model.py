import numpy as np
import mindspore
from mindspore import nn, ops
from mindspore_example.locally_connected import LocallyConnected

class NonLinearModel(nn.Cell):
    def __init__(self, dims, bias=True, device=None):
        super().__init__()
        d = dims[0]
        self.dims = dims
        self.device = device
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Dense(d, d*dims[1], bias_init=bias)
        self.fc1_neg = nn.Dense(d, d*dims[1], bias_init=bias)
        # nn.init.constant_(self.fc1_pos.weight, val=0.0)
        # nn.init.constant_(self.fc1_neg.weight, val=0.0)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims)-2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.CellList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def construct(self, x):
        x = self.fc1_pos(x) - self.fc1_neg(x) # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = ops.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = ops.squeeze(x, axis=2)  # [n, d]
        return x

    def h_func(self):
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = ops.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        # h = trace_expm(A) - d  # (Zheng et al. 2018)
        init_e = ops.eye(d)
        M = init_e + A / d  # (Yu et al. 2019)
        E = ops.float_power(M, d - 1)
        h = (E.t() * M).sum() - d
        return h

    def local_L2(self, global_model):
        reg = 0.
        reg += ops.sum((self.fc1_pos.weight - global_model.fc1_pos.weight) ** 2)
        reg += ops.sum((self.fc1_neg.weight - global_model.fc1_neg.weight) ** 2)
        return reg

    def global_L2(self, params_pos, params_neg):
        reg = 0.
        reg += ops.sum((self.fc1_pos.weight - params_pos['fc1_pos.weight']) ** 2)
        reg += ops.sum((self.fc1_neg.weight - params_neg['fc1_neg.weight']) ** 2)
        return reg

    def l2_reg(self):
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += ops.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += ops.sum(fc.weight ** 2)
        return reg

    def fc1_l2_reg(self):
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += ops.sum(fc1_weight ** 2)
        return reg

    def fc1_l1_reg(self):
        reg = ops.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = ops.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = ops.sqrt(A)  # [i, j]
        W = W.asnumpy()  # [i, j]
        return W


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * ops.sum((output - target) ** 2)
    return loss