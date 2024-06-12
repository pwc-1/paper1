import mindspore
import numpy as np
from mindspore import nn, ops, Parameter, Tensor
import math

class LocallyConnected(nn.Cell):
    def __init__(self, num_linear, input_features, output_features, bias=True):
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = Parameter(Tensor(np.zeros((num_linear,
                                                input_features,
                                                output_features)), dtype=mindspore.float32))
        if bias:
            self.bias = Parameter(Tensor(np.zeros((num_linear, output_features)), dtype=mindspore.float32))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = Tensor(math.sqrt(k))
        self.weight = ops.uniform(self.weight.shape, -bound, bound)
        if self.bias is not None:
            self.bias = ops.uniform(self.bias.shape, -bound, bound)

    def construct(self, input_x):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = ops.matmul(ops.unsqueeze(input_x, dim=2), ops.unsqueeze(self.weight, dim=0))
        out = ops.squeeze(out, axis=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        return 'num_linear={}, in_features={}, out_features={}, bias={}'.format(
            self.num_linear, self.in_features, self.out_features,
            self.bias is not None
        )
