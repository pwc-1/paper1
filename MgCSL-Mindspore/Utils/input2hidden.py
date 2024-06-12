import mindspore as ms
from mindspore import nn, Tensor, Parameter
import mindspore.common.initializer as init
import math


class Input2Hidden(nn.Cell):

    def __init__(self, num_linear, input_features, output_features, bias=True):
        super(Input2Hidden, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        k = 1.0 / self.input_features
        bound = math.sqrt(k)

        self.weight = Parameter(Tensor(shape=(num_linear,
                                                input_features,
                                                output_features), dtype=ms.float32, init=init.Uniform(bound)))
        if bias:
            self.bias = Parameter(Tensor(shape=(num_linear, output_features), dtype=ms.float32, init=init.Uniform(bound)))
        else:
            self.register_parameter('bias', None)


    def construct(self, input_x: Tensor):
        # [d, n, m1] = [d, n, 1] @ [d, 1, m1]
        out = Tensor.matmul(input_x, self.weight)

        if self.bias is not None:
            # [d, n, q] += [d, 1, q]
            out += self.bias.unsqueeze(dim=1)
        return out
