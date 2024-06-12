# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mindspore as ms
from mindspore import nn, Tensor, Parameter
import mindspore.common.initializer as init
import math


class LocallyConnected(nn.Cell):
    """
    Local linear layer, i.e. Conv1dLocal() with filter size 1.

    Parameters
    ----------
    num_linear: num of local linear layers, i.e.
    input_features: m1
        Shape: [n, d, m1]
    output_features: m2
        Shape: [n, d, m2]
    bias: whether to include bias or not

    Attributes
    ----------
    weight: [d, m1, m2]
    bias: [d, m2]
    """

    def __init__(self, num_linear, input_features, output_features, bias=True):
        super(LocallyConnected, self).__init__()
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
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

    def construct(self, input_x: Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = Tensor.matmul(input_x.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(axis=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        """
        (Optional)Set the extra information about this module. You can test
        it by printing an object of this class.

        Returns
        -------

        """

        return 'num_linear={}, in_features={}, out_features={}, bias={}'.format(
            self.num_linear, self.in_features, self.out_features,
            self.bias is not None
        )
