from mindspore import nn, Tensor
import numpy as np
from scipy.linalg import schur

class Acyclic(nn.Cell):
    def __init__(self):
        super(Acyclic, self).__init__()

    def construct(self, input):
        D = input
        E, U = schur(D.asnumpy())

        h = 0.5 * np.sum(np.diag(E) ** 2)

        E = Tensor(E, input.dtype)
        U = Tensor(U, input.dtype)

        self.D, self.E, self.U = D, E, U
        return h

    def bprop(self, input, output, grad_output):
        D, E, U = self.D, self.E, self.U
        G_h = U @ (Tensor.diag(Tensor.diag(E.t())) @ U.t())
        grad_input = grad_output * G_h * Tensor.sqrt(D) * 2
        return grad_input

acyclic = Acyclic()
