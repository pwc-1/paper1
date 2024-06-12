import copy
import random
import numpy as np
import mindspore
from mindspore import ops, value_and_grad
from tqdm import tqdm
from mindspore_example.model import NonLinearModel
from mindspore_example.non_linear_client import NonlinearClient
from mindspore_example.lbfgsb_scipy import LBFGSBScipy

class NonlinearServer:
    def __init__(self, args, datas):
        super().__init__()
        self.dims = args.dims
        self.ns = args.ns
        self.bias = args.bias
        self.model = NonLinearModel(self.dims, self.bias, args.device)
        self.K = args.K
        self.C = args.C
        self.max_iter = args.max_iter
        self.device = args.device
        self.rho_max = args.rho_max
        self.h_tol = args.h_tol
        self.rho = 1.0
        self.alpha = 0.0
        self.h = np.inf
        self.threshold = args.threshold
        self.models = []
        for i in range(self.K):
            temp = copy.deepcopy(self.model)
            temp.fc1_pos.weight.bounds = temp._bounds()
            temp.fc1_neg.weight.bounds = temp._bounds()
            clienti = NonlinearClient(args, datas[i].astype(np.float32))
            self.models.append([clienti, temp])

    def run(self, B_true):
        self.B_true = B_true
        for t in tqdm(range(self.max_iter)):
            h_mark = None
            while self.rho < self.rho_max:
                #sample clients
                m = np.max([int(self.C * self.K), 1])
                index = random.sample(range(0, self.K), m)
                #clone fc1
                self.dispatch(index)
                #local update
                self.client_update(index)
                #aggregation
                h_new, h_mark = self.aggregation(index)
                if h_new > 0.25 * self.h:
                    self.rho *= 10
                else:
                    break
            self.h = h_new
            self.rho *= 10
            self.alpha += self.rho * h_mark
            if self.h <= self.h_tol or self.rho >= self.rho_max:
                break

    def dispatch(self, index):
        for j in index:
            for old_params, new_params in zip(self.models[j][1].fc1_pos.trainable_params(), self.model.fc1_pos.trainable_params()):
                old_params.set_data(new_params)
            for old_params, new_params in zip(self.models[j][1].fc1_neg.trainable_params(), self.model.fc1_neg.trainable_params()):
                old_params.set_data(new_params)

    def client_update(self, index):
        for j in index:
            self.models[j][0].train(self.models[j][1], self.model, self.rho, self.alpha, self.h)

    def aggregation(self, index):
        s = {}
        s_total = 0
        for j in index:
            s[j] = self.ns[j]
            s_total += s[j]

        params_pos, params_neg, params_poss, params_negs = {}, {}, {}, {}
        for p in self.model.fc1_pos.get_parameters():
            params_pos[p.name] = ops.zeros_like(p.data)
        for p in self.model.fc1_neg.get_parameters():
            params_neg[p.name] = ops.zeros_like(p.data)
        for j in index:
            params_poss[j], params_negs[j] = {}, {}
            for p in self.models[j][1].fc1_pos.get_parameters():
                params_pos[p.name] += p.data * (s[j] / s_total)
                params_poss[j][p.name] = p.data
            for p in self.models[j][1].fc1_neg.get_parameters():
                params_neg[p.name] += p.data * (s[j] / s_total)
                params_negs[j][p.name] = p.data
        for p in self.model.fc1_pos.get_parameters():
            p.set_data(params_pos[p.name])
        for p in self.model.fc1_neg.get_parameters():
            p.set_data(params_neg[p.name])
        h_mark = self.model.h_func().item()

        h_new = None
        optimizer = LBFGSBScipy(self.model.trainable_params())

        def forward_fn():
            tol_L2 = 0
            for j in index:
                L2 = 0.5 * self.model.global_L2(params_poss[j], params_negs[j])
                tol_L2 += L2 * (s[j] / s_total)
            h = self.model.h_func()
            obj = tol_L2 + 0.5 * self.rho * h * h + self.alpha * h
            return obj
        grad_fn = value_and_grad(forward_fn, grad_position=None, weights=self.model.trainable_params())
        def closure():
            obj, params_gradient = grad_fn()
            return obj, params_gradient
        optimizer.step(closure, self.model.trainable_params(), self.model, self.device)
        model = self.model
        h_new = model.h_func().item()
        return h_new, h_mark

    def get_adj(self):
        W_est = self.model.fc1_to_adj()
        B_est = (abs(W_est) > self.threshold).astype(int)
        return B_est