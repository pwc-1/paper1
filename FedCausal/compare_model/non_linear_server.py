import copy
import random
import numpy as np
import torch
from tqdm import tqdm
from compare_model.non_linear_client import NonlinearClient
from utils.evaluation import MetricsDAG
from models.model import NonLinearModel
from utils.lbfgsb_scipy import LBFGSBScipy


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
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
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
            clienti = NonlinearClient(args, datas[i])
            self.models.append([clienti, temp])

    def run(self, B_true):
        self.B_true = B_true
        for t in tqdm(range(self.max_iter)):
            while self.rho < self.rho_max:
                #sample clients
                m = np.max([int(self.C * self.K), 1])
                index = random.sample(range(0, self.K), m)
                #clone fc1
                self.dispatch(index)
                #local update
                self.client_update(index)
                #aggregation
                h_new = self.aggregation(index)
                B_est = self.get_adj()
                met = MetricsDAG(B_est, self.B_true)
                print(met.metrics)
                if h_new > 0.25 * self.h:
                    self.rho *= 10
                else:
                    break
            self.h = h_new
            self.alpha += self.rho * self.h
            if self.h <= self.h_tol or self.rho >= self.rho_max:
                break

    def dispatch(self, index):
        for j in index:
            for old_params, new_params in zip(self.models[j][1].fc1_pos.parameters(), self.model.fc1_pos.parameters()):
                old_params.data = new_params.data.clone()
            for old_params, new_params in zip(self.models[j][1].fc1_neg.parameters(), self.model.fc1_neg.parameters()):
                old_params.data = new_params.data.clone()

    def client_update(self, index):
        for j in index:
            self.models[j][0].train(self.models[j][1], self.model, self.rho, self.alpha, self.h)

            W_est = self.models[j][1].fc1_to_adj()
            B_est = (abs(W_est) > self.threshold).astype(int)
            met = MetricsDAG(B_est, self.B_true)
            print('client '+str(j)+str(met.metrics))

    def aggregation(self, index):
        s = {}
        s_total = 0
        for j in index:
            s[j] = self.ns[j]
            s_total += s[j]

        params_pos = {}
        params_neg = {}
        for k, v in self.model.fc1_pos.named_parameters():
            params_pos[k] = torch.zeros_like(v.data)
        for k, v in self.model.fc1_neg.named_parameters():
            params_neg[k] = torch.zeros_like(v.data)
        for j in index:
            for k, v in self.models[j][1].fc1_pos.named_parameters():
                params_pos[k] += v.data * (s[j]/s_total)
            for k, v in self.models[j][1].fc1_neg.named_parameters():
                params_neg[k] += v.data * (s[j]/s_total)

        for k, v in self.model.fc1_pos.named_parameters():
            v.data = params_pos[k]
        for k, v in self.model.fc1_neg.named_parameters():
            v.data = params_neg[k]
        h_new = self.model.h_func().item()
        return h_new

    def get_adj(self):
        W_est = self.model.fc1_to_adj()
        B_est = (abs(W_est) > self.threshold).astype(int)
        return B_est
