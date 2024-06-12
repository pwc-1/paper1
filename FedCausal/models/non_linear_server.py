import copy
import random
import numpy as np
import torch
from tqdm import tqdm
from models.model import NonLinearModel
from models.non_linear_client import NonlinearClient
from utils.lbfgsb_scipy import LBFGSBScipy


#linear model server
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
            clienti = NonlinearClient(args, datas[i])
            self.models.append([clienti, temp])

    # algorithm run function
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

    # server send parameters to clients
    def dispatch(self, index):
        for j in index:
            parameters_size = 0
            for old_params, new_params in zip(self.models[j][1].fc1_pos.parameters(), self.model.fc1_pos.parameters()):
                parameters_size += sum([p.numel() for p in new_params])
                old_params.data = new_params.data.clone()
            for old_params, new_params in zip(self.models[j][1].fc1_neg.parameters(), self.model.fc1_neg.parameters()):
                parameters_size += sum([p.numel() for p in new_params])
                old_params.data = new_params.data.clone()

    # clients update parameters
    def client_update(self, index):
        for j in index:
            self.models[j][0].train(self.models[j][1], self.model, self.rho, self.alpha, self.h)

    # server aggregate clients' parameters
    def aggregation(self, index):
        s = {}
        s_total = 0
        for j in index:
            s[j] = self.ns[j]
            s_total += s[j]

        params_pos, params_neg, params_poss, params_negs = {}, {}, {}, {}
        for k, v in self.model.fc1_pos.named_parameters():
            params_pos[k] = torch.zeros_like(v.data)
        for k, v in self.model.fc1_neg.named_parameters():
            params_neg[k] = torch.zeros_like(v.data)
        for j in index:
            params_poss[j], params_negs[j] = {}, {}
            for k, v in self.models[j][1].fc1_pos.named_parameters():
                params_pos[k] += v.data * (s[j] / s_total)
                params_poss[j][k] = v.data
            for k, v in self.models[j][1].fc1_neg.named_parameters():
                params_neg[k] += v.data * (s[j] / s_total)
                params_negs[j][k] = v.data
        for k, v in self.model.fc1_pos.named_parameters():
            v.data = params_pos[k]
        for k, v in self.model.fc1_neg.named_parameters():
            v.data = params_neg[k]
        h_mark = self.model.h_func().item()

        h_new = None
        optimizer = LBFGSBScipy(self.model.parameters())
        def closure():
            optimizer.zero_grad()
            tol_L2 = 0
            for j in index:
                L2 = 0.5 * self.model.global_L2(params_poss[j], params_negs[j])
                tol_L2 += L2 * (s[j]/s_total)
            h = self.model.h_func()
            obj = tol_L2 + 0.5 * self.rho * h * h + self.alpha * h
            obj.backward()
            return obj
        optimizer.step(closure, self.device)
        with torch.no_grad():
            model = self.model.to(self.device)
            h_new = model.h_func().item()
        return h_new, h_mark

    # output adjacency matrix
    def get_adj(self):
        W_est = self.model.fc1_to_adj()
        B_est = (abs(W_est) > self.threshold).astype(int)
        return B_est
