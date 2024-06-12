import random
import numpy as np
from tqdm import tqdm
from compare_model.linear_client import LinearClient
from utils.evaluation import MetricsDAG


class LinearServer:
    def __init__(self, args, datas):
        super().__init__()
        self.d = args.d
        self.ns = args.ns
        self.W_est = np.zeros(2 * self.d * self.d)
        self.K = args.K
        self.C = args.C
        self.max_iter = args.max_iter
        self.lambda2 = args.lambda2
        self.rho_max = args.rho_max
        self.h_tol = args.h_tol
        self.rho = 1.0
        self.alpha = 0.0
        self.h = np.inf
        self.threshold = args.threshold
        self.clients = []
        for i in range(self.K):
            # temp = np.array(list(self.W_est))
            client_i = LinearClient(args, datas[i], self.W_est)
            self.clients.append(client_i)

    def run(self, B_ture):
        for t in tqdm(range(self.max_iter)):
            while self.rho < self.rho_max:
                # sample clients
                m = np.max([int(self.C * self.K), 1])
                index = random.sample(range(0, self.K), m)
                # clone fc1
                self.dispatch(index)
                # local update
                self.client_update(index)
                # aggregation
                w_new, h_new = self.aggregation(index)
                if h_new > 0.25 * self.h:
                    self.rho *= 10
                else:
                    break
            self.W_est, self.h = w_new, h_new
            self.alpha += self.rho * self.h
            if self.h <= self.h_tol or self.rho >= self.rho_max:
                break
            B_est = self.get_adj()
            met = MetricsDAG(B_est, B_ture)
            print(met.metrics)

    def dispatch(self, index):
        for j in index:
            self.clients[j].set_W_est(self.W_est)

    def client_update(self, index):
        for j in index:
            self.clients[j].train(self.W_est, self.rho, self.alpha, self.h)

    def aggregation(self, index):
        s = {}
        s_total = 0
        for j in index:
            s[j] = self.ns[j]
            s_total += s[j]
        params = np.zeros_like(self.W_est)
        for j in index:
            params += self.clients[j].get_W_est() * (s[j]/s_total)

        w_new = params
        h_new, _ = self._h(self._adj(w_new))
        return w_new, h_new
        # def _func(W_est):
        #     W = self._adj(W_est)
        #     W_z = self._adj(params)
        #     L2, G_L2 = self._L2(W, W_z)
        #     h, G_h = self._h(W)
        #     obj = L2 + 0.5 * self.rho * h * h + self.alpha * h + self.lambda2 * W.sum()
        #     G_smooth = G_L2 + (self.rho * h + self.alpha) * G_h
        #     g_obj = np.concatenate((G_smooth + self.lambda2, - G_smooth + self.lambda2), axis=None)
        #     return obj, g_obj
        # bnds = [(0, 0) if i == j else (0, None) for _ in range(2)
        #         for i in range(self.d) for j in range(self.d)]
        # sol = sopt.minimize(_func, self.W_est, method='L-BFGS-B', jac=True, bounds=bnds)
        # w_new = sol.x
        # h_new, _ = self._h(self._adj(w_new))
        # return w_new, h_new

    def _adj(self, W_est):
        return (W_est[:self.d * self.d] - W_est[self.d * self.d:]).reshape([self.d, self.d])

    def _h(self, W_est):
        M = np.eye(self.d) + W_est * W_est / self.d  # (Yu et al. 2019)
        E = np.linalg.matrix_power(M, self.d - 1)
        h = (E.T * M).sum() - self.d
        G_h = E.T * W_est * 2
        return h, G_h

    def _L2(self, W_est, global_W_est):
        L2 = 0.5 * ((W_est-global_W_est) ** 2).sum()
        G_L2 = 1.0 * (W_est-global_W_est)
        return L2, G_L2

    def get_adj(self):
        W_est = (self.W_est[:self.d * self.d] - self.W_est[self.d * self.d:]).reshape([self.d, self.d])
        B_est = (abs(W_est) > self.threshold).astype(int)
        return B_est
