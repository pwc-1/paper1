import numpy as np
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


class LinearClient:
    def __init__(self, args, X, W_est):
        super().__init__()
        self.X = X
        self.n, self.d = X.shape
        self.W_est = W_est
        self.max_iter = args.max_iter
        self.rho_max = args.rho_max
        self.h_tol = args.h_tol
        self.lambda1 = args.lambda1
        self.loss_type = args.loss_type

    def train(self, global_W_est, global_rho, global_alpha, global_h):
        def _func(W_est):
            W = self._adj(W_est)
            global_W = self._adj(global_W_est)
            loss, G_loss = self._loss(W)
            h, G_h = self._h(W)
            #L2, G_L2 = self._L2(W, global_W, self.lambda1)
            obj = loss + 0.5 * rho * h * h + alpha * h + self.lambda1 * W.sum() #+ L2
            G_smooth = G_loss + (rho * h + alpha) * G_h #+ G_L2
            g_obj = np.concatenate((G_smooth + self.lambda1, - G_smooth + self.lambda1), axis=None)
            return obj, g_obj
        rho, alpha, h = global_rho, global_alpha, global_h
        bnds = [(0, 0) if i == j else (0, None) for _ in range(2)
                for i in range(self.d) for j in range(self.d)]
        sol = sopt.minimize(_func, self.W_est, method='L-BFGS-B', jac=True, bounds=bnds)
        self.W_est = sol.x

    def _loss(self, W_est):
        M = self.X @ W_est
        if self.loss_type == 'l2':
            R = self.X - M
            loss = 0.5 / self.X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / self.X.shape[0] * self.X.T @ R
        elif self.loss_type == 'logistic':
            loss = 1.0 / self.X.shape[0] * (np.logaddexp(0, M) - self.X * M).sum()
            G_loss = 1.0 / self.X.shape[0] * self.X.T @ (sigmoid(M) - self.X)
        elif self.loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / self.X.shape[0] * (S - self.X * M).sum()
            G_loss = 1.0 / self.X.shape[0] * self.X.T @ (S - self.X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _adj(self, W_est):
        return (W_est[:self.d * self.d] - W_est[self.d * self.d:]).reshape([self.d, self.d])

    def _h(self, W_est):
        M = np.eye(self.d) + W_est * W_est / self.d  # (Yu et al. 2019)
        E = np.linalg.matrix_power(M, self.d - 1)
        h = (E.T * M).sum() - self.d
        G_h = E.T * W_est * 2
        return h, G_h

    def _L2(self, W_est, global_W_est, lambda1):
        L2 = 0.5 * lambda1 * ((W_est-global_W_est) ** 2).sum()
        G_L2 = 1.0 * lambda1 * (W_est-global_W_est)
        return L2, G_L2

    def set_W_est(self,W_est):
        self.W_est = W_est

    def get_W_est(self):
        return self.W_est

