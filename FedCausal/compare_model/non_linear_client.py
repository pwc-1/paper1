import numpy as np
import torch
from models.model import squared_loss
from utils.lbfgsb_scipy import LBFGSBScipy


class NonlinearClient:
    def __init__(self, args, X):
        super().__init__()
        self.X = X
        self.n, self.d = X.shape
        self.max_iter = args.max_iter
        self.rho_max = args.rho_max
        self.h_tol = args.h_tol
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.device = args.device

    def train(self, model, global_model, global_rho, global_alpha, global_h):
        rho, alpha, h = global_rho, global_alpha, global_h
        optimizer = LBFGSBScipy(model.parameters())
        X_torch = torch.from_numpy(self.X).to(self.device)

        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = squared_loss(X_hat, X_torch)
            h = model.h_func()

            l2_reg = 0.5 * self.lambda2 * model.l2_reg()
            l1_reg = self.lambda1 * model.fc1_l1_reg()

            #L2 = 0.5 * self.lambda1 * model.local_L2(global_model) #with ho data
            obj = loss + 0.5 * rho * h * h + alpha * h + l2_reg + l1_reg #+ L2
            obj.backward()
            return obj
        optimizer.step(closure, self.device)