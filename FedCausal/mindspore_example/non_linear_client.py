import mindspore
from mindspore import Tensor, value_and_grad
from mindspore_example.model import squared_loss
from mindspore_example.lbfgsb_scipy import LBFGSBScipy

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
        self.lambda3 = args.lambda3
        self.device = args.device

    # train function
    def train(self, model, global_model, global_rho, global_alpha, global_h):
        rho, alpha, h = global_rho, global_alpha, global_h
        optimizer = LBFGSBScipy(model.trainable_params())
        X_torch = Tensor.from_numpy(self.X)

        def forward_fn(X_torch):
            X_hat = model(X_torch)
            loss = squared_loss(X_hat, X_torch)
            h = model.h_func()
            l2_reg = 0.5 * self.lambda2 * model.l2_reg()
            l1_reg = self.lambda1 * model.fc1_l1_reg()
            L2 = 0.5 * self.lambda3 * model.local_L2(global_model)
            obj = loss + 0.5 * rho * h * h + alpha * h + l2_reg + l1_reg + L2
            return obj
        grad_fn = value_and_grad(forward_fn, grad_position=None, weights=model.trainable_params())
        def closure():
            obj, params_gradient = grad_fn(X_torch)
            return obj, params_gradient
        optimizer.step(closure, model.trainable_params(), model, self.device)