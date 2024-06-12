import mindspore
import mindspore as ms
from mindspore import nn, Tensor, ops, grad
import scipy.optimize as sopt

class LBFGSBScipy(nn.Optimizer):
    def __init__(self, params, learning_rate=0.01):
        super(LBFGSBScipy, self).__init__(learning_rate, params)
        self._numel = sum([p.numel() for p in self.parameters])

    def _gather_flat_grad(self, gradient):
        views = []
        for p in gradient:
            view = p.view(-1)
            views.append(view)
        return ops.cat(views, 0)

    def _gather_flat_bounds(self, params):
        bounds = []
        for p in params:
            if hasattr(p, 'bounds'):
                b = p.bounds
            else:
                b = [(None, None)] * p.numel()
            bounds += b
        return bounds

    def _gather_flat_params(self, params):
        views = []
        for p in params:
            view = p.data.view(-1)
            views.append(view)
        return ops.concat(views, 0)

    def _distribute_flat_params(self, model, params):
        offset = 0
        for p in model.trainable_params():
            numel = p.numel()
            p.data.set_data(params[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel

    def step(self, closure, params, model, device):

        def wrapped_closure(flat_params):
            """closure must call zero_grad() and backward()"""
            flat_params = Tensor.from_numpy(flat_params)
            flat_params = flat_params.to(ms.float32)
            self._distribute_flat_params(model, flat_params)
            loss, grad = closure()
            loss = loss.item()
            gradient = self._gather_flat_grad(grad).numpy()
            return loss, gradient.astype('float64')

        initial_params = self._gather_flat_params(params)
        initial_params = initial_params.numpy()

        bounds = self._gather_flat_bounds(params)

        # Magic
        sol = sopt.minimize(wrapped_closure,
                            initial_params,
                            method='L-BFGS-B',
                            jac=True,
                            bounds=bounds)

        final_params = Tensor.from_numpy(sol.x)
        final_params = final_params.to(ms.float32)
        self._distribute_flat_params(model, final_params)