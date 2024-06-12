import mindspore as ms
from mindspore import nn, Tensor, ops
import mindspore.common.initializer as init
from mindspore.ops import stop_gradient
import numpy as np

from Utils.input2hidden import Input2Hidden
from Utils.hidden2latent import Hidden2Latent
from Utils.locally_connected import LocallyConnected
from Utils.lbfgsb_scipy import LBFGSBScipy

class SAE(nn.Cell):
    def __init__(self,
                 neurons,
                 act_func,
                 bias=True):
        super().__init__()
        self.neurons = neurons
        self.act_func = act_func
        self.bias = bias
        self.d = neurons[0]
        self.q = neurons[-1]

        # encoder
        encoder = nn.SequentialCell()
        encoder.append(Input2Hidden(num_linear=self.d,input_features=1,output_features=neurons[1],bias=bias))
        encoder.append(self.act_func)
        encoder.append(Hidden2Latent(num_linear=self.d,input_features=neurons[1],output_features=neurons[2],bias=bias))
        self.encoder = encoder
        # decoder
        decoder = nn.SequentialCell()
        decoder.append(nn.Dense(in_channels=neurons[-1],out_channels=neurons[-2],has_bias=bias))
        decoder.append(self.act_func)
        decoder.append(nn.Dense(in_channels=neurons[-2],out_channels=neurons[-3],has_bias=bias))
        self.decoder = decoder

        self._init_weights()

    def _init_weights(self):
        for m in self.encoder:
            if isinstance(m, Input2Hidden) or isinstance(m, Hidden2Latent):
                m.weight.set_dtype(ms.float64)
                m.weight.set_data(init.initializer(init=init.Uniform(0.1), shape=m.weight.data.shape, dtype=ms.float64))
                m.bias.set_dtype(ms.float64)
                m.bias.set_data(init.initializer(init=init.Zero(), shape=m.bias.data.shape, dtype=ms.float64))
        for m in self.decoder:
            if isinstance(m, nn.Dense):
                m.weight.set_dtype(ms.float64)
                m.bias.set_dtype(ms.float64)

    def construct(self, x):  # x:[n, d]
        x = x.t().unsqueeze(2)  # [d, n, 1]
        z = Tensor.sum(self.encoder(x), 0)  # [n, q]
        y = self.decoder(z)  # [n, d]
        return z, y

    def get_l1reg(self):
        l1_reg = 0.
        for l in self.encoder:
            if isinstance(l, Input2Hidden) or isinstance(l, Hidden2Latent):
                l1_reg += Tensor.sum(Tensor.abs(l.weight.data))
                l1_reg += Tensor.sum(Tensor.abs(l.bias.data))
        return l1_reg

    def get_l2reg(self):
        l2_reg = 0.
        for l in self.encoder.modules():
            if isinstance(l, Input2Hidden) or isinstance(l, Hidden2Latent):
                l2_reg += Tensor.sum(l.weight.data ** 2)
                l2_reg += Tensor.sum(l.bias.data ** 2)
        return l2_reg

    def get_path_product(self):  # -> [d, q]
        A = Tensor.abs(self.encoder[0].weight.data)  # [d, 1, m1]
        A = A.matmul(Tensor.abs(self.encoder[2].weight.data))  # [d, 1, q] = [d, 1, m1] @ [d, m1, q]
        A = A.squeeze(1)  # [d, q]
        return A

class MLP(nn.Cell):
    def __init__(self,
                 neurons,
                 q,
                 act_func,
                 bias=True
                 ):
        super().__init__()
        self.neurons = neurons
        self.act_func = act_func
        self.bias = bias
        d = neurons[0]
        self.d = d
        self.q = q

        # the number of inputs is d+q
        self.fw_pos = nn.Dense(d+q, d*neurons[1], has_bias=bias)
        self.fw_neg = nn.Dense(d+q, d*neurons[1], has_bias=bias)
        self.init_bounds = self._bounds()
        self.fw_pos.weight.bounds = self.init_bounds
        self.fw_neg.weight.bounds = self.init_bounds

        # the number of MLPs is d
        fc = nn.SequentialCell()
        for l in range(len(neurons)-2):
            fc.append(self.act_func)
            fc.append(LocallyConnected(d, neurons[l+1], neurons[l+2], bias=bias))
        self.fc = fc

        self._init_weights()

    def _bounds(self):
        d = self.d
        q = self.q
        bounds = []
        for j in range(d):
            for m in range(self.neurons[1]):
                for i in range(d+q):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def _init_weights(self):
        for n, m in self.parameters_and_names():
            if n.split('.')[-1] == 'weight':
                m.set_dtype(ms.float64)
                m.set_data(init.initializer(init=init.Uniform(), shape=m.data.shape, dtype=ms.float64))
            elif n.split('.')[-1] == 'bias':
                m.set_dtype(ms.float64)

    def construct(self, xz):  # xz:[n, d+q]
        xz = self.fw_pos(xz) - self.fw_neg(xz)  # [n, d*m1]
        xz = xz.view(-1, self.neurons[0], self.neurons[1])  # [n, d, m1]
        xz = self.fc(xz)
        xz = xz.squeeze(axis=2)  # [n, d]
        return xz

    def get_fw(self):
        return self.fw_pos.weight - self.fw_neg.weight

    def get_abs_fw(self):
        return self.fw_pos.weight + self.fw_neg.weight

    def get_l1reg(self):
        l1_reg = Tensor.sum(self.get_abs_fw())
        return l1_reg

    def get_l2reg(self):
        l2_reg = Tensor.sum(self.get_fw() ** 2)
        # l2_reg = 0.
        for f in self.fc:
            if isinstance(f, LocallyConnected):
                l2_reg += Tensor.sum(f.weight ** 2)
        return l2_reg

    def get_sole(self):
        d, q = self.d, self.q
        fw = self.get_fw()  # [d*m1, d+q]
        fw = fw.view(d, -1, d + q)  # [d, m1, d+q]
        S = Tensor.sum(fw * fw, axis=1).t()[d:,:]  # [d+q,d]
        reg = Tensor.sum(S)-Tensor.sum(Tensor.max(S,1)[0])
        return reg

    def get_D(self, A):  # D = C.*C
        d, q = self.d, self.q
        fw = self.get_fw()  # [d*m1, d+q]
        fw = fw.view(d, -1, d+q).transpose(0, 2, 1)  # [d, d+q, m1]
        S = Tensor.matmul(A, fw[:, d:, :])
        S[S<0.3] = 0
        D = S + fw[:, :d, :]  # [d, d, m1]
        D = Tensor.sum(D * D, axis=2).t()
        return D

    def get_macro_D(self):  # -> S:qxd
        d, q = self.d, self.q
        fw = self.get_fw().data  # [d*m1, d+q]
        fw = fw.view(d, -1, d + q)  # [d, m1, d+q]
        S = Tensor.sqrt(Tensor.sum(fw * fw, axis=1).t())[d:,]  # [q,d]
        return S

    def h_func(self, A):
        d = self.d
        D = self.get_D(A)
        from Utils.Acyclic import acyclic
        D[D < 0.01] = 0
        h = acyclic(D)/d
        return h

    def get_penalty(self, A):
        d, q = self.d, self.q
        A = Tensor.abs(A)
        fw = self.get_abs_fw()  # [d*m1, d+q]
        fw = fw.view(d, -1, d+q).transpose(0, 2, 1)  # [d, d+q, m1]
        return Tensor.sum(Tensor.sum(A.matmul(fw[:,d:,:]), axis=2).multiply(Tensor.sum(fw[:,:d,:], axis=2)))

    def get_macro_adj(self):  # -> S:(d+q)xd
        d, q = self.d, self.q
        fw = self.get_fw()  # [d*m1, d+q]
        fw = fw.view(d, -1, d + q)  # [d, m1, d+q]
        S = Tensor.sqrt(Tensor.sum(fw * fw, axis=1).t())  # [d+q,d]
        S = S.cpu().detach().numpy()
        return S

    def get_micro_adj(self, A):
        D = self.get_D(A)
        C = Tensor.sqrt(D)
        C = C.numpy()
        return C



class MgCSL(nn.Cell):
    def __init__(self,
                 AEneurons,
                 MLPneurons,
                 device_type,
                 device_num=0,
                 macro_graph=False,
                 sae_activation='LeakyReLU',
                 mlp_activation='Tanh',
                 bias=True,
                 seed=24,
                 mu=1e-3,
                 gamma=0.,
                 eta=300,
                 max_iter=100,
                 h_tol=0.1,
                 mu_max=1e+16,
                 C_threshold=0.2):
        super().__init__()
        self.AEneurons = AEneurons
        self.MLPneuron = MLPneurons
        self.device_type = device_type
        self.device_num = device_num
        self.macro_graph = macro_graph
        self.sae_activation = sae_activation
        self.mlp_activation = mlp_activation
        self.bias = bias
        self.seed = seed
        self.mu = mu
        self.gamma = gamma
        self.eta = eta
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.mu_max = mu_max
        self.C_threshold = C_threshold

        if device_type == 'cpu':
            ms.set_context(device_target='CPU')
        elif device_type == 'gpu':
            ms.set_context(device_target='GPU', device_id=device_num)
        else:
            raise ValueError("GPU is unavailable, please set device_type to 'cpu'")
        sae_act_func = eval('nn.{}()'.format(sae_activation))
        mlp_act_func = eval('nn.{}()'.format(mlp_activation))
        self.sae_act_func = sae_act_func
        self.mlp_act_func = mlp_act_func
        self.criterion = nn.MSELoss()
        self.d = AEneurons[0]
        self.q = AEneurons[-1]
        import random
        random.seed(seed)
        np.random.seed(seed)

        self.sae = SAE(neurons=AEneurons, act_func=sae_act_func, bias=bias)
        self.mlp = MLP(neurons=MLPneurons, q=self.q, act_func=mlp_act_func, bias=bias)

    def construct(self, x):
        z, y = self.sae(x)
        xz = ops.concat([x,z], 1)
        x_hat = self.mlp(xz)
        return y, x_hat

    def forward_fn(self, x, mu, gamma):
        # loss of auto-encoder
        A = self.sae.get_path_product()
        y, x_hat = self(x)
        encoder_l1reg = 0.1 * self.sae.get_l1reg()
        L1 = 0.01 * self.squared_loss(y, x) + encoder_l1reg
        # loss of mlp
        mlp_l1reg = 0.01 * self.mlp.get_l1reg()
        mlp_l2reg = 0.5 * 0.01 * self.mlp.get_l2reg()
        h_val = self.mlp.h_func(A)
        acyclic = 0.5 * mu * h_val * h_val + gamma * h_val
        penalty = 0.01 * self.mlp.get_penalty(A)
        L2 = self.squared_loss(x_hat, x) + mlp_l1reg + mlp_l2reg + acyclic + penalty + self.mlp.get_sole()
        obj = L1 + L2
        return obj

    def squared_loss(self, output, target):
        n = target.shape[0]
        loss = 0.5 / n * Tensor.sum((output - target) ** 2)
        return loss

    def learn(self, x):
        if not isinstance(x, Tensor):
            raise ValueError('Type of x must be tensor!')
        mu, gamma, eta, h = self.mu, self.gamma, self.eta, 1e16
        max_iter, h_tol, mu_max, C_threshold = self.max_iter, self.h_tol, self.mu_max, self.C_threshold
        optimizer = LBFGSBScipy(self.trainable_params(), 1e-3)
        grad_fn = ms.value_and_grad(self.forward_fn, None, optimizer.parameters)
        for _ in range(max_iter):
            while mu < mu_max:
                def closure():
                    obj, grads = grad_fn(x, mu, gamma)
                    return obj
                optimizer.step(closure)

                A = self.sae.get_path_product()
                h_new = self.mlp.h_func(A).item()
                h_new = stop_gradient(h_new)
                if h_new > 0.25*h:
                    mu *= eta
                else:
                    break
            gamma += mu*h_new
            h = h_new
            if h_new <= h_tol or mu >= mu_max:
                break
        A = self.sae.get_path_product()
        if self.macro_graph:
            S = self.mlp.get_macro_adj()
            return A, S
        else:
            C = self.mlp.get_micro_adj(A)

            C[C<C_threshold] = 0
            from Utils.is_acyclic import is_acyclic
            # Find the smallest threshold that removes all cycle-inducing edges
            thresholds = np.unique(C)
            epsilon = 1e-8
            for step, t in enumerate(thresholds):
                to_keep = np.array(C > t + epsilon)
                new_adj = C * to_keep
                if is_acyclic(new_adj):
                    C = new_adj
                    break
            C[C!=0] = 1
            return C