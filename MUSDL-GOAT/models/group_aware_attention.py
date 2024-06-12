import math
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn


class Attention(nn.Cell):
    def __init__(self, dim, linear_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.linear_dim = linear_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # QKV matrix
        self.q_matrix = nn.Dense(linear_dim, linear_dim, has_bias=qkv_bias)
        self.k_matrix = nn.Dense(linear_dim, linear_dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.bn = nn.BatchNorm1d(540, eps=1e-05, momentum=0.9, affine=True)

        self.relu = nn.ReLU()
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeNormal(), cell.weight.shape, cell.weight.dtype))
                if cell.has_bias:
                    cell.bias.set_data(ms.common.initializer.initializer("zeros", cell.bias.shape, cell.bias.dtype))

    def construct(self, q_in, k_in, x):
        B, N, C = x.shape
        q = self.q_matrix(q_in).reshape(B, N, self.num_heads, self.linear_dim // self.num_heads).permute(0, 2, 1, 3)  # B,num_heads,N,C'
        k = self.k_matrix(k_in).reshape(B, N, self.num_heads, self.linear_dim // self.num_heads).permute(0, 2, 1, 3)  # B,num_heads,N,C'

        attn = (q @ k.swapaxes(-2, -1)) * self.scale  # B,num_heads,N,N
        attn = ops.softmax(attn)
        attn = self.attn_drop(attn)

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B,num_heads,N,C'
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # B,N,C
        x = x + (attn @ v).swapaxes(1, 2).reshape(B, N, C)  # B,N,C
        x = self.bn(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        q = q.permute(0, 2, 1, 3).reshape(B, N, self.linear_dim)
        k = k.permute(0, 2, 1, 3).reshape(B, N, self.linear_dim)
        return q, k, x, attn


class Encoder_Blocks(nn.Cell):
    def __init__(self, qk_dim, dim, linear_dim, num_heads, num_layers, attn_drop=0., proj_drop=0.):
        super(Encoder_Blocks, self).__init__()
        model_list = []
        for i in range(num_layers):
            model_list.append(Attention(dim, linear_dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop))
        self.model = nn.CellList(model_list)
        self.linear_q = nn.Dense(qk_dim, linear_dim)
        self.linear_k = nn.Dense(qk_dim, linear_dim)
        self.qk_dim = qk_dim

    def construct(self, q, k, x):
        attn_qk = 0
        q = self.linear_q(q)
        k = self.linear_k(k)
        for i, _layer in enumerate(self.model):
            q, k, x, attn = _layer(q, k, x)
            if i == 3:
                attn_qk = attn
        return x, attn_qk


def temporal_position_encoding(size):
    bs = size[0]
    max_len = size[1]
    d_model = size[2]
    pe = ops.zeros(max_len, d_model)
    position = ops.arange(0, max_len).unsqueeze(1)
    div_term = ops.exp(ops.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = ops.sin(position * div_term)
    pe[:, 1::2] = ops.cos(position * div_term)
    pe = pe.unsqueeze(0)
    pe_b = ops.cat([pe for i in range(bs)])
    return pe_b
