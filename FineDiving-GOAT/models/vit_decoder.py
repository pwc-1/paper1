import mindspore.nn as nn
import mindspore.ops as ops

from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)

        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B,num_heads,N,C'

        attn = (q @ k.swapaxes(-2, -1)) * self.scale  # B,num_heads,N,N
        attn = ops.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)  # B,N,C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Cell):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Dense(dim, out_dim, has_bias=qkv_bias)
        self.k_map = nn.Dense(dim, out_dim, has_bias=qkv_bias)
        self.v_map = nn.Dense(dim, out_dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)

        self.proj = nn.Dense(out_dim, out_dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.shape[1]

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.swapaxes(-2, -1)) * self.scale
        attn = ops.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Cell):
    def __init__(self, dim, num_heads, dim_q=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        dim_q = dim_q or dim
        self.norm_q = norm_layer((dim_q,))
        self.norm_v = norm_layer((dim,))
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, q, v):
        q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q


class decoder_fuser(nn.Cell):
    def __init__(self, dim, num_heads, num_layers):
        super(decoder_fuser, self).__init__()
        model_list = []
        for i in range(num_layers):
            model_list.append(DecoderBlock(dim, num_heads))
        self.model = nn.CellList(model_list)

    def construct(self, q, v):
        for _layer in self.model:
            q = _layer(q, v)
        return q
