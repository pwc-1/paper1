from einops import rearrange, repeat, reduce
import numpy as np


from models2.weeight_init_MS import trunc_normal_, constant_init_, kaiming_init_
import math
import numpy as np
from mindspore import nn
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.nn import TransformerEncoder, Dense
# from mindspore.nn.transformer import MultiHeadAttention
from mindspore.nn.layer.transformer import MultiheadAttention
# from mindspore. import MultiHeadAttention
from mindspore.common import dtype as mstype
import mindspore.numpy as mnp
import mindspore.common.initializer as weight_init
from mindspore.common.initializer import Uniform
import mindspore.ops as ops


def _calculate_fan_in_and_fan_out(tensor):
    """_calculate_fan_in_and_fan_out"""
    dimensions = len(tensor)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = tensor[1]
        fan_out = tensor[0]
    else:
        num_input_fmaps = tensor[1]
        num_output_fmaps = tensor[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[2] * tensor[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out

def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    res = 0
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            neg_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            neg_slope = param
        else:
            raise ValueError("neg_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + neg_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    return res

def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Unsupported mode {}, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def kaiming_uniform(inputs_shape, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, size=inputs_shape).astype(np.float32)

def get_sine_cosine_pos_emb(n_position, d_hid): 
    ''' Sinusoid position encoding table '''
    def get_position_angle_vec(position):
        return [position / mnp.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = mnp.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = mnp.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = mnp.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return mnp.array(sinusoid_table).unsqueeze(0)


class DropPath(nn.Cell):
    def __init__(self, dropout_p=None):
        super(DropPath, self).__init__()
        self.dropout_p = dropout_p

    def construct(self, x):
        return self.drop_path(x, self.dropout_p, self.training)

    def drop_path(self, x, dropout_p=0., training=False):
        if dropout_p == 0. or not training:
            return x
        keep_prob = 1 - dropout_p
        shape = (ops.shape(x)[0],) + (1,) * (ops.ndim(x) - 1)
        random_tensor = keep_prob + ops.random.uniform(shape).astype(x.dtype)
        random_tensor = ops.floor(random_tensor)  # binarize
        output = x / keep_prob * random_tensor
        return output

     

class ClassificationHead(nn.Cell):
    def __init__(self, num_classes, in_channels, init_std=0.02, eval_metrics='finetune'):
        super(ClassificationHead, self).__init__()
        self.init_std = init_std
        self.eval_metrics = eval_metrics
        self.cls_head = nn.Dense(in_channels, num_classes, weight_init=weight_init.TruncatedNormal(init_std))

    def construct(self, x):
        cls_score = self.cls_head(x)
        return cls_score




class PatchEmbed(nn.Cell):
    def __init__(self, img_size, patch_size, tube_size=2, in_channels=3, embed_dims=768, conv_type='Conv2d'):
        super(PatchEmbed, self).__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.tube_size = tube_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.conv_type = conv_type
        
        # self.weights_init = Uniform(-1,1)

        num_patches = (self.img_size[1] // self.patch_size[1]) * (self.img_size[0] // self.patch_size[0])
        self.num_patches = num_patches

        if conv_type == 'Conv2d':
            self.projection = nn.Conv2d(in_channels, embed_dims, kernel_size=patch_size, stride=patch_size)
        elif conv_type == 'Conv3d':
            self.projection = nn.Conv3d(in_channels, embed_dims, kernel_size=(tube_size, patch_size, patch_size),
                                         stride=(tube_size, patch_size, patch_size))
        else:
            raise TypeError(f'Unsupported conv layer type {conv_type}')

        self.init_weights(self.projection)

    def init_weights(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            # module.weight.set_data(initializer('kaiming_uniform', module.weight.shape, a=0, mode='fan_in', nonlinearity='relu'))
            module.weight = Tensor(kaiming_uniform(module.weight.shape, a=0, mode='fan_in', nonlinearity='relu'))
            # module.weight.set_data(self.weights_init(module.weight.shape))
            # module.weight.set_data()
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.set_data(initializer('zeros', module.bias.shape))

    def construct(self, x):
        layer_type = type(self.projection)
        if layer_type == nn.Conv3d:
            x = P.Reshape()(x, (x.shape[0], x.shape[1], x.shape[2], x.shape[3] * x.shape[4]))
            x = self.projection(x)
            x = P.Reshape()(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        elif layer_type == nn.Conv2d:
            x = P.Reshape()(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
            x = self.projection(x)
            x = P.Reshape()(x, (x.shape[0], x.shape[1] * x.shape[2]))
        else:
            raise TypeError(f'Unsupported conv layer type {layer_type}')

        return x

    
class FFNWithPreNorm(nn.Cell):
    def __init__(self, embed_dims=256, hidden_channels=1024, num_layers=2, act_layer=nn.GELU, norm_layer=nn.LayerNorm, dropout_p=0.2, layer_drop=None, **kwargs):
        super(FFNWithPreNorm, self).__init__()
        assert num_layers >= 2, f'num_layers should be no less than 2. got {num_layers}.'
        self.embed_dims = embed_dims
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.norm = norm_layer((embed_dims,))
        layers = []
        in_channels = embed_dims
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Dense(in_channels, hidden_channels),
                act_layer(),
                nn.Dropout(dropout_p)
            ])
            in_channels = hidden_channels
        layers.extend([
            nn.Dense(hidden_channels, embed_dims),
            nn.Dropout(dropout_p)
        ])
        self.layers = nn.SequentialCell(layers)

        if layer_drop:
            dropout_p = layer_drop.pop('dropout_p')
            layer_drop = layer_drop.pop('type')
            self.layer_drop = layer_drop(dropout_p)
        else:
            self.layer_drop = nn.Identity()

    def construct(self, x):
        residual = x
        x = self.norm(x)
        x = self.layers(x)
        return residual + self.layer_drop(x)


class TransformerContainer(nn.Cell):

    def __init__(self,
                 num_transformer_layers,
                 embed_dims,
                 num_heads,
                 num_frames,
                 hidden_channels,
                 operator_order,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 num_layers=2):
        super().__init__()
        self.layers = nn.CellList([])
        self.num_transformer_layers = num_transformer_layers
        dpr = np.linspace(0, drop_path_rate, num_transformer_layers)
        for i in range(num_transformer_layers):
            self.layers.append(
                BasicTransformerBlock(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_frames=num_frames,
                    hidden_channels=hidden_channels,
                    operator_order=operator_order,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    num_layers=num_layers,
                    dpr=dpr[i] + 0.01))

    def construct(self, x, return_attention=False):
        layer_idx = 0
        for layer in self.layers:
            if layer_idx >= self.num_transformer_layers - 1 and return_attention:
                x = layer(x, return_attention=True)
            else:
                x = layer(x)
            layer_idx += 1
        return x

class LocalAttention(nn.Cell):
    """
    LW-MSA: Local Window-based MSA
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., local_size=(1, 1, 1)):
        super(LocalAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.local_size = local_size

    def construct(self, x, D, H, W):
        B, N, C = x.shape

        nd, nh, nw = D // self.local_size[0], H // self.local_size[1], W // self.local_size[2]
        nl = nd * nh * nw  # the number of local windows

        x = x.view(B, nd, self.local_size[0], nh, self.local_size[1],
                      nw, self.local_size[2], C).permute(0, 1, 3, 5, 2, 4, 6, 7)

        qkv = self.qkv(x).view(B, nl, -1, 3, self.num_heads,
                                  C // self.num_heads).permute(3, 0, 1, 4, 2, 5)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = P.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(
            B, nd, nh, nw, self.local_size[0], self.local_size[1], self.local_size[2], C)
        x = attn.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GlobalAttention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 fine_pysize=(8, 7, 7),  resolution=(16, 56, 56), stage=0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.kv = nn.Dense(dim, dim * 2, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.resolution = resolution
        self.stage = stage

        self.fine_pysize = fine_pysize
        self.fine_kernel_size = [rs // ts for rs,
                                 ts in zip(resolution, fine_pysize)]
        self.fine_kernel_size = [2, 1, 1]
        self.fine_kernel_size = self.fine_kernel_size[0]
        if np.prod(self.fine_kernel_size) > 1:
            # Fine-grained level pyramid
            self.sr = nn.Conv3d(dim, dim, kernel_size=self.fine_kernel_size, stride=self.fine_kernel_size, group=dim)

            # self.sr = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=self.fine_kernel_size, stride=self.fine_kernel_size, group=dim)
            self.norm = nn.LayerNorm([int(dim)])

    def construct(self, x, D, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(
            B, N, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)

        if np.prod(self.fine_kernel_size) > 1:
            x_ = x.transpose(0, 2, 1).reshape(B, C, D, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).transpose(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).transpose(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).transpose(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = nn.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BasicTransformerBlock(nn.Cell):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 hidden_channels,
                 operator_order,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 num_layers=2,
                 dpr=0.2,
                 ):

        super().__init__()
        self.attentions = nn.CellList([])
        self.ffns = nn.CellList([])

        for i, operator in enumerate(operator_order):
            # if operator == 'self_attn':
            #     self.attentions.append(
            #         MultiheadAttention(embed_dims=embed_dims, num_heads=num_heads, dropout_rate=dpr))
            # elif operator == 'time_attn':
            #     self.attentions.append(
            #         nn.TransformerEncoderLayer(d_model=embed_dims, nhead=num_heads))
            if operator == "local_attn":
                self.attentions.append(
                    LocalAttention(
						dim=embed_dims,
						# num_heads=num,
						num_heads=num_heads,
						qkv_bias=False,
						qk_scale=None,
						attn_drop=0.2,
						proj_drop=0.2,
                        # attn_drop=0.,
						# proj_drop=0.,
						local_size=(1, 1, 1)
					))
            elif operator == "global_attn":
                self.attentions.append(
                    GlobalAttention(
						dim=embed_dims,
						num_heads=8,
						qkv_bias=False,
						qk_scale=None,
						# attn_drop=0.,
						# proj_drop=0.,
                        attn_drop=0.2,
						proj_drop=0.2,
						fine_pysize=(8, 7, 7),
						resolution=(16, 56, 56),
						stage=0
					))
            # elif operator == 'space_attn':
            #     self.attentions.append(
            #         nn.MultiheadAttention(embed_dim=embed_dims, num_heads=num_heads, dropout=0))
            elif operator == 'ffn':
                self.ffns.append(
                    FFNWithPreNorm(
						embed_dims=embed_dims,
						hidden_channels=hidden_channels,
						num_layers=num_layers,
						act_layer=act_layer,
						norm_layer=norm_layer,
						layer_drop=dict(type=DropPath, dropout_p=dpr))
                )
            else:
                raise TypeError(f'Unsupported operator type {operator}')

    def construct(self, x, return_attention=False):
        attention_idx = 0

        for layer in self.attentions:
            if attention_idx >= len(self.attentions)-1 and return_attention:
                x = layer(x, x, x)[0]
                return x
            else:
                x = layer(x)
            attention_idx += 1

        for layer in self.ffns:
            x = layer(x)
        return x


