import math
import logging
from functools import partial
from collections import OrderedDict
import os
from symbol import parameters 
import mindspore as ms
import mindspore.ops as ops
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.nn import Cell, Dense, Dropout, SequentialCell
from mindspore import Tensor
import mindspore.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv, resolve_pretrained_cfg, checkpoint_seq
from timm.models.layers import trunc_normal_, lecun_normal_, _assert,DropPath
from timm.models.layers.helpers import to_2tuple
from timm.models.registry import register_model
from .representation import *
import mindvision as msd
    
from timm.models.registry import register_model

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class Mlp(nn.Cell):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        
        self.fc1 = Dense(in_features, hidden_features, has_bias=True)
        self.act = act_layer()
        #self.drop1 = ops.dropout()
        self.fc2 = Dense(hidden_features, out_features, has_bias=True)
        #self.drop2 = ops.dropout)
        #self.drop = nn.Dropout(0.9)
    def construct(self, x):  
        # print("premlp"+str(x))
        # print(ops.sum(x))
        x = self.fc1(x)
        # print("fc1"+str(x))         
        # print(ops.sum(x))        
        x = self.act(x)


        #x = self.drop1(x)
        x = self.fc2(x)

        #x = self.drop(x)
        # print("postmlp"+str(x))
        # print(ops.sum(x))
        return x

class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Dense(dim, dim * 3, has_bias=True)
        self.attn_drop =attn_drop
        self.proj = Dense(dim, dim, has_bias=True)
        self.proj_drop = proj_drop



    def construct(self, x):
        #ones=ops.Ones()
        #x=ones((32,197,768),mindspore.float32)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(0,1,-1, -2)) * self.scale
        attn = ops.softmax(attn,axis=-1)
        #attn = ops.dropout(attn,self.attn_drop)

        x = (attn @ v).transpose(0,2,1,3).reshape(B, N, C)
        x = self.proj(x)
        #x = ops.dropout(x,self.proj_drop)
        return x


class LayerScale(nn.Cell):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        ones=ops.Ones(dim, mindspore.float32)
        self.gamma = nn.Parameter(init_values * ones)

    def construct(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Cell):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, tuning_mode=None,i=None):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer([dim])
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer([dim])
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.tuning_mode = tuning_mode
        if self.tuning_mode == 'psrp':
            self.psrp = PSRP(dim)


    def construct(self, x):
        if self.tuning_mode == 'psrp':
            #print("psrp input"+str(x))  
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))

            #print("psrp output"+str(x))   
            weight, bias = self.psrp(x)
            #print("weight"+str(weight))
            #print("bias"+str(bias))

            x = x + self.drop_path2(self.ls2(bias + (weight + 1)*self.mlp(self.norm2(x))))
            #print("psrp output"+str(x))
        else:
            # print(self.drop_path1)
            # print(self.drop_path2)            
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
#            print("atten"+str(x))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            # print("block"+str(x))
            # print(ops.sum(x))
        return x


class ResPostBlock(nn.Cell):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.init_values = init_values

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def construct(self, x):
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class ParallelBlock(nn.Cell):

    def __init__(
            self, dim, num_heads, num_parallel=2, mlp_ratio=4., qkv_bias=False, init_values=None,
            drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(nn.SequentialCell(OrderedDict([
                ('norm', norm_layer(dim)),
                ('attn', Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))
            self.ffns.append(nn.SequentialCell(OrderedDict([
                ('norm', norm_layer(dim)),
                ('mlp', Mlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))

    def _forward(self, x):
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def construct(self, x):
        return self._forward(x)


class PatchEmbed(nn.Cell):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, tuning_mode=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.norm_layer = norm_layer

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,has_bias=True,pad_mode='valid')
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.tuning_mode = tuning_mode



    def construct(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")

        x = self.proj(x) 
        if self.flatten:
            x = x.flatten(start_dim=2).transpose(0,2, 1)  # BCHW -> BNC
        
        x = self.norm(x)
        return x



class VisionTransformer(nn.Cell):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='',
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, tuning_mode='linear_probe', probing_mode='mp'): 
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = nn.LayerNorm
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1 if class_token else 0
        self.grad_checkpointing = False 

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, tuning_mode=tuning_mode)
        num_patches = self.patch_embed.num_patches

        self.cls_token = mindspore.Parameter(ops.zeros((1, 1, embed_dim))) if self.num_tokens > 0 else None
        self.pos_embed = mindspore.Parameter(ops.randn(1, num_patches + self.num_tokens, embed_dim) * .02)
        #self.pos_drop = ops.dropout(p=drop_rate)

        dpr = [x.item() for x in ops.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.tuning_mode = tuning_mode
        tuning_mode_list = [tuning_mode] * depth 

        self.probing_mode = probing_mode
        
        self.blocks = nn.SequentialCell(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, tuning_mode=tuning_mode_list[i],i=i)
            for i in range(depth)])


        self.norm = norm_layer([embed_dim]) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer([embed_dim]) if use_fc_norm else nn.Identity()
        
        if self.probing_mode == 'cls_token' or self.probing_mode == 'gap':
            self.head = Dense(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif self.probing_mode == 'mp':
            self.head = Moment_Probing_ViT(in_dim=self.embed_dim,num_classes=num_classes)
            



  
    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        #ones=ops.Ones()
        #x=ones((32,3,224,224),mindspore.float32)
        #cls=32*3*224*224
        #sequence = mindspore.numpy.arange(0.0001, 0.0001*cls + 0.0001, 0.0001,dtype=mindspore.float32)
        #tensor = sequence.reshape(32,3,224,224)
        #x = mindspore.Tensor(tensor, mindspore.float32)
        x = self.patch_embed(x)
        #print("patchembed"+str(x))
        # print(ops.sum(x))
        if self.cls_token is not None:
            x = ops.cat((self.cls_token.broadcast_to((x.shape[0], -1, -1)), x), axis=1)

        x = x + self.pos_embed
        #print("poseembed"+str(x))
        # print(ops.sum(x))
        # print("clstoken"+str(self.cls_token))
        # print(ops.sum(self.cls_token))
        if self.grad_checkpointing :
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        #print("body"+str(x))
        x = self.norm(x)           
        return x 

    def forward_head(self, x):
        if self.probing_mode == 'mp':
            cls_token = self.fc_norm(x[:, 0])
            return self.head(cls_token, x)
        elif self.probing_mode == 'gap':
            x = x[:, self.num_tokens:].mean(dim=1)
            x = self.fc_norm(x)
            return self.head(x)
        elif self.probing_mode == 'cls_token': 
            x = x[:, 0]
            x = self.fc_norm(x)
            return self.head(x)
        else:
            assert 0, 'please choose from mp, gap, cls_token !'

    def construct(self, x):
        x = self.forward_features(x)
        #print("prehead"+str(x))
        #print(ops.sum(x))
        #ones=ops.Ones()
        #x=ones((32,197,768),mindspore.float32)    
        x = self.forward_head(x)

        #print("head"+str(x))
        #print(ops.sum(x))
        return x 




def vit_base_patch16(args):
    """vit_base_patch16"""
    d_model = 768
    depth = 12
    heads = 12
    mlp_dim = 3072
    dim_head = 768 // 12
    patch_size = 16
    normalized_shape = 768
    image_size = 224
    num_classes = args.num_classes
    tuning_mode=args.tuning_mode

    model = VisionTransformer(num_classes =num_classes,tuning_mode=tuning_mode)
    return model


if __name__ =='__main__':
    x = torch.randn(size=(2, 3, 224, 224))
    model = vit_base_patch16_224(probing_mode='mp')  
    y = model(x)
    print(y.shape)  
