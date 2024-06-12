from einops import rearrange, reduce, repeat
import numpy as np
from mindspore import Parameter

from models2.transformer_MS import PatchEmbed, TransformerContainer, get_sine_cosine_pos_emb
from models2.weeight_init_MS import (trunc_normal_, init_from_vit_pretrain_,
	init_from_mae_pretrain_, init_from_kinetics_pretrain_)

import math
from functools import partial

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore.common import dtype as mstype
from mindspore import Tensor
import mindspore.numpy as mnp
import mindspore.ops as ops

class TimeSformer(nn.Cell):
    """TimeSformer. A MindSpore impl of `Is Space-Time Attention All You Need for
    Video Understanding? <https://arxiv.org/abs/2102.05095>`_

    Args:
        num_frames (int): Number of frames in the video.
        img_size (int | tuple): Size of input image.
        patch_size (int): Size of one patch.
        pretrained (str | None): Name of pretrained model. Default: None.
        embed_dims (int): Dimensions of embedding. Defaults to 512.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder. Defaults to 8.
        num_transformer_layers (int): Number of transformer layers. Defaults to
            4.
        in_channels (int): Channel num of input features. Defaults to 256.
        dropout_p (float): Probability of dropout layer. Defaults to 0.
        conv_type (str): Type of the convolution in PatchEmbed layer. Defaults to 'Conv2d'.
        attention_type (str): Type of attentions in TransformerCoder. Choices
            are 'divided_space_time', 'space_only' and 'joint_space_time'.
            Defaults to 'divided_space_time'.
        norm_layer (dict): Config for norm layers. Defaults to nn.LayerNorm.
        copy_strategy (str): Copy or Initial to zero towards the new additional layer.
        use_learnable_pos_emb (bool): Whether to use learnable position embeddings.
        return_cls_token (bool): Whether to use cls_token to predict class label.
    """
    supported_attention_types = [
        'divided_space_time', 'space_only', 'joint_space_time',
    ]

    def __init__(self,
                 num_frames,
                 img_size=224,
                 patch_size=16,
                 pretrained=None,
                 embed_dims=512,
                 num_heads=8,
                 num_transformer_layers=4,
                 in_channels=256,
                 conv_type='Conv2d',
                 dropout_p=0.,
                 attention_type='divided_space_time',
                 norm_layer=nn.LayerNorm,
                 copy_strategy='repeat',
                 use_learnable_pos_emb=True,
                 return_cls_token=True,
                 **kwargs):
        super().__init__()
        assert attention_type in self.supported_attention_types, (
            f'Unsupported Attention Type {attention_type}!')

        self.num_frames = num_frames
        self.pretrained = pretrained
        self.embed_dims = embed_dims
        self.num_transformer_layers = num_transformer_layers
        self.attention_type = attention_type
        self.copy_strategy = copy_strategy
        self.conv_type = conv_type
        self.use_learnable_pos_emb = use_learnable_pos_emb
        self.return_cls_token = return_cls_token

        # tokenize & position embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type=conv_type)
        num_patches = self.patch_embed.num_patches

        if self.attention_type == 'divided_space_time':
            # Divided Space Time Attention
            # operator_order = ['time_attn', 'space_attn', 'ffn']
            # operator_order = ['space_attn','time_attn', 'ffn']
            operator_order = ['local_attn','global_attn', 'ffn']
            container = TransformerContainer(
                num_transformer_layers=num_transformer_layers,
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_frames=num_frames,
                norm_layer=norm_layer,
                hidden_channels=embed_dims * 4,
                operator_order=operator_order)

            transformer_layers = container

        else:
            # Sapce Only & Joint Space Time Attention
            operator_order = ['self_attn', 'ffn']
            container = TransformerContainer(
                num_transformer_layers=num_transformer_layers,
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_frames=num_frames,
                norm_layer=norm_layer,
                hidden_channels=embed_dims * 4,
                operator_order=operator_order)

            transformer_layers = container

        self.transformer_layers = transformer_layers
        # self.norm = norm_layer(embed_dims, eps=1e-6)
        self.norm = norm_layer(normalized_shape=[embed_dims,])

        self.cls_token = Parameter(initializer('zeros', [1, 1, embed_dims], mstype.float32))
        # whether to add one cls_token in temporal pos_emb
        self.use_cls_token_temporal = operator_order[-2] == 'time_attn'
        if self.use_cls_token_temporal:
            num_frames = num_frames + 1
        else:
            num_patches = num_patches + 1

        # spatial pos_emb
        if use_learnable_pos_emb:
            self.pos_embed = Parameter(initializer('zeros', [1, num_patches, embed_dims], mstype.float32))
        else:
            self.pos_embed = get_sine_cosine_pos_emb(num_patches, embed_dims)
        self.drop_after_pos = nn.Dropout(1 - dropout_p)

        # temporal pos_emb
        if self.attention_type != 'space_only':
            if use_learnable_pos_emb:
                self.time_embed = Parameter(initializer('zeros', [1, num_frames, embed_dims], mstype.float32))
            else:
                self.time_embed = get_sine_cosine_pos_emb(num_frames, embed_dims)
            self.drop_after_time = nn.Dropout(1 - dropout_p)

        self.init_weights()


    # def init_weights(self):
    #     if self.use_learnable_pos_emb:
    #         # trunc_normal_(self.pos_embed, std=.02)
    #         nn.init.trunc_normal_(self.pos_embed, std=.02)
    #         if self.attention_type != 'space_only':
    #             nn.init.trunc_normal_(self.time_embed, std=.02)
    #     trunc_normal_(self.cls_token, std=.02)
    
    def init_weights(self):
        if self.use_learnable_pos_emb:
            # trunc_normal_(self.pos_embed, std=.02)
            # self.pos_embed.set_data(initializer('truncated_normal', [1, self.pos_embed.shape[1], self.pos_embed.shape[2]], 'float32', 0.0, 0.02))
            self.pos_embed.set_data(initializer('TruncatedNormal', [1, self.pos_embed.shape[1], self.pos_embed.shape[2]], mstype.float32))
            if self.attention_type != 'space_only':
                self.time_embed.set_data(initializer('TruncatedNormal', [1, self.time_embed.shape[1], self.time_embed.shape[2]], mstype.float32))
        self.cls_token.set_data(initializer('TruncatedNormal', [1, 1, self.cls_token.shape[2]], mstype.float32))



    def interpolate_pos_encoding(pos_embed, patch_embed, x, w, h):
        interpolate = ops.Interpolate(mode='bicubic')
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return pos_embed
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // patch_embed.patch_size[0]
        h0 = h // patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = interpolate(
            patch_pos_embed.reshape(1, int(mnp.sqrt(N)), int(mnp.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / mnp.sqrt(N), h0 / mnp.sqrt(N))
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return mnp.concatenate((class_pos_embed.unsqueeze(0), patch_pos_embed), axis=1)


    
    def prepare_tokens(self, x, patch_embed):
        b, t, c, h, w = x.shape
        x = patch_embed(x)

        # Add Position Embedding
        cls_tokens = ops.tile(self.cls_token, (x.shape[0], 1))
        if self.use_cls_token_temporal:
            if self.use_learnable_pos_emb:
                x = x + self.pos_embed
            else:
                x = x + self.pos_embed.astype(x.dtype).detach()
            x = ops.concat((cls_tokens, x), axis=1)
        else:
            x = ops.concat((cls_tokens, x), axis=1)
            if self.use_learnable_pos_emb:
                x = x + self.interpolate_pos_encoding(x, w, h)  # self.pos_embed
            else:
                x = x + self.interpolate_pos_encoding(x, w, h).astype(x.dtype).detach()  # self.pos_embed
        x = self.drop_after_pos(x)

        # Add Time Embedding
        if self.attention_type != 'space_only':
            cls_tokens = x[:b, 0, :].unsqueeze(1)
            if self.use_cls_token_temporal:
                x = ops.reshape(x[:, 1:, :], (b, -1, c))
                cls_tokens = ops.tile(cls_tokens, (1, x.shape[1], 1))
                x = ops.concat((cls_tokens, x), axis=1)
                if self.use_learnable_pos_emb:
                    x = x + self.time_embed
                else:
                    x = x + self.time_embed.astype(x.dtype).detach()
                cls_tokens = x[:b, 0, :].unsqueeze(1)
                x = ops.reshape(x[:, 1:, :], (b, -1, c))
                x = ops.concat((cls_tokens, x), axis=1)
            else:
                x = ops.reshape(x[:, 1:, :], (b, -1, c))
                if self.use_learnable_pos_emb:
                    x = x + self.time_embed
                else:
                    x = x + self.time_embed.astype(x.dtype).detach()
                x = ops.reshape(x, (b, -1, c))
                x = ops.concat((cls_tokens, x), axis=1)
            x = self.drop_after_time(x)

        return x, b



def forward(self, x):
    # print(x.shape)#[32, 16, 256, 7, 7]
    x, b = self.prepare_tokens(x)
    # print(x.shape)#[32, 785, 512]
    # Video transformer forward
    x = x[:,1:,:]
    x = self.transformer_layers(x)
    # print(x.shape)

    if self.attention_type == 'space_only':
        x = P.Reshape()(x, (x.shape[0], -1, x.shape[-1]))
        x = P.ReduceMean(1)(x, 1)

    x = self.norm(x)
    # Return Class Token
    if self.return_cls_token:
        return x[:, 0, :]
    else:
        return P.ReduceMean(1)(x[:, 1:, :], 1)

def get_last_selfattention(self, x):
    x, b = self.prepare_tokens(x)
    x = self.transformer_layers(x, return_attention=True)
    return x


if __name__ == '__main__':
    model = TimeSformer(num_frames=16, img_size=7, patch_size=1,
                        attention_type='divided_space_time',
                        use_learnable_pos_emb=True, return_cls_token=True)
    input_data = Tensor(np.random.rand(2, 16, 256, 7, 7).astype(np.float32))
    output = model(input_data)
    print(output.shape)
