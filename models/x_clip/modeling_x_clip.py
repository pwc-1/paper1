# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================
"""MindSpore X-CLIP model."""

from copy import copy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import Normal

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ....utils import logging
from ...modeling_outputs import ModelOutput
from .configuration_x_clip import XCLIPConfig, XCLIPTextConfig, XCLIPVisionConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "microsoft/xclip-base-patch32"


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: ms.Tensor) -> ms.Tensor:
    return ops.cross_entropy(logits, ops.arange(len(logits)))


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->x_clip
def x_clip_loss(similarity: ms.Tensor) -> ms.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
class XCLIPOutput(ModelOutput):
    """
    Args:
        loss (`ms.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for video-text similarity.
        logits_per_video (`ms.Tensor` of shape `(video_batch_size, text_batch_size)`):
            The scaled dot product scores between `video_embeds` and `text_embeds`. This represents the video-text
            similarity scores.
        logits_per_text (`ms.Tensor` of shape `(text_batch_size, video_batch_size)`):
            The scaled dot product scores between `text_embeds` and `video_embeds`. This represents the text-video
            similarity scores.
        text_embeds(`ms.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`XCLIPTextModel`].
        video_embeds(`ms.Tensor` of shape `(batch_size, output_dim`):
            The video embeddings obtained by applying the projection layer to the pooled output of
            [`XCLIPVisionModel`].
        text_model_output (`BaseModelOutputWithPooling`):
            The output of the [`XCLIPTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`XCLIPVisionModel`].
        mit_output (`BaseModelOutputWithPooling`):
            The output of `XCLIPMultiframeIntegrationTransformer` (MIT for short).
    """

    loss: Optional[ms.Tensor] = None
    logits_per_video: ms.Tensor = None
    logits_per_text: ms.Tensor = None
    text_embeds: ms.Tensor = None
    video_embeds: ms.Tensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None
    mit_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["text_model_output", "vision_model_output", "mit_output"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# Copied from transformers.models.clip.modeling_clip.CLIPVisionEmbeddings with CLIP->XCLIP
class XCLIPVisionEmbeddings(nn.Cell):
    def __init__(self, config: XCLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = ms.Parameter(ops.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            has_bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(
            self.num_positions, self.embed_dim)
        self.position_ids = ops.arange(
            self.num_positions).broadcast_to((1, -1))

    def construct(self, pixel_values: ms.Tensor) -> ms.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(
            dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(start_dim=2).swapaxes(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = ops.cat([class_embeds, patch_embeds], axis=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPTextEmbeddings with CLIP->XCLIP
class XCLIPTextEmbeddings(nn.Cell):
    def __init__(self, config: XCLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, embed_dim)

        self.position_ids = ops.arange(
            config.max_position_embeddings).expand((1, -1))

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPAttention with CLIP->XCLIP
class XCLIPAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: ms.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        causal_attention_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(
            query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.shape[1]
        attn_weights = ops.bmm(query_states, key_states.swapaxes(1, 2))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.shape}"
                )
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        attn_weights = ops.softmax(attn_weights, axis=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following #zyf
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = ops.dropout(
            attn_weights, p=self.dropout, training=self.training)

        attn_output = ops.bmm(attn_probs, value_states)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(
            bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->XCLIP
class XCLIPMLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Dense(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Dense(config.intermediate_size, config.hidden_size)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->XCLIP
class XCLIPEncoderLayer(nn.Cell):
    def __init__(self, config: XCLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = XCLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(
            self.embed_dim, epsilon=config.layer_norm_eps)
        self.mlp = XCLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(
            self.embed_dim, epsilon=config.layer_norm_eps)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: ms.Tensor,
        causal_attention_mask: ms.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor]:
        """
        Args:
            hidden_states (`ms.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`ms.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: ms.Tensor, drop_prob: float = 0.0, training: bool = False) -> ms.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = keep_prob + ops.rand(shape, dtype=input.dtype)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->XCLIP
class XCLIPDropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class XCLIPVisionEncoderLayer(nn.Cell):
    """
    This corresponds to the `CrossFramelAttentionBlock` class in the original implementation.
    """

    def __init__(self, config: XCLIPConfig):
        super().__init__()
        self.num_frames = config.num_frames
        self.embed_dim = config.hidden_size

        self.message_fc = nn.Dense(self.embed_dim, self.embed_dim)
        self.message_ln = nn.LayerNorm(
            self.embed_dim, epsilon=config.layer_norm_eps)
        self.message_attn = XCLIPAttention(config)

        self.drop_path = XCLIPDropPath(
            config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        self.self_attn = XCLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(
            self.embed_dim, epsilon=config.layer_norm_eps)
        self.mlp = XCLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(
            self.embed_dim, epsilon=config.layer_norm_eps)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: ms.Tensor,
        causal_attention_mask: ms.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ms.Tensor]:
        """
        Args:
            hidden_states (`ms.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`ms.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            causal_attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        batch_time, seq_length, hidden_size = hidden_states.shape
        batch_size = batch_time // self.num_frames
        msg_token = self.message_fc(hidden_states[:, 0, :])
        msg_token = msg_token.view(batch_size, self.num_frames, hidden_size)

        msg_token = msg_token + \
            self.drop_path(self.message_attn(self.message_ln(msg_token))[0])
        # add dummy sequence dimension
        msg_token = msg_token.view(-1, 1, hidden_size)

        hidden_states = ops.cat([hidden_states, msg_token], axis=1)

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        hidden_states = hidden_states[:, :seq_length, :]

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class XCLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = XCLIPConfig
    base_model_prefix = "x_clip"
    supports_gradient_checkpointing = False

    def _init_weights(self, cell):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(cell, XCLIPTextEmbeddings):
            # cell.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            # cell.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)

            cell.token_embedding.weight.data.initialize(Normal(factor * 0.02))
            cell.position_embedding.weight.data.initialize(
                Normal(factor * 0.02))
        elif isinstance(cell, XCLIPVisionEmbeddings):
            factor = self.config.initializer_factor

            cell.class_embedding.initialize(
                Normal(cell.embed_dim**-0.5 * factor))
            cell.patch_embedding.weight.data.initialize(
                Normal(cell.config.initializer_range * factor))
            cell.position_embedding.weight.data.initialize(
                Normal(cell.config.initializer_range * factor))

            # nn.init.normal_(cell.class_embedding, mean=0.0, std=cell.embed_dim**-0.5 * factor)
            # nn.init.normal_(cell.patch_embedding.weight, std=cell.config.initializer_range * factor)
            # nn.init.normal_(cell.position_embedding.weight, std=cell.config.initializer_range * factor)
        elif isinstance(cell, XCLIPAttention):
            factor = self.config.initializer_factor
            in_proj_std = (cell.embed_dim**-0.5) * \
                ((2 * cell.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (cell.embed_dim**-0.5) * factor

            cell.q_proj.weight.data.initialize(Normal(in_proj_std))
            cell.k_proj.weight.data.initialize(Normal(in_proj_std))
            cell.v_proj.weight.data.initialize(Normal(in_proj_std))
            cell.out_proj.weight.data.initialize(Normal(out_proj_std))

            # nn.init.normal_(cell.q_proj.weight, std=in_proj_std)
            # nn.init.normal_(cell.k_proj.weight, std=in_proj_std)
            # nn.init.normal_(cell.v_proj.weight, std=in_proj_std)
            # nn.init.normal_(cell.out_proj.weight, std=out_proj_std)
        elif isinstance(cell, XCLIPMLP):
            factor = self.config.initializer_factor
            in_proj_std = (cell.config.hidden_size**-0.5) * \
                ((2 * cell.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * cell.config.hidden_size) ** -0.5 * factor

            cell.fc1.weight.data.initialize(Normal(fc_std))
            cell.fc2.weight.data.initialize(Normal(in_proj_std))

        elif isinstance(cell, XCLIPModel):
            factor = self.config.initializer_factor

            cell.text_projection.weight.data.initialize(
                Normal(cell.text_embed_dim**-0.5 * factor))
            cell.visual_projection.weight.data.initialize(
                Normal(cell.vision_embed_dim**-0.5 * factor))
            cell.prompts_visual_projection.initialize(
                Normal(cell.vision_embed_dim**-0.5 * factor))

        elif isinstance(cell, XCLIPMultiframeIntegrationTransformer):
            cell.position_embedding.initialize(
                Normal(self.config.initializer_factor))

        if isinstance(cell, nn.LayerNorm):
            cell.bias.initialize('zeros')
            cell.weight.data.fill(1.0)
        if isinstance(cell, nn.Dense):
            cell.weight.data.initialize(Normal(self.config.initializer_factor))
            if cell.bias is not None:
                cell.bias.initialize('zeros')


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->XCLIP
class XCLIPEncoder(nn.Cell):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`XCLIPEncoderLayer`].

    Args:
        config: XCLIPConfig
    """

    def __init__(self, config: XCLIPConfig):
        super().__init__()
        self.config = config
        self.layers = nn.CellList([XCLIPEncoderLayer(config)
                                  for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        inputs_embeds,
        attention_mask: Optional[ms.Tensor] = None,
        causal_attention_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class XCLIPTextTransformer(nn.Cell):
    def __init__(self, config: XCLIPTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = XCLIPTextEmbeddings(config)
        self.encoder = XCLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(
            embed_dim, epsilon=config.layer_norm_eps)

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""

        Returns:
            `Union[Tuple, BaseModelOutputWithPooling]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids)

        # X_CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype
        )
        # expand attention_mask
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(
                attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        pooled_output = last_hidden_state[ops.arange(
            last_hidden_state.shape[0]), input_ids.argmax(axis=-1)]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class XCLIPTextModel(XCLIPPreTrainedModel):
    config_class = XCLIPTextConfig

    def __init__(self, config: XCLIPTextConfig):
        super().__init__(config)
        self.text_model = XCLIPTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""

        Returns:
            `Union[Tuple, BaseModelOutputWithPooling]`

        Example:
            ```python
            >>> from transformers import AutoTokenizer, XCLIPTextModel
            ...
            >>> model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
            >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
            ...
            >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
            ...
            >>> outputs = model(**inputs)
            >>> last_hidden_state = outputs.last_hidden_state
            >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
            ```
        """
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class XCLIPVisionEncoder(nn.Cell):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`XCLIPVisionEncoderLayer`].

    Args:
        config: XCLIPConfig
    """

    def __init__(self, config: XCLIPConfig):
        super().__init__()
        self.config = config
        self.layers = nn.CellList([XCLIPVisionEncoderLayer(
            config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        inputs_embeds,
        attention_mask: Optional[ms.Tensor] = None,
        causal_attention_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class XCLIPVisionTransformer(nn.Cell):
    """
    This corresponds to the `CrossFrameCommunicationTransformer` class in the original implementation.
    """

    def __init__(self, config: XCLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = XCLIPVisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(embed_dim, epsilon=config.layer_norm_eps)
        self.encoder = XCLIPVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            embed_dim, epsilon=config.layer_norm_eps)

    def construct(
        self,
        pixel_values: ms.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""

        Returns:
            `Union[Tuple, BaseModelOutputWithPooling]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class XCLIPVisionModel(XCLIPPreTrainedModel):
    config_class = XCLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: XCLIPVisionConfig):
        super().__init__(config)
        self.vision_model = XCLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        return self.vision_model.embeddings.patch_embedding

    def construct(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
            `Union[Tuple, BaseModelOutputWithPooling]`

        Example:
            ```python
            >>> import av
            >>> import torch
            >>> import numpy as np
            ...
            >>> from transformers import AutoProcessor, XCLIPVisionModel
            >>> from huggingface_hub import hf_hub_download
            ...
            >>> np.random.seed(0)
            ...
            ...
            >>> def read_video_pyav(container, indices):
            ...     '''
            ...     Decode the video with PyAV decoder.
            ...     Args:
            ...         container (`av.container.input.InputContainer`): PyAV container.
            ...         indices (`List[int]`): List of frame indices to decode.
            ...     Returns:
            ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
            ...     '''
            ...     frames = []
            ...     container.seek(0)
            ...     start_index = indices[0]
            ...     end_index = indices[-1]
            ...     for i, frame in enumerate(container.decode(video=0)):
            ...         if i > end_index:
            ...             break
            ...         if i >= start_index and i in indices:
            ...             frames.append(frame)
            ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])
            ...
            ...
            >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
            ...     '''
            ...     Sample a given number of frame indices from the video.
            ...     Args:
            ...         clip_len (`int`): Total number of frames to sample.
            ...         frame_sample_rate (`int`): Sample every n-th frame.
            ...         seg_len (`int`): Maximum allowed index of sample's last frame.
            ...     Returns:
            ...         indices (`List[int]`): List of sampled frame indices
            ...     '''
            ...     converted_len = int(clip_len * frame_sample_rate)
            ...     end_idx = np.random.randint(converted_len, seg_len)
            ...     start_idx = end_idx - converted_len
            ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
            ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
            ...     return indices
            ...
            ...
            >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
            >>> file_path = hf_hub_download(
            ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
            ... )
            >>> container = av.open(file_path)
            ...
            >>> # sample 16 frames
            >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            >>> video = read_video_pyav(container, indices)
            ...
            >>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
            >>> model = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")
            ...
            >>> pixel_values = processor(videos=list(video), return_tensors="pt").pixel_values
            ...
            >>> batch_size, num_frames, num_channels, height, width = pixel_values.shape
            >>> pixel_values = pixel_values.reshape(-1, num_channels, height, width)
            ...
            >>> outputs = model(pixel_values)
            >>> last_hidden_state = outputs.last_hidden_state
            ```
        """
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class XCLIPMultiframeIntegrationTransformer(nn.Cell):
    """
    This corresponds to the `MultiframeIntegrationTransformer` class in the original implementation.
    """

    def __init__(self, config: XCLIPVisionConfig):
        super().__init__()

        self.position_embedding = ms.Parameter(
            ms.numpy.empty((1, config.num_frames, config.hidden_size)))
        self.encoder = XCLIPEncoder(config)

    def construct(
        self,
        hidden_states,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        residual = hidden_states

        # add position embeddings
        hidden_states = hidden_states + self.position_embedding

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]

        last_hidden_state = last_hidden_state.type(
            hidden_states.dtype) + residual

        pooled_output = last_hidden_state.mean(axis=1, keep_dims=False)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class XCLIPCrossAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.prompt_num_attention_heads

        dim = config.projection_dim
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Dense(dim, dim, has_bias=False)
        self.k_proj = nn.Dense(dim, dim, has_bias=False)
        self.v_proj = nn.Dense(dim, dim, has_bias=False)

        self.attn_drop = nn.Dropout(config.prompt_attention_dropout)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(config.prompt_projection_dropout)

    def _shape(self, tensor: ms.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(self, queries, keys, values):
        """Input shape: Batch x Time x Channel"""
        batch_size, query_seq_len, hidden_size = queries.shape
        batch_size, key_seq_len, hidden_size = keys.shape
        queries = (
            self.q_proj(queries)
            .reshape(batch_size, query_seq_len, self.num_heads, hidden_size // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(keys)
            .reshape(batch_size, key_seq_len, self.num_heads, hidden_size // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        values = (
            self.v_proj(values)
            .reshape(batch_size, key_seq_len, self.num_heads, hidden_size // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (queries @ keys.swapaxes(-2, -1)) * self.scale
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ values).swapaxes(1, 2).reshape(batch_size,
                                                   query_seq_len, hidden_size)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PromptGeneratorLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()

        embed_dim = config.projection_dim
        self.cross_attn = XCLIPCrossAttention(config)
        self.norm1 = nn.LayerNorm(
            embed_dim, epsilon=config.text_config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(
            embed_dim, epsilon=config.text_config.layer_norm_eps)
        self.mlp = nn.SequentialCell([
            nn.Dense(embed_dim, embed_dim * 4),
            ACT2FN[config.prompt_hidden_act],
            nn.Dropout(config.prompt_attention_dropout),
            nn.Dense(embed_dim * 4, embed_dim),
        ])

    def construct(self, x, visual):
        x = x + self.cross_attn(self.norm1(x), visual, visual)
        x = x + self.mlp(self.norm3(x))
        return x


class XCLIPPromptGenerator(nn.Cell):
    """This corresponds to the `VideoSpecificPrompt` class in the original implementation."""

    def __init__(self, config):
        super().__init__()
        embed_dim = config.projection_dim
        self.layernorm = nn.LayerNorm(
            embed_dim, epsilon=config.vision_config.layer_norm_eps)
        self.decoder = nn.CellList([PromptGeneratorLayer(
            config) for _ in range(config.prompt_layers)])
        self.alpha = ms.Parameter(ops.ones(embed_dim) * config.prompt_alpha)

    def construct(self, text, visual):
        visual = self.layernorm(visual)
        for layer in self.decoder:
            text = layer(text, visual)

        return self.alpha * text


class XCLIPModel(XCLIPPreTrainedModel):
    config_class = XCLIPConfig

    def __init__(self, config: XCLIPConfig):
        super().__init__(config)

        if not isinstance(config.text_config, XCLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type XCLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, XCLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type XCLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = XCLIPTextTransformer(text_config)
        self.vision_model = XCLIPVisionTransformer(vision_config)

        self.visual_projection = nn.Dense(
            self.vision_embed_dim, self.projection_dim, has_bias=False)
        self.text_projection = nn.Dense(
            self.text_embed_dim, self.projection_dim, has_bias=False)
        self.logit_scale = ms.Parameter(
            ms.Tensor(self.config.logit_scale_init_value))

        self.prompts_visual_layernorm = nn.LayerNorm(
            self.vision_embed_dim, epsilon=config.vision_config.layer_norm_eps)
        self.prompts_visual_projection = ms.Parameter(
            ops.randn(self.vision_embed_dim, self.projection_dim))

        mit_config = copy(vision_config)
        mit_config.hidden_size = vision_config.mit_hidden_size
        mit_config.intermediate_size = vision_config.mit_intermediate_size
        mit_config.num_hidden_layers = vision_config.mit_num_hidden_layers
        mit_config.num_attention_heads = vision_config.mit_num_attention_heads
        self.mit = XCLIPMultiframeIntegrationTransformer(mit_config)

        self.prompts_generator = XCLIPPromptGenerator(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_text_features(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ms.Tensor:
        r"""

        Returns:
            text_features (`ms.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`XCLIPTextModel`].

        Example:
            ```python
            >>> from transformers import AutoTokenizer, AutoModel
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
            >>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
            ...
            >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
            >>> text_features = model.get_text_features(**inputs)
            ```
        """
        # Use X_CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        return text_embeds

    def get_video_features(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ms.Tensor:
        r"""

        Returns:
            video_features (`ms.Tensor` of shape `(batch_size, output_dim`): The video embeddings obtained by
                applying the projection layer to the pooled output of [`XCLIPVisionModel`] and
                [`XCLIPMultiframeIntegrationTransformer`].

        Example:
            ```python
            >>> import av
            >>> import torch
            >>> import numpy as np
            ...
            >>> from transformers import AutoProcessor, AutoModel
            >>> from huggingface_hub import hf_hub_download
            ...
            >>> np.random.seed(0)
            ...
            ...
            >>> def read_video_pyav(container, indices):
            ...     '''
            ...     Decode the video with PyAV decoder.
            ...     Args:
            ...         container (`av.container.input.InputContainer`): PyAV container.
            ...         indices (`List[int]`): List of frame indices to decode.
            ...     Returns:
            ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
            ...     '''
            ...     frames = []
            ...     container.seek(0)
            ...     start_index = indices[0]
            ...     end_index = indices[-1]
            ...     for i, frame in enumerate(container.decode(video=0)):
            ...         if i > end_index:
            ...             break
            ...         if i >= start_index and i in indices:
            ...             frames.append(frame)
            ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])
            ...
            ...
            >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
            ...     '''
            ...     Sample a given number of frame indices from the video.
            ...     Args:
            ...         clip_len (`int`): Total number of frames to sample.
            ...         frame_sample_rate (`int`): Sample every n-th frame.
            ...         seg_len (`int`): Maximum allowed index of sample's last frame.
            ...     Returns:
            ...         indices (`List[int]`): List of sampled frame indices
            ...     '''
            ...     converted_len = int(clip_len * frame_sample_rate)
            ...     end_idx = np.random.randint(converted_len, seg_len)
            ...     start_idx = end_idx - converted_len
            ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
            ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
            ...     return indices
            ...
            ...
            >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
            >>> file_path = hf_hub_download(
            ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
            ... )
            >>> container = av.open(file_path)
            ...
            >>> # sample 8 frames
            >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            >>> video = read_video_pyav(container, indices)
            ...
            >>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
            >>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
            ...
            >>> inputs = processor(videos=list(video), return_tensors="pt")
            ...
            >>> video_features = model.get_video_features(**inputs)
            ```
        """
        # Use X_CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        video_embeds = vision_outputs[1]
        video_embeds = self.visual_projection(video_embeds)

        cls_features = video_embeds.view(batch_size, num_frames, -1)

        mit_outputs = self.mit(
            cls_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        video_embeds = mit_outputs[1]

        return video_embeds

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        pixel_values: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, XCLIPOutput]:
        r"""

        Returns:
            `Union[Tuple, XCLIPOutput]`

        Example:
            ```python
            >>> import av
            >>> import torch
            >>> import numpy as np
            ...
            >>> from transformers import AutoProcessor, AutoModel
            >>> from huggingface_hub import hf_hub_download
            ...
            >>> np.random.seed(0)
            ...
            ...
            >>> def read_video_pyav(container, indices):
            ...     '''
            ...     Decode the video with PyAV decoder.
            ...     Args:
            ...         container (`av.container.input.InputContainer`): PyAV container.
            ...         indices (`List[int]`): List of frame indices to decode.
            ...     Returns:
            ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
            ...     '''
            ...     frames = []
            ...     container.seek(0)
            ...     start_index = indices[0]
            ...     end_index = indices[-1]
            ...     for i, frame in enumerate(container.decode(video=0)):
            ...         if i > end_index:
            ...             break
            ...         if i >= start_index and i in indices:
            ...             frames.append(frame)
            ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])
            ...
            ...
            >>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
            ...     '''
            ...     Sample a given number of frame indices from the video.
            ...     Args:
            ...         clip_len (`int`): Total number of frames to sample.
            ...         frame_sample_rate (`int`): Sample every n-th frame.
            ...         seg_len (`int`): Maximum allowed index of sample's last frame.
            ...     Returns:
            ...         indices (`List[int]`): List of sampled frame indices
            ...     '''
            ...     converted_len = int(clip_len * frame_sample_rate)
            ...     end_idx = np.random.randint(converted_len, seg_len)
            ...     start_idx = end_idx - converted_len
            ...     indices = np.linspace(start_idx, end_idx, num=clip_len)
            ...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
            ...     return indices
            ...
            ...
            >>> # video clip consists of 300 frames (10 seconds at 30 FPS)
            >>> file_path = hf_hub_download(
            ...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
            ... )
            >>> container = av.open(file_path)
            ...
            >>> # sample 8 frames
            >>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            >>> video = read_video_pyav(container, indices)
            ...
            >>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
            >>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
            ...
            >>> inputs = processor(
            ...     text=["playing sports", "eating spaghetti", "go shopping"],
            ...     videos=list(video),
            ...     return_tensors="pt",
            ...     padding=True,
            ... )
            ...
            >>> # forward pass
            >>> with torch.no_grad():
            ...     outputs = model(**inputs)
            ...
            >>> logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
            >>> probs = logits_per_video.softmax(dim=1)  # we can take the softmax to get the label probabilities
            >>> print(probs)
            tensor([[1.9496e-04, 9.9960e-01, 2.0825e-04]])
            ```
        """
        # Use X_CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        video_embeds = vision_outputs[1]
        video_embeds = self.visual_projection(video_embeds)

        cls_features = video_embeds.view(batch_size, num_frames, -1)

        mit_outputs = self.mit(
            cls_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        video_embeds = mit_outputs[1]

        img_features = vision_outputs[0][:, 1:, :]
        img_features = self.prompts_visual_layernorm(img_features)
        img_features = img_features @ self.prompts_visual_projection
        img_features = img_features.view(
            batch_size, num_frames, -1, video_embeds.shape[-1])
        img_features = img_features.mean(axis=1, keep_dims=False)

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        text_embeds = text_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        text_embeds = text_embeds + \
            self.prompts_generator(text_embeds, img_features)

        # normalized features
        video_embeds = video_embeds / \
            video_embeds.norm(ord=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / \
            text_embeds.norm(ord=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_video = ops.einsum(
            "bd,bkd->bk", video_embeds, logit_scale * text_embeds)
        logits_per_text = logits_per_video.T

        loss = None
        if return_loss:
            loss = x_clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_video, logits_per_text, text_embeds,
                      video_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return XCLIPOutput(
            loss=loss,
            logits_per_video=logits_per_video,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            video_embeds=video_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
            mit_output=mit_outputs,
        )


__all__ = [
    "XCLIPModel",
    "XCLIPTextModel",
    "XCLIPVisionModel",
]
