# coding=utf-8
# Copyright 2022 UW-Madison The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Nystromformer model."""

import math
from typing import Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import ops
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.common.initializer import initializer, Normal
# from mindspore.ops import operations as P
# from mindspore.ops import functional as F
from mindnlp.utils import logging

from .configuration_nystromformer import NystromformerConfig

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer



logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "uw-madison/nystromformer-512"
_CONFIG_FOR_DOC = "NystromformerConfig"


class NystromformerEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings + 2, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids=ops.arange(config.max_position_embeddings).expand((1, -1)) + 2
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.token_type_ids=ops.zeros(self.position_ids.shape, dtype=mstype.int64)

    def construct(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=mstype.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NystromformerSelfAttention(nn.Cell):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.num_landmarks = config.num_landmarks
        self.seq_len = config.segment_means_seq_len
        self.conv_kernel_size = config.conv_kernel_size

        if config.inv_coeff_init_option:
            self.init_option = config["inv_init_coeff_option"]
        else:
            self.init_option = "original"

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )

        if self.conv_kernel_size is not None:
            self.conv = nn.Conv2d(
                in_channels=self.num_attention_heads,
                out_channels=self.num_attention_heads,
                kernel_size=(self.conv_kernel_size, 1),
                # padding=(self.conv_kernel_size // 2, 0),
                padding=(self.conv_kernel_size//2, self.conv_kernel_size//2, 0, 0),
                has_bias=False,
                group=self.num_attention_heads,
                pad_mode="pad"
            )

    # Function to approximate Moore-Penrose inverse via the iterative method
    def iterative_inv(self, mat, n_iter=6):
        identity = ops.eye(mat.shape[-1])
        key = mat

        # The entries of key are positive and ||key||_{\infty} = 1 due to softmax
        if self.init_option == "original":
            # This original implementation is more conservative to compute coefficient of Z_0.
            value = 1 / ops.max(ops.sum(key, dim=-2)) * key.transpose(-1, -2)
        else:
            # This is the exact coefficient computation, 1 / ||key||_1, of initialization of Z_0, leading to faster convergence.
            value = 1 / ops.max(ops.sum(key, dim=-2), axis=-1).values[:, :, None, None] * key.transpose(-1, -2)

        for _ in range(n_iter):
            key_value = ops.matmul(key, value)
            value = ops.matmul(
                0.25 * value,
                13 * identity
                - ops.matmul(key_value, 15 * identity - ops.matmul(key_value, 7 * identity - key_value)),
            )
        return value

    def transpose_for_scores(self, layer):
        new_layer_shape = layer.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        layer = layer.view(*new_layer_shape)
        return layer.permute(0, 2, 1, 3)

    def construct(self, hidden_states, attention_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        query_layer = query_layer / math.sqrt(math.sqrt(self.attention_head_size))
        key_layer = key_layer / math.sqrt(math.sqrt(self.attention_head_size))

        if self.num_landmarks == self.seq_len:
            attention_scores = ops.matmul(query_layer, key_layer.transpose(0, 1, 3, 2))

            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in NystromformerModel forward() function)
                attention_scores = attention_scores + attention_mask

            attention_probs = ops.softmax(attention_scores, axis=-1)
            context_layer = ops.matmul(attention_probs, value_layer)

        else:
            q_landmarks = query_layer.reshape(
                -1,
                self.num_attention_heads,
                self.num_landmarks,
                self.seq_len // self.num_landmarks,
                self.attention_head_size,
            ).mean(axis=-2)
            k_landmarks = key_layer.reshape(
                -1,
                self.num_attention_heads,
                self.num_landmarks,
                self.seq_len // self.num_landmarks,
                self.attention_head_size,
            ).mean(axis=-2)

            kernel_1 = ops.softmax(ops.matmul(query_layer, k_landmarks.transpose(-1, -2)), axis=-1)
            kernel_2 = ops.softmax(ops.matmul(q_landmarks, k_landmarks.transpose(-1, -2)), axis=-1)

            attention_scores = ops.matmul(q_landmarks, key_layer.transpose(-1, -2))

            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in NystromformerModel forward() function)
                attention_scores = attention_scores + attention_mask

            kernel_3 = ops.softmax(attention_scores, axis=-1)
            attention_probs = ops.matmul(kernel_1, self.iterative_inv(kernel_2))
            new_value_layer = ops.matmul(kernel_3, value_layer)
            context_layer = ops.matmul(attention_probs, new_value_layer)

        if self.conv_kernel_size is not None:
            context_layer += self.conv(value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class NystromformerSelfOutput(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class NystromformerAttention(nn.Cell):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = NystromformerSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = NystromformerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, axis=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(self, hidden_states, attention_mask=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Nystromformer
class NystromformerIntermediate(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Nystromformer
class NystromformerOutput(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class NystromformerLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = NystromformerAttention(config)
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = NystromformerIntermediate(config)
        self.output = NystromformerOutput(config)

    def construct(self, hidden_states, attention_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class NystromformerEncoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.CellList([NystromformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->Nystromformer
class NystromformerPredictionHeadTransform(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->Nystromformer
class NystromformerLMPredictionHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.transform = NystromformerPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        self.bias = mindspore.Parameter(ops.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def construct(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->Nystromformer
class NystromformerOnlyMLMHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.predictions = NystromformerLMPredictionHead(config)

    def construct(self, sequence_output: mindspore.Tensor) -> mindspore.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class NystromformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = NystromformerConfig
    base_model_prefix = "nystromformer"
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, (nn.Dense, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                             cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0
            cell.weight.set_data(Tensor(weight, cell.weight.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


NYSTROMFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Cell](https://pytorch.org/docs/stable/nn.html#torch.nn.Cell) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`NystromformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

NYSTROMFORMER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`mindspore.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`mindspore.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            
            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`mindspore.Tensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
            
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            
            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`mindspore.Tensor` of shape `({0})`, *optional*):
        
            - Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            - [What are position IDs?](../glossary#position-ids)
        head_mask (`mindspore.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`mindspore.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class NystromformerModel(NystromformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = NystromformerEmbeddings(config)
        self.encoder = NystromformerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = ops.ones(((batch_size, seq_length)))

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=mstype.int64)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: mindspore.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class NystromformerForMaskedLM(NystromformerPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder"]

    def __init__(self, config):
        super().__init__(config)

        self.nystromformer = NystromformerModel(config)
        self.cls = NystromformerOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], MaskedLMOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.nystromformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NystromformerClassificationHead(nn.Cell):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.out_proj = nn.Dense(config.hidden_size, config.num_labels)

        self.config = config

    def construct(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NystromformerForSequenceClassification(NystromformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.nystromformer = NystromformerModel(config)
        self.classifier = NystromformerClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], SequenceClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.nystromformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype in (mstype.int64, mstype.int32)):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = ops.mse(logits.squeeze(), labels.squeeze())
                else:
                    loss = ops.mse(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = mindspore.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NystromformerForMultipleChoice(NystromformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.nystromformer = NystromformerModel(config)
        self.pre_classifier = nn.Dense(config.hidden_size, config.hidden_size)
        self.classifier = nn.Dense(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], MultipleChoiceModelOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.shape[-1]) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            if inputs_embeds is not None
            else None
        )

        outputs = self.nystromformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = outputs[0]  # (bs * num_choices, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss = ops.cross_entropy(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NystromformerForTokenClassification(NystromformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.nystromformer = NystromformerModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], TokenClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.nystromformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NystromformerForQuestionAnswering(NystromformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.nystromformer = NystromformerModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], QuestionAnsweringModelOutput]:
        r"""
        Args:
            start_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.nystromformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            start_loss = ops.cross_entropy(start_logits, start_positions,ignore_index=ignored_index)
            end_loss = ops.cross_entropy(end_logits, end_positions,ignore_index=ignored_index)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__=["NystromformerEmbeddings",
         "NystromformerSelfAttention",
         "NystromformerSelfOutput",
         "NystromformerAttention",
         "NystromformerIntermediate",
         "NystromformerOutput",
         "NystromformerLayer",
         "NystromformerEncoder",
         "NystromformerPredictionHeadTransform",
         "NystromformerLMPredictionHead",
         "NystromformerOnlyMLMHead",
         "NystromformerPreTrainedModel",
         "NystromformerModel",
         "NystromformerForMaskedLM",
         "NystromformerClassificationHead",
         "NystromformerForSequenceClassification",
         "NystromformerForMultipleChoice",
         "NystromformerForTokenClassification",
         "NystromformerForQuestionAnswering"]
