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
# ============================================================================
"""Mindspore SqueezeBert model."""

import math
from typing import Optional, Tuple, Union

import mindspore
import numpy as np
from mindspore import nn, ops
from mindspore.common.initializer import Normal, initializer

from mindnlp.utils import logging

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_squeezebert import SqueezeBertConfig

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "squeezebert/squeezebert-uncased"
_CONFIG_FOR_DOC = "SqueezeBertConfig"


class SqueezeBertEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.embedding_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(
            [config.hidden_size], epsilon=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = ops.arange(config.max_position_embeddings).broadcast_to(
            (1, -1)
        )

    def construct(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None
    ):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MatMulWrapper(nn.Cell):
    """
    Wrapper for ops.matmul(). This makes flop-counting easier to implement. Note that if you directly call
    ops.matmul() in your code, the flop counter will typically ignore the flops of the matmul.
    """

    def __init__(self):
        super().__init__()

    def construct(self, mat1, mat2):
        """
        Here are the typical dimensions found in BERT (the B is optional) mat1.shape: [B, <optional extra dims>, M, K]
        mat2.shape: [B, <optional extra dims>, K, N] output shape: [B, <optional extra dims>, M, N]

        Args:
            mat1: a tensor
            mat2: a tensor

        Returns:
            matmul of these tensors
        """
        return ops.matmul(mat1, mat2)


class SqueezeBertLayerNorm(nn.LayerNorm):
    """
    This is a nn.LayerNorm subclass that accepts NCW data layout and performs normalization in the C dimension.

    N = batch C = channels W = sequence length
    """

    def __init__(self, hidden_size, epsilon=1e-12):
        nn.LayerNorm.__init__(
            self,
            normalized_shape=hidden_size,
            epsilon=epsilon,
        )  # instantiates self.{weight, bias, eps}
    def construct(self, x):
        x = x.permute(0, 2, 1)
        x = nn.LayerNorm.construct(self, x)
        return x.permute(0, 2, 1)


class ConvDropoutLayerNorm(nn.Cell):
    """
    ConvDropoutLayerNorm: Conv, Dropout, LayerNorm
    """

    def __init__(self, cin, cout, group, dropout_prob):
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=cin, out_channels=cout, kernel_size=1, group=group
        )
        self.layernorm = SqueezeBertLayerNorm(cout)
        self.dropout = nn.Dropout(p=dropout_prob)

    def construct(self, hidden_states, input_tensor):
        x = self.conv1d(hidden_states)
        x = self.dropout(x)
        x = x + input_tensor
        x = self.layernorm(x)
        return x


class ConvActivation(nn.Cell):
    """
    ConvActivation: Conv, Activation
    """

    def __init__(self, cin, cout, group, act):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=cin, out_channels=cout, kernel_size=1, group=group
        )
        self.act = ACT2FN[act]

    def construct(self, x):
        output = self.conv1d(x)
        return self.act(output)


class SqueezeBertSelfAttention(nn.Cell):

    def __init__(self, config, cin, q_groups=1, k_groups=1, v_groups=1):
        """
        config = used for some things; ignored for others (work in progress...) cin = input channels = output channels
        group = number of group to use in conv1d layers
        """
        super().__init__()
        if cin % config.num_attention_heads != 0:
            raise ValueError(
                f"cin ({cin}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(cin / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Conv1d(
            in_channels=cin, out_channels=cin, kernel_size=1, group=q_groups
        )
        self.key = nn.Conv1d(
            in_channels=cin, out_channels=cin, kernel_size=1, group=k_groups
        )
        self.value = nn.Conv1d(
            in_channels=cin, out_channels=cin, kernel_size=1, group=v_groups
        )

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(axis=-1)

        self.matmul_qk = MatMulWrapper()
        self.matmul_qkv = MatMulWrapper()

    def transpose_for_scores(self, x):
        """
        Input/Output:
            - input: [N, C, W]
            - output: [N, C1, W, C2] where C1 is the head index, and C2 is one head's contents
        """
        new_x_shape = (
            x.shape[0],
            self.num_attention_heads,
            self.attention_head_size,
            x.shape[-1],
        )  # [N, C1, C2, W]
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 3, 2)  # [N, C1, C2, W] --> [N, C1, W, C2]

    def transpose_key_for_scores(self, x):
        """
        Input/Output:
            - input: [N, C, W]
            - output: [N, C1, C2, W] where C1 is the head index, and C2 is one head's contents
        """
        new_x_shape = (
            x.shape[0],
            self.num_attention_heads,
            self.attention_head_size,
            x.shape[-1],
        )  # [N, C1, C2, W]
        x = x.view(*new_x_shape)
        # no `permute` needed
        return x

    def transpose_output(self, x):
        """
        Input/Output:
            - input: [N, C1, W, C2]
            - output: [N, C, W]
        """
        x = x.permute(0, 1, 3, 2)  # [N, C1, C2, W]
        new_x_shape = (x.shape[0], self.all_head_size, x.shape[3])  # [N, C, W]
        x = x.view(*new_x_shape)
        return x

    def construct(self, hidden_states, attention_mask, output_attentions):
        """
        expects hidden_states in [N, C, W] data layout.

        The attention_mask data layout is [N, W], and it does not need to be transposed.
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_score = self.matmul_qk(query_layer, key_layer)
        attention_score = attention_score / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel construct() function)
        attention_score = attention_score + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_score)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = self.matmul_qkv(attention_probs, value_layer)
        context_layer = self.transpose_output(context_layer)

        result = {"context_layer": context_layer}
        if output_attentions:
            result["attention_score"] = attention_score
        return result


class SqueezeBertModule(nn.Cell):
    def __init__(self, config):
        """
        Args:
            config:
                containing:

                - hidden_size = input chans = output chans for Q, K, V (they are all the same ... for now) = output
                chans for the Cell.
                - intermediate_size = output chans for intermediate layer
                - group = number of group for all layers in the BertModule. (eventually we could change the interface to
                  allow different group for different layers)
        """
        super().__init__()

        c0 = config.hidden_size
        c1 = config.hidden_size
        c2 = config.intermediate_size
        c3 = config.hidden_size

        self.attention = SqueezeBertSelfAttention(
            config=config,
            cin=c0,
            q_groups=config.q_groups,
            k_groups=config.k_groups,
            v_groups=config.v_groups,
        )
        self.post_attention = ConvDropoutLayerNorm(
            cin=c0,
            cout=c1,
            group=config.post_attention_groups,
            dropout_prob=config.hidden_dropout_prob,
        )
        self.intermediate = ConvActivation(
            cin=c1, cout=c2, group=config.intermediate_groups, act=config.hidden_act
        )
        self.output = ConvDropoutLayerNorm(
            cin=c2,
            cout=c3,
            group=config.output_groups,
            dropout_prob=config.hidden_dropout_prob,
        )

    def construct(self, hidden_states, attention_mask, output_attentions):
        att = self.attention(hidden_states, attention_mask, output_attentions)
        attention_output = att["context_layer"]

        post_attention_output = self.post_attention(attention_output, hidden_states)
        intermediate_output = self.intermediate(post_attention_output)
        layer_output = self.output(intermediate_output, post_attention_output)

        output_dict = {"feature_map": layer_output}
        if output_attentions:
            output_dict["attention_score"] = att["attention_score"]

        return output_dict


class SqueezeBertEncoder(nn.Cell):
    def __init__(self, config):
        super().__init__()

        assert config.embedding_size == config.hidden_size, (
            "If you want embedding_size != intermediate hidden_size, "
            "please insert a Conv1d layer to adjust the number of channels "
            "before the first SqueezeBertModule."
        )

        self.layers = nn.CellList(
            [SqueezeBertModule(config) for _ in range(config.num_hidden_layers)]
        )

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        if head_mask is None:
            head_mask_is_all_none = True
        elif head_mask.count(None) == len(head_mask):
            head_mask_is_all_none = True
        else:
            head_mask_is_all_none = False
        assert (
            head_mask_is_all_none is True
        ), "head_mask is not yet supported in the SqueezeBert implementation."

        # [batch_size, sequence_length, hidden_size] --> [batch_size, hidden_size, sequence_length]
        hidden_states = hidden_states.permute(0, 2, 1)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                hidden_states = hidden_states.permute(0, 2, 1)
                all_hidden_states += (hidden_states,)
                hidden_states = hidden_states.permute(0, 2, 1)

            layer_output = layer.construct(
                hidden_states, attention_mask, output_attentions
            )

            hidden_states = layer_output["feature_map"]

            if output_attentions:
                all_attentions += (layer_output["attention_score"],)

        # [batch_size, hidden_size, sequence_length] --> [batch_size, sequence_length, hidden_size]
        hidden_states = hidden_states.permute(0, 2, 1)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class SqueezeBertPooler(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SqueezeBertPredictionHeadTransform(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(
            [config.hidden_size], epsilon=config.layer_norm_eps
        )

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class SqueezeBertLMPredictionHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.transform = SqueezeBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        self.bias = mindspore.Parameter(ops.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self) -> None:
        self.decoder.bias = self.bias

    def construct(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class SqueezeBertOnlyMLMHead(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.predictions = SqueezeBertLMPredictionHead(config)

    def construct(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class SqueezeBertPreTrainedModel(PreTrainedModel, nn.Cell):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SqueezeBertConfig
    base_model_prefix = "transformer"

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, (nn.Dense, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(
                initializer(
                    Normal(self.config.initializer_range),
                    cell.weight.shape,
                    cell.weight.dtype,
                )
            )
            if cell.bias is not None:
                cell.bias.set_data(
                    initializer("zeros", cell.bias.shape, cell.bias.dtype)
                )
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(
                0.0, self.config.initializer_range, cell.weight.shape
            )

            if cell.padding_idx:
                weight[cell.padding_idx] = 0
            cell.weight.set_data(mindspore.Tensor(weight, dtype=cell.weight.dtype))
        elif isinstance(cell, SqueezeBertLayerNorm):
            cell.weight.set_data(
                initializer("ones", cell.weight.shape, cell.weight.dtype)
            )
            cell.bias.set_data(initializer("zeros", cell.bias.shape, cell.bias.dtype))


class SqueezeBertModel(SqueezeBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = SqueezeBertEmbeddings(config)
        self.encoder = SqueezeBertEncoder(config)
        self.pooler = SqueezeBertPooler(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

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
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = ops.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )
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
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SqueezeBertForMaskedLM(SqueezeBertPreTrainedModel):
    _tied_weights_keys = [
        "cls.predictions.decoder.weight",
        "cls.predictions.decoder.bias",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.transformer = SqueezeBertModel(config)
        self.cls = SqueezeBertOnlyMLMHead(config)

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
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.transformer(
            input_ids=input_ids,
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
            loss_fct = ops.cross_entropy  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SqueezeBertForSequenceClassification(SqueezeBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.transformer = SqueezeBertModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, self.config.num_labels)

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
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.transformer(
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

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in [
                    mindspore.int64,
                    mindspore.int32,
                ]:
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = ops.mse_loss
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = ops.cross_entropy
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = ops.binary_cross_entropy_with_logits
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SqueezeBertForMultipleChoice(SqueezeBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.transformer = SqueezeBertModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
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
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where *num_choices* is the size of the second dimension of the input tensors. (see
                *input_ids* above)
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        num_choices = (
            input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        )

        input_ids = (
            input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        )
        attention_mask = (
            attention_mask.view(-1, attention_mask.shape[-1])
            if attention_mask is not None
            else None
        )
        token_type_ids = (
            token_type_ids.view(-1, token_type_ids.shape[-1])
            if token_type_ids is not None
            else None
        )
        position_ids = (
            position_ids.view(-1, position_ids.shape[-1])
            if position_ids is not None
            else None
        )
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            if inputs_embeds is not None
            else None
        )

        outputs = self.transformer(
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

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = ops.cross_entropy
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SqueezeBertForTokenClassification(SqueezeBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = SqueezeBertModel(config)
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
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.transformer(
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
            loss_fct = ops.cross_entropy
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SqueezeBertForQuestionAnswering(SqueezeBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = SqueezeBertModel(config)
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
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        Args:
            start_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
                are not taken into account for computing the loss.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.transformer(
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

            loss_fct = ops.cross_entropy
            start_loss = loss_fct(
                start_logits, start_positions, ignore_index=ignored_index
            )
            end_loss = loss_fct(end_logits, end_positions, ignore_index=ignored_index)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
__all__ = [
    "SqueezeBertForMaskedLM",
    "SqueezeBertForMultipleChoice",
    "SqueezeBertForQuestionAnswering",
    "SqueezeBertForSequenceClassification",
    "SqueezeBertForTokenClassification",
    "SqueezeBertModel",
    "SqueezeBertModule",
    "SqueezeBertPreTrainedModel",
]
