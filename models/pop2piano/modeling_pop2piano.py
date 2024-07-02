# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ============================================================================
""" Mindspore Pop2Piano model."""

import copy
import math
from typing import Optional, Tuple, Union

import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer, Normal

from mindnlp.transformers.generation import GenerationConfig

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ....utils import logging
from ....modules.functional import finfo
from .configuration_pop2piano import Pop2PianoConfig


logger = logging.get_logger(__name__)

POP2PIANO_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "sweetcocoa/pop2piano",
    # See all Pop2Piano models at https://hf-mirror.com/models?filter=pop2piano
]

# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->Pop2Piano
class Pop2PianoLayerNorm(nn.Cell):

    """
    Pop2PianoLayerNorm class represents a layer normalization module in the Pop2Piano style, designed without bias and
    mean subtraction.
    This class inherits from nn.Cell and provides functionality for performing layer normalization on hidden states in
    a neural network.
    The class includes methods for initialization and construction, applying the Pop2Piano style normalization to
    the input hidden states.
    The 'Pop2PianoLayerNorm' class is suitable for use in deep learning models requiring efficient and effective
    normalization techniques.
    """
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the Pop2Piano style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = Parameter(initializer('zeros', (hidden_size,), mindspore.float32), 'weight')
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        """
        Method 'construct' in the class 'Pop2PianoLayerNorm'.
        
        Args:
            self: Represents the instance of the class Pop2PianoLayerNorm. It is used to access attributes and methods
                of the class.

                - Type: Pop2PianoLayerNorm object
                - Purpose: To operate on the instance of the class.
                - Restrictions: None
            
            hidden_states:
                Represents the hidden states input to the method.

                - Type: Tensor
                - Purpose: Input hidden states that need to be normalized.
                - Restrictions: Should be convertible to float32. Expected shape: (batch_size, seq_length, hidden_size).
        
        Returns:
            None: This method does not return a value but updates the hidden_states in-place after normalizing them.
        
        Raises:
            None.
        """
        # Pop2Piano uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(mindspore.float32).pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [mindspore.float16, mindspore.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


ALL_LAYERNORM_LAYERS.append(Pop2PianoLayerNorm)


# Copied from transformers.models.t5.modeling_t5.T5DenseActDense with T5->Pop2Piano,t5->pop2piano
class Pop2PianoDenseActDense(nn.Cell):

    """
    This class represents a Pop2PianoDenseActDense layer, which is used in neural network models.
    It inherits from the nn.Cell class.
    
    The Pop2PianoDenseActDense layer consists of two dense linear transformations (wi and wo),
    an activation function (act), and a dropout layer (dropout). The layer takes a tensor of hidden states as input
    and applies the following operations to the input:

    1. The input tensor is passed through the wi dense linear transformation.
    2. The result is then passed through the activation function specified by the Pop2PianoConfig's dense_act_fn
    attribute.
    3. The output of the activation function is then passed through the dropout layer, which randomly sets elements
    of the tensor to zero with a probability specified by the Pop2PianoConfig's dropout_rate attribute.
    4. If the weight of the wo dense linear transformation is a tensor and the input tensor's dtype is different from
    the weight's dtype, and the weight's dtype is not int8, the input tensor is converted to the same dtype as the
    weight.
    5. The converted input tensor is then passed through the wo dense linear transformation.
    6. The final output of the layer is returned.

    Please note that this class assumes the existence of the Pop2PianoConfig class, which should be passed as an
    argument to the class's constructor.

    Example:
        ```python
        >>> config = Pop2PianoConfig(...)
        >>> layer = Pop2PianoDenseActDense(config)
        >>> hidden_states = ...
        >>> output = layer.construct(hidden_states)
        ```
    """
    def __init__(self, config: Pop2PianoConfig):
        """
        Initializes the Pop2PianoDenseActDense class.

        Args:
            self: The instance of the class.
            config (Pop2PianoConfig): An instance of the Pop2PianoConfig class containing the configuration parameters
                for the model. It specifies the model's dimensions and activation function for the dense layers.

        Returns:
            None.

        Raises:
            TypeError: If the 'config' parameter is not of type Pop2PianoConfig.
            ValueError: If the 'config' parameter does not contain valid configuration parameters.
        """
        super().__init__()
        self.wi = nn.Dense(config.d_model, config.d_ff, has_bias=False)
        self.wo = nn.Dense(config.d_ff, config.d_model, has_bias=False)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def construct(self, hidden_states):
        """
        Constructs the Pop2PianoDenseActDense object.

        Args:
            self: The instance of the Pop2PianoDenseActDense class.
            hidden_states (mindspore.Tensor): The hidden states to be processed.
                It should have a shape of (batch_size, feature_size).

        Returns:
            mindspore.Tensor: The processed hidden states. It has the same shape as the input hidden_states.

        Raises:
            TypeError: If the hidden_states parameter is not of type mindspore.Tensor.
            ValueError: If the shape of the hidden_states parameter is not (batch_size, feature_size).
        """
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, mindspore.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != mindspore.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5->Pop2Piano
class Pop2PianoDenseGatedActDense(nn.Cell):

    """
    This class represents a custom neural network module called Pop2PianoDenseGatedActDense that implements a dense
    gated activation function using Pop2PianoConfig parameters.
    The module consists of dense layers with gated activation functions for neural network computations.
    It inherits from the nn.Cell class and provides methods for initializing and constructing the neural network layers.
    The class contains methods for initializing network parameters and performing forward computations through the
    network layers.
    """
    def __init__(self, config: Pop2PianoConfig):
        """
        Initializes a Pop2PianoDenseGatedActDense instance with the provided configuration.

        Args:
            self (Pop2PianoDenseGatedActDense): The instance of the Pop2PianoDenseGatedActDense class.
            config (Pop2PianoConfig):
                An instance of Pop2PianoConfig containing configuration parameters.

                - This parameter is used to configure the dense layers and activation functions.
                - It specifies the dimensions of the model, feed-forward layers, dropout rate, and activation
                function type.

        Returns:
            None.

        Raises:
            ValueError: If the configuration parameters are invalid or missing.
            TypeError: If the data types of the configuration parameters are incorrect.
            KeyError: If the activation function specified in the configuration is not supported.
        """
        super().__init__()
        self.wi_0 = nn.Dense(config.d_model, config.d_ff, has_bias=False)
        self.wi_1 = nn.Dense(config.d_model, config.d_ff, has_bias=False)
        self.wo = nn.Dense(config.d_ff, config.d_model, has_bias=False)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def construct(self, hidden_states):
        """
        This method 'construct' in the class 'Pop2PianoDenseGatedActDense' constructs hidden states based on the
        provided input hidden states.

        Args:
            self: Instance of the class Pop2PianoDenseGatedActDense. It is used to access the class attributes and
                methods.

            hidden_states: A tensor representing the input hidden states. It is used as the initial input to construct
                the final hidden states. Type: Tensor.

        Returns:
            None: This method does not return any value but updates the hidden_states variable within the method.

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If there are issues with the shapes or values of the tensors being manipulated.
            RuntimeError: If there are runtime issues during the execution of the method.
        """
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, mindspore.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != mindspore.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5LayerFF with T5->Pop2Piano
class Pop2PianoLayerFF(nn.Cell):

    """
    This class represents a feed-forward layer used in the Pop2Piano model. It is inherited from the nn.Cell class.

    Attributes:
        DenseReluDense (Pop2PianoDenseGatedActDense or Pop2PianoDenseActDense): A dense layer with gated activation
            function, if config.is_gated_act is True, otherwise a dense layer with regular activation function.
        layer_norm (Pop2PianoLayerNorm): A layer normalization module.
        dropout (nn.Dropout): A dropout module.

    Methods:
        __init__: Initializes the Pop2PianoLayerFF instance with the provided configuration.
        construct: Constructs the feed-forward layer by applying layer normalization, dense layer, dropout,
            and residual connection.

    """
    def __init__(self, config: Pop2PianoConfig):
        """
        Initializes the Pop2PianoLayerFF class instance with the provided configuration.

        Args:
            self (Pop2PianoLayerFF): The instance of the Pop2PianoLayerFF class.
            config (Pop2PianoConfig): An instance of the Pop2PianoConfig class containing configuration parameters.
                This parameter is required for configuring the behavior of the Pop2PianoLayerFF instance.
                It should be of type Pop2PianoConfig and must not be None.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = Pop2PianoDenseGatedActDense(config)
        else:
            self.DenseReluDense = Pop2PianoDenseActDense(config)

        self.layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def construct(self, hidden_states):
        """
        Constructs the forward pass of the Pop2PianoLayerFF model.

        Args:
            self (Pop2PianoLayerFF): An instance of the Pop2PianoLayerFF class.
            hidden_states (torch.Tensor): The input hidden states. A tensor of shape (batch_size, hidden_size).

        Returns:
            torch.Tensor: The updated hidden states. A tensor of shape (batch_size, hidden_size).

        Raises:
            None.
        """
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5Attention with T5->Pop2Piano,t5->pop2piano
class Pop2PianoAttention(nn.Cell):

    """
    This class represents a self-attention mechanism with optional relative attention bias for the Pop2Piano model.
    It inherits from nn.Cell and provides functionalities for attention computation and head pruning.

    Attributes:
        config: Pop2PianoConfig, the configuration for the attention mechanism
        has_relative_attention_bias: bool, flag indicating whether relative attention bias is enabled
        relative_attention_num_buckets: int, the number of buckets for relative attention
        relative_attention_max_distance: int, the maximum distance for relative attention
        d_model: int, the model dimension
        key_value_proj_dim: int, the dimension of projected key and value
        n_heads: int, the number of attention heads
        dropout: float, dropout rate
        inner_dim: int, the inner dimension for multi-head attention
        q: nn.Dense, query projection layer
        k: nn.Dense, key projection layer
        v: nn.Dense, value projection layer
        o: nn.Dense, output projection layer
        relative_attention_bias: nn.Embedding, embedding layer for relative attention bias
        pruned_heads: set, set of pruned attention heads
        gradient_checkpointing: bool, flag for gradient checkpointing

    Methods:
        prune_heads: Prunes specified attention heads from the model
        _relative_position_bucket: Computes relative position buckets
        compute_bias: Computes binned relative position bias
        construct: Constructs attention mechanism

    Note:
        For detailed information on each method and attribute, refer to the method and attribute documentation in the
        class implementation.
    """
    def __init__(self, config: Pop2PianoConfig, has_relative_attention_bias=False):
        """
        Initializes an instance of the Pop2PianoAttention class.

        Args:
            self: The instance of the Pop2PianoAttention class.
            config (Pop2PianoConfig): An instance of Pop2PianoConfig containing the configuration parameters.
            has_relative_attention_bias (bool): A boolean indicating whether relative attention bias is enabled.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Dense(self.d_model, self.inner_dim, has_bias=False)
        self.k = nn.Dense(self.d_model, self.inner_dim, has_bias=False)
        self.v = nn.Dense(self.d_model, self.inner_dim, has_bias=False)
        self.o = nn.Dense(self.inner_dim, self.d_model, has_bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        """
        This method 'prune_heads' is defined within the class 'Pop2PianoAttention' and is responsible for pruning the
        attention heads based on the provided criteria.

        Args:
            self: Represents the instance of the class 'Pop2PianoAttention'.
                It is used to access the class attributes and methods.

            heads: A list containing the indices of attention heads to be pruned.
                The indices should be within the range of the total number of attention heads.
                If the list is empty, no action will be taken.

        Returns:
            None: However, it modifies the internal state of the
                'Pop2PianoAttention' instance by pruning the attention heads and updating the relevant attributes.

        Raises:
            No specific exceptions are documented to be raised within this method. However, it is important to handle
            potential exceptions related to the internal functions being called within this method,
            such as 'find_pruneable_heads_and_indices' and 'prune_linear_layer'.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, axis=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(mindspore.int64) * num_buckets
            relative_position = ops.abs(relative_position)
        else:
            relative_position = -ops.minimum(relative_position, ops.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            ops.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(mindspore.int64)
        relative_position_if_large = ops.minimum(
            relative_position_if_large, ops.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += ops.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        # if device is None:
        #     device = self.relative_attention_bias.weight.device
        context_position = ops.arange(query_length, dtype=mindspore.int64)[:, None]
        memory_position = ops.arange(key_length, dtype=mindspore.int64)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def construct(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(0, 2, 1, 3)

        def unshape(states):
            """reshape"""
            return states.transpose(0, 2, 1, 3).view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = ops.cat([past_key_value, hidden_states], axis=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = ops.matmul(
            query_states, key_states.transpose(0, 1, 3, 2)
        )  # equivalent of ops.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = ops.zeros(
                    (1, self.n_heads, real_seq_length, key_length), dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.shape[1] :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = ops.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = ops.softmax(scores.float(), axis=-1).astype(
            scores.dtype
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = ops.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(ops.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerSelfAttention with T5->Pop2Piano,t5->pop2piano
class Pop2PianoLayerSelfAttention(nn.Cell):

    """This class represents a self-attention mechanism used in the Pop2PianoLayer model.

    The Pop2PianoLayerSelfAttention class is a subclass of the nn.Cell class in the PyTorch library.
    It is responsible for performing self-attention on the input hidden states.

    Attributes:
        SelfAttention (Pop2PianoAttention): An instance of the Pop2PianoAttention class used for self-attention
            computation.
        layer_norm (Pop2PianoLayerNorm): An instance of the Pop2PianoLayerNorm class used for layer normalization.
        dropout (nn.Dropout): An instance of the Dropout class used for dropout regularization.

    Methods:
        __init__: Constructs a new Pop2PianoLayerSelfAttention object.
        construct: Performs self-attention on the input hidden states.

    """
    def __init__(self, config, has_relative_attention_bias=False):
        """
        Initializes an instance of the Pop2PianoLayerSelfAttention class.

        Args:
            self: The instance of the class.
            config (object): An object containing configuration parameters for the attention layer.
            has_relative_attention_bias (bool, optional):
                Specifies whether the attention layer has relative attention bias. Defaults to False.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.SelfAttention = Pop2PianoAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Constructs the Pop2PianoLayerSelfAttention.

        This method is responsible for constructing the Pop2PianoLayerSelfAttention in the given class.
        It takes in several parameters to perform the construction and returns None.

        Args:
            self (Pop2PianoLayerSelfAttention): An instance of the Pop2PianoLayerSelfAttention class.
            hidden_states (Tensor): The input hidden states.
            attention_mask (Tensor, optional): An optional mask tensor. Default is None.
            position_bias (Tensor, optional): An optional tensor for position bias. Default is None.
            layer_head_mask (Tensor, optional): An optional tensor for layer head mask. Default is None.
            past_key_value (Tuple[Tensor], optional): An optional tuple of past key and value tensors. Default is None.
            use_cache (bool, optional): A flag indicating whether to use cache. Default is False.
            output_attentions (bool, optional): A flag indicating whether to output attentions. Default is False.

        Returns:
            None

        Raises:
            None
        """
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5LayerCrossAttention with T5->Pop2Piano,t5->pop2piano
class Pop2PianoLayerCrossAttention(nn.Cell):

    """
    The Pop2PianoLayerCrossAttention class represents a layer that performs cross-attention within the Pop2Piano model architecture.
    This class inherits from nn.Cell and contains methods for initializing the layer and constructing the cross-attention mechanism.

    Attributes:
        EncDecAttention: Instance of Pop2PianoAttention for performing cross-attention.
        layer_norm: Instance of Pop2PianoLayerNorm for layer normalization.
        dropout: Dropout layer for regularization.

    Methods:
        __init__: Initializes the Pop2PianoLayerCrossAttention with the given configuration.

        construct: Constructs the cross-attention mechanism by applying layer normalization, attention computation,
            and dropout.

    Returns:
        outputs: Tuple containing the layer output and additional attention outputs.

    """
    def __init__(self, config):
        """
        Initialize a Pop2PianoLayerCrossAttention object.

        Args:
            self (Pop2PianoLayerCrossAttention): The instance of the Pop2PianoLayerCrossAttention class.
            config (object):
                Configuration object containing necessary parameters for initialization.

                - Type: object
                - Purpose: Contains configuration settings for the attention layer.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.EncDecAttention = Pop2PianoAttention(config, has_relative_attention_bias=False)
        self.layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def construct(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        """
        Method 'construct' in the class 'Pop2PianoLayerCrossAttention'.

        This method constructs the output of the Pop2PianoLayerCrossAttention layer.

        Args:
            self: The instance of the class.
            hidden_states (tensor): The input hidden states to the layer.
            key_value_states (tensor): The key-value states used in attention computation.
            attention_mask (tensor, optional): Mask to avoid attending to certain positions.
            position_bias (tensor, optional): Bias applied to positions for relative attention.
            layer_head_mask (tensor, optional): Mask applied to the heads in the layer.
            past_key_value (tuple, optional): Tuple containing past key and value tensors.
            use_cache (bool, optional): If True, cache the computed key-value states.
            query_length (int, optional): Length of the query sequence.
            output_attentions (bool, optional): If True, return attention weights.

        Returns:
            tuple: A tuple containing the layer output tensor and additional outputs from attention computation.

        Raises:
            None
        """
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.t5.modeling_t5.T5Block with T5->Pop2Piano,t5->pop2piano
class Pop2PianoBlock(nn.Cell):

    """
    This class represents a block of the Pop2Piano model. It is a subclass of nn.Cell and contains layers for
    self-attention, cross-attention (if applicable), and feed-forward processing.

    Attributes:
        is_decoder (bool): Indicates whether the block is a decoder block or not.
        layer (nn.CellList): List of layers in the block, including self-attention, cross-attention, and
            feed-forward layers.

    Methods:
        __init__: Initializes a new instance of the Pop2PianoBlock class.
        construct: Constructs the block by applying the layers sequentially to the input hidden states.

    """
    def __init__(self, config, has_relative_attention_bias=False):
        """
        Initializes a new instance of the Pop2PianoBlock class.

        Args:
            self: The class instance that the method operates on.
            config: An instance of the configuration class that contains the model configuration.
            has_relative_attention_bias: A boolean value indicating whether the model has relative attention bias.
                Defaults to False.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.CellList()
        self.layer.append(Pop2PianoLayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(Pop2PianoLayerCrossAttention(config))

        self.layer.append(Pop2PianoLayerFF(config))

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Constructs the Pop2PianoBlock.

        This method constructs the Pop2PianoBlock by performing self-attention and cross-attention operations on the
        given input hidden states.

        Args:
            self (Pop2PianoBlock): The instance of the Pop2PianoBlock class.
            hidden_states (Tensor): The input hidden states. It has shape (batch_size, sequence_length, hidden_size).
            attention_mask (Tensor, optional): The attention mask tensor. It has shape (batch_size, sequence_length)
                and each element is either 0 or 1. Defaults to None.
            position_bias (Tensor, optional): The position bias tensor.
                It has shape (batch_size, num_heads, sequence_length, sequence_length). Defaults to None.
            encoder_hidden_states (Tensor, optional): The encoder hidden states tensor.
                It has shape (batch_size, sequence_length, hidden_size). Defaults to None.
            encoder_attention_mask (Tensor, optional): The encoder attention mask tensor.
                It has shape (batch_size, sequence_length) and each element is either 0 or 1. Defaults to None.
            encoder_decoder_position_bias (Tensor, optional): The encoder-decoder position bias tensor.
                It has shape (batch_size, num_heads, sequence_length, sequence_length). Defaults to None.
            layer_head_mask (Tensor, optional): The layer head mask tensor.
                It has shape (num_hidden_layers, num_heads) and each element is either 0 or 1. Defaults to None.
            cross_attn_layer_head_mask (Tensor, optional): The cross-attention layer head mask tensor.
                It has shape (num_hidden_layers, num_heads) and each element is either 0 or 1. Defaults to None.
            past_key_value (Tuple[Tensor], optional): The tuple of past key-value state tensors.
                The tuple contains two tensors for self-attention and four tensors for cross-attention. Defaults to None.
            use_cache (bool, optional): Whether to use cache for the attention outputs. Defaults to False.
            output_attentions (bool, optional): Whether to output attentions. Defaults to False.

        Returns:
            Tuple[Tensor]: The tuple containing the output hidden states tensor and other optional tensors,
                depending on the value of use_cache.

        Raises:
            ValueError: If the length of past_key_value is not equal to the expected number of past states.
            Warning: If past_key_values is passed to the encoder instead of the decoder.
        """
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == mindspore.float16:
            clamp_value = finfo(hidden_states.dtype, 'max') - 1000 if ops.isinf(hidden_states).any() else \
                finfo(hidden_states.dtype, 'max')
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == mindspore.float16:
                clamp_value = finfo(hidden_states.dtype, 'max') - 1000 if ops.isinf(hidden_states).any() else \
                    finfo(hidden_states.dtype, 'max')
                hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == mindspore.float16:
            clamp_value = finfo(hidden_states.dtype, 'max') - 1000 if ops.isinf(hidden_states).any() else \
                    finfo(hidden_states.dtype, 'max')
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class Pop2PianoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Pop2PianoConfig
    base_model_prefix = "transformer"
    is_parallelizable = False
    supports_gradient_checkpointing = True
    _no_split_modules = ["Pop2PianoBlock"]
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, Pop2PianoLayerNorm):
            module.weight.data.set_data(initializer(Normal(factor * 1.0), \
                                                    module.weight.data.shape, module.weight.data.dtype))
        elif isinstance(module, Pop2PianoConcatEmbeddingToMel):
            module.embedding.weight.data.set_data(initializer(Normal(factor * 1.0), \
                                                              module.embedding.weight.data.shape, \
                                                              module.embedding.weight.data.dtype))
        elif isinstance(module, Pop2PianoForConditionalGeneration):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.set_data(initializer(Normal(factor * 1.0), \
                                               module.shared.weight.data.shape, \
                                               module.shared.weight.data.dtype))
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.set_data(initializer(Normal(factor * 1.0), \
                                                    module.lm_head.weight.data.shape, \
                                                    module.lm_head.weight.data.dtype))
        elif isinstance(module, Pop2PianoDenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.set_data(initializer(Normal(factor * ((self.config.d_model) ** -0.5)), \
                                           module.wi.weight.data.shape, \
                                           module.wi.weight.data.dtype))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.set_data(initializer("zero", module.wi.bias.data.shape, \
                                                         module.wi.bias.data.dtype))
            module.wo.weight.data.set_data(initializer(Normal(factor * ((self.config.d_ff) ** -0.5)), \
                                           module.wo.weight.data.shape, \
                                           module.wo.weight.data.dtype))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.set_data(initializer("zero", module.wo.bias.data.shape, \
                                                         module.wo.bias.data.dtype))
        elif isinstance(module, Pop2PianoDenseGatedActDense):
            module.wi_0.weight.data.set_data(initializer(Normal(factor * ((self.config.d_model) ** -0.5)), \
                                             module.wi_0.weight.data.shape, \
                                             module.wi_0.weight.data.dtype))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.set_data(initializer("zero", module.wi_0.bias.data.shape, \
                                                           module.wi_0.bias.data.dtype))
            module.wi_1.weight.data.set_data(initializer(Normal(factor * ((self.config.d_model) ** -0.5)), \
                                             module.wi_1.weight.data.shape, \
                                             module.wi_1.weight.data.dtype))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.set_data(initializer("zero", module.wi_1.bias.data.shape, \
                                               module.wi_1.bias.data.dtype))
            module.wo.weight.data.set_data(initializer(Normal(factor * ((self.config.d_ff) ** -0.5)), \
                                           module.wo.weight.data.shape, \
                                           module.wo.weight.data.dtype))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.set_data(initializer("zero", module.wo.bias.data.shape, \
                                                         module.wo.bias.data.dtype))
        elif isinstance(module, Pop2PianoAttention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.set_data(initializer(Normal(factor * ((d_model * key_value_proj_dim) ** -0.5)), \
                                          module.q.weight.data.shape, \
                                          module.q.weight.data.dtype))
            module.k.weight.data.set_data(initializer(Normal(factor * (d_model**-0.5)), \
                                          module.k.weight.data.shape, \
                                          module.k.weight.data.dtype))
            module.v.weight.data.set_data(initializer(Normal(factor * (d_model**-0.5)), \
                                          module.v.weight.data.shape, \
                                          module.v.weight.data.dtype))
            module.o.weight.data.set_data(initializer(Normal(factor * ((n_heads * key_value_proj_dim) ** -0.5)), \
                                          module.o.weight.data.shape, \
                                          module.o.weight.data.dtype))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.set_data(initializer(Normal(factor * ((d_model) ** -0.5)), \
                                                                    module.relative_attention_bias.weight.data.shape, \
                                                                    module.relative_attention_bias.weight.data.dtype))

    def _shift_right(self, input_ids):
        """
        Shifts the input sequence to the right by one position for decoding in the Pop2PianoPreTrainedModel class.

        Args:
            self (Pop2PianoPreTrainedModel): The instance of the Pop2PianoPreTrainedModel class.
            input_ids (torch.Tensor): The input tensor of shape [batch_size, sequence_length] containing the input IDs
                for each token in the sequence.

        Returns:
            torch.Tensor: The shifted input tensor of the same shape as input_ids, where the first token in
                each sequence is replaced with the decoder_start_token_id, and subsequent tokens are shifted one
                position to the right.

        Raises:
            ValueError: If self.model.config.decoder_start_token_id is not defined or is None.
            ValueError: If self.model.config.pad_token_id is not defined or is None.
        """
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In Pop2Piano it is usually set to the pad_token_id."
            )

        # shift inputs to the right
        # if is_torch_fx_proxy(input_ids):
        #     # Item assignment is not supported natively for proxies.
        #     shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
        #     shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        # else:
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].copy()
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = shifted_input_ids.masked_fill(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class Pop2PianoStack(Pop2PianoPreTrainedModel):

    """
    This class represents a stack of Pop2Piano blocks that can be used for modeling and processing tasks in a
    Pop2Piano-based architecture. The class inherits from Pop2PianoPreTrainedModel and includes methods for initializing
    the model, setting input embeddings, and constructing the model with various input and output options.

    The class includes methods for initializing the model with token embeddings, processing input data, and generating
    model outputs. It also supports features such as caching, attention masks, and output options for hidden states and
    attentions.

    The Pop2PianoStack class is designed to handle multiple layers of Pop2Piano blocks and provides flexibility for
    customizing model behavior and output based on the input configurations.

    For more detailed information on the methods and their parameters, refer to the method docstrings within the
    class implementation.
    """
    # Copied from transformers.models.t5.modeling_t5.T5Stack.__init__ with T5->Pop2Piano,t5->pop2piano
    def __init__(self, config, embed_tokens=None):
        """
        Initializes a Pop2PianoStack instance.

        Args:
            self: The instance of the Pop2PianoStack class.
            config:
                A configuration object containing parameters for the model.

                - Type: Any
                - Purpose: Specifies the configuration settings for the model.
            embed_tokens:
                Tokens used for embedding.

                - Type: Any
                - Purpose: Optional tokens for embedding.
                - Restrictions: Default value is None.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.CellList(
            [Pop2PianoBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    # Copied from transformers.models.t5.modeling_t5.T5Stack.get_input_embeddings
    def get_input_embeddings(self):
        '''
        This method retrieves the input embeddings from the Pop2PianoStack class.

        Args:
            self: Pop2PianoStack instance. The self parameter is the instance of the Pop2PianoStack class.

        Returns:
            embed_tokens: This method returns the embed_tokens attribute of the Pop2PianoStack instance,
                which represents the input embeddings.

        Raises:
            This method does not raise any exceptions.
        '''
        return self.embed_tokens

    # Copied from transformers.models.t5.modeling_t5.T5Stack.set_input_embeddings
    def set_input_embeddings(self, new_embeddings):
        """
        Set the input embeddings for the Pop2PianoStack model.

        Args:
            self (Pop2PianoStack): The instance of the Pop2PianoStack class.
            new_embeddings (object): The new embeddings to be set for input.

        Returns:
            None: This method updates the embed_tokens attribute of the Pop2PianoStack instance.

        Raises:
            None.
        """
        self.embed_tokens = new_embeddings

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        This method constructs the Pop2PianoStack model with the specified input parameters.

        Args:
            self: The instance of the Pop2PianoStack class.
            input_ids (optional): Tensor of shape (batch_size, sequence_length) representing input token IDs.
            attention_mask (optional): Tensor of shape (batch_size, sequence_length) representing attention mask.
            encoder_hidden_states (optional): Tensor representing hidden states from the encoder.
            encoder_attention_mask (optional): Tensor representing the attention mask for encoder_hidden_states.
            inputs_embeds (optional): Tensor representing the input embeddings.
            head_mask (optional): Tensor representing the head mask for self-attention.
            cross_attn_head_mask (optional): Tensor representing the head mask for cross-attention.
            past_key_values (optional): List of past key values for caching.
            use_cache (optional): Boolean indicating whether to use caching.
            output_attentions (optional): Boolean indicating whether to output attentions.
            output_hidden_states (optional): Boolean indicating whether to output hidden states.
            return_dict (optional): Boolean indicating whether to return a dictionary.

        Returns:
            None

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified simultaneously.
            ValueError: If neither input_ids nor inputs_embeds are specified.
            ValueError: If model is not initialized with valid token embeddings.
            ValueError: If `use_cache` is set to True when model is not used as a decoder.
            Warning: If `use_cache=True` is incompatible with gradient checkpointing.

        Note: Detailed implementation logic is provided in the method's code.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        if attention_mask is None:
            attention_mask = ops.ones((batch_size, mask_seq_length))
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = ops.ones(
                (batch_size, encoder_seq_length), dtype=mindspore.int64
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class Pop2PianoConcatEmbeddingToMel(nn.Cell):
    """Embedding Matrix for `composer` tokens."""
    def __init__(self, config):
        """
        Initializes the Pop2PianoConcatEmbeddingToMel class.

        Args:
            self: The instance of the Pop2PianoConcatEmbeddingToMel class.
            config:
                A configuration object containing parameters for the initialization.

                - Type: Config
                - Purpose: Specifies the configuration settings for the embedding layer.
                - Restrictions: Must contain the following attributes:

                    - composer_vocab_size: An integer specifying the vocabulary size for the composer.
                    - d_model: An integer specifying the dimension of the embedding.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size=config.composer_vocab_size, embedding_size=config.d_model)

    def construct(self, feature, index_value, embedding_offset):
        """
        This method constructs inputs_embeds for Pop2PianoConcatEmbeddingToMel model.

        Args:
            self (object): The instance of the class Pop2PianoConcatEmbeddingToMel.
            feature (array): The input feature array to be concatenated with composer_embedding.
            index_value (int): The index value used for embedding lookup.
            embedding_offset (int): The offset value to adjust the index_value for embedding lookup.

        Returns:
            None.

        Raises:
            None
        """
        index_shifted = index_value - embedding_offset
        composer_embedding = self.embedding(index_shifted).unsqueeze(1)
        inputs_embeds = ops.cat([composer_embedding, feature], axis=1)
        return inputs_embeds

class Pop2PianoForConditionalGeneration(Pop2PianoPreTrainedModel):

    """
    The `Pop2PianoForConditionalGeneration` class is a subclass of `Pop2PianoPreTrainedModel` that represents a
    Pop2Piano model for conditional generation. It is specifically designed for generating MIDI token ids based on
    given input features.

    Initialization:
        The class constructor `__init__` takes a `Pop2PianoConfig` object as an argument and initializes the model.
        It sets up the necessary components like the shared embedding layer, encoder, decoder, and LM head.

    Model Components:
        - `shared`: An embedding layer that maps token ids to their corresponding embeddings.
        - `encoder`: The Pop2PianoStack module responsible for encoding the input features.
        - `decoder`: The Pop2PianoStack module responsible for decoding and generating the output sequence.
        - `lm_head`: A linear layer that maps the decoder output to the vocabulary space.

    Getter and Setter Methods:
        - `get_input_embeddings`: Returns the shared embedding layer.
        - `set_input_embeddings`: Sets the shared embedding layer to the provided `new_embeddings`.
        - `set_output_embeddings`: Sets the LM head to the provided `new_embeddings`.
        - `get_output_embeddings`: Returns the LM head.
        - `get_encoder`: Returns the encoder module.
        - `get_decoder`: Returns the decoder module.

    Generation Methods:
        - `get_mel_conditioner_outputs()`: Concatenates mel conditioner tokens to the front of the input features for
        controlling the type of MIDI token generated by the model. It takes the input features, composer name,
        generation config, and attention mask as inputs.
        - `construct()`: Constructs the model for conditional generation. It takes various inputs like input ids,
        attention mask, decoder input ids, etc., and returns the generated MIDI token ids.
        - `generate()`: Generates token ids for MIDI outputs. It takes input features, attention mask, composer name,
        generation config, and additional kwargs as inputs. It returns the generated MIDI token ids.
        - `prepare_inputs_for_generation()`: Prepares the inputs for generation. It takes input ids, past key values,
        attention mask, and various masks as inputs and returns a dictionary of prepared inputs.
        - `prepare_decoder_input_ids_from_labels()`: Prepares the decoder input ids from labels.
        It takes labels as input and returns the shifted right labels.
        - `_reorder_cache()`: Reorders the past key values according to the beam index.

    Please refer to the documentation of the parent class `Pop2PianoPreTrainedModel` for more details on other
    inherited methods and attributes.
    """
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: Pop2PianoConfig):
        """
        Initializes an instance of the Pop2PianoForConditionalGeneration class.

        Args:
            self: The object instance.
            config (Pop2PianoConfig):
                The configuration object used for initializing the model.

                - The 'config' parameter is of type Pop2PianoConfig.
                - This parameter is required to create an instance of the model.
                - It contains various configuration settings for the model.
                - The 'config' parameter is used to set the attributes of the model object.
                - The 'config' parameter should not be None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.config = config
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        self.mel_conditioner = Pop2PianoConcatEmbeddingToMel(config)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.encoder = Pop2PianoStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = Pop2PianoStack(decoder_config, self.shared)

        self.lm_head = nn.Dense(config.d_model, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method, 'get_input_embeddings', is defined within the class 'Pop2PianoForConditionalGeneration' and
        is used to retrieve the input embeddings.

        Args:
            self (object):
                The instance of the class.

                - Purpose: Represents the current instance of the class.
                - Restrictions: Must be an instance of 'Pop2PianoForConditionalGeneration'.

        Returns:
            None.

        Raises:
            None.
        """
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        """
        Set the input embeddings for the Pop2PianoForConditionalGeneration model.

        Args:
            self (Pop2PianoForConditionalGeneration): The instance of the Pop2PianoForConditionalGeneration class.
            new_embeddings (object): The new input embeddings to be set for the model.
                Should be compatible with the model's encoder and decoder.

        Returns:
            None.

        Raises:
            None.
        """
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings of the Pop2PianoForConditionalGeneration model.

        Args:
            self (Pop2PianoForConditionalGeneration): The instance of the Pop2PianoForConditionalGeneration class.
            new_embeddings (object): The new embeddings to be set as the output embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        """
        Method to retrieve the output embeddings from the Pop2PianoForConditionalGeneration class.

        Args:
            self: Pop2PianoForConditionalGeneration object. Represents the instance of the class.

        Returns:
            lm_head: The method returns the output embeddings from the 'lm_head' attribute of the instance.

        Raises:
            None.
        """
        return self.lm_head

    def get_encoder(self):
        """
        Returns the encoder used for Pop2PianoForConditionalGeneration.

        Args:
            self: An instance of the Pop2PianoForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.encoder

    def get_decoder(self):
        """
        Returns the decoder model used for conditional generation in the Pop2PianoForConditionalGeneration class.

        Args:
            self (Pop2PianoForConditionalGeneration): The instance of the Pop2PianoForConditionalGeneration class.
                This parameter is required to access the decoder model.

        Returns:
            None.

        Raises:
            None.
        """
        return self.decoder

    def get_mel_conditioner_outputs(
        self,
        input_features: mindspore.Tensor,
        composer: str,
        generation_config: GenerationConfig,
        attention_mask: mindspore.Tensor = None,
    ):
        """
        This method is used to concatenate mel conditioner tokens at the front of the input_features in order to
        control the type of MIDI token generated by the model.

        Args:
            input_features (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                input features extracted from the feature extractor.
            composer (`str`):
                composer token which determines the type of MIDI tokens to be generated.
            generation_config (`~generation.GenerationConfig`):
                The generation is used to get the composer-feature_token pair.
            attention_mask (``, *optional*):
                For batched generation `input_features` are padded to have the same shape across all examples.
                `attention_mask` helps to determine which areas were padded and which were not.

                - 1 for tokens that are **not padded**,
                - 0 for tokens that are **padded**.
        """
        composer_to_feature_token = generation_config.composer_to_feature_token
        if composer not in composer_to_feature_token.keys():
            raise ValueError(
                f"Please choose a composer from {list(composer_to_feature_token.keys())}. Composer received - {composer}"
            )
        composer_value = composer_to_feature_token[composer]
        composer_value = mindspore.tensor(composer_value)
        composer_value = composer_value.repeat(input_features.shape[0])

        embedding_offset = min(composer_to_feature_token.values())

        input_features = self.mel_conditioner(
            feature=input_features,
            index_value=composer_value,
            embedding_offset=embedding_offset,
        )
        if attention_mask is not None:
            input_features[~attention_mask[:, 0].bool()] = 0.0

            # since self.mel_conditioner adds a new array at the front of inputs_embeds we need to do the same for attention_mask to keep the shapes same
            attention_mask = ops.cat([attention_mask[:, 0].view(-1, 1), attention_mask], axis=1)
            return input_features, attention_mask

        return input_features, None

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        input_features: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], Seq2SeqLMOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
                config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
                labels in `[0, ..., config.vocab_size]`

        Returns:
            Union[Tuple[mindspore.Tensor], Seq2SeqLMOutput]
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None and input_features is not None:
            raise ValueError("Both `inputs_embeds` and `input_features` received! Please provide only one of them")
        if input_features is not None and inputs_embeds is None:
            inputs_embeds = input_features

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss = ops.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1), ignore_index=-100)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def generate(
        self,
        input_features,
        attention_mask=None,
        composer="composer1",
        generation_config=None,
        **kwargs,
    ):
        """
        Generates token ids for midi outputs.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`. For an overview of generation
        strategies and code examples, check out the [following guide](./generation_strategies).

        </Tip>

        Parameters:
            input_features (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                This is the featurized version of audio generated by `Pop2PianoFeatureExtractor`.
            attention_mask:
                For batched generation `input_features` are padded to have the same shape across all examples.
                `attention_mask` helps to determine which areas were padded and which were not.

                - 1 for tokens that are **not padded**,
                - 0 for tokens that are **padded**.
            composer (`str`, *optional*, defaults to `"composer1"`):
                This value is passed to `Pop2PianoConcatEmbeddingToMel` to generate different embeddings for each
                `"composer"`. Please make sure that the composet value is present in `composer_to_feature_token` in
                `generation_config`. For an example please see
                https://hf-mirror.com/sweetcocoa/pop2piano/blob/main/generation_config.json .
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them.

                If `generation_config` is not provided, the default will be used, which had the following loading
                priority:

                1. from the `generation_config.json` model file, if it exists;
                2. from the model configuration. Please note that unspecified parameters will inherit
                [`~generation.GenerationConfig`]'s default values, whose documentation should be checked to parameterize
                generation.
            kwargs:
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Returns:
            [`~utils.ModelOutput`] or `mindspore.Tensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
                or when `config.return_dict_in_generate=True`) or a `mindspore.Tensor`.

                Since Pop2Piano is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        if generation_config is None:
            generation_config = self.generation_config
        generation_config.update(**kwargs)

        # check for composer_to_feature_token
        if not hasattr(generation_config, "composer_to_feature_token"):
            raise ValueError(
                "`composer_to_feature_token` was not found! Please refer to "
                "https://hf-mirror.com/sweetcocoa/pop2piano/blob/main/generation_config.json"
                "and parse a dict like that."
            )

        if len(generation_config.composer_to_feature_token) != self.config.composer_vocab_size:
            raise ValueError(
                "config.composer_vocab_size must be same as the number of keys in "
                f"generation_config.composer_to_feature_token! "
                f"Found {self.config.composer_vocab_size} vs {len(generation_config.composer_to_feature_token)}."
            )

        # to control the variation of generated MIDI tokens we concatenate mel-conditioner tokens(which depends on composer_token)
        # at the front of input_features.
        input_features, attention_mask = self.get_mel_conditioner_outputs(
            input_features=input_features,
            attention_mask=attention_mask,
            composer=composer,
            generation_config=generation_config,
        )

        return super().generate(
            inputs=None,
            inputs_embeds=input_features,
            attention_mask=attention_mask,
            generation_config=generation_config,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        This method prepares inputs for generation in the Pop2PianoForConditionalGeneration class.

        Args:
            self: The instance of the class.
            input_ids (Tensor): The input tensor containing the token ids for the input sequence.
            past_key_values (Tuple): A tuple of tensors containing the past key and value states for fast decoding.
                Defaults to None.
            attention_mask (Tensor): An optional tensor of the same size as input_ids, used to mask the input tokens.
                Defaults to None.
            head_mask (Tensor): An optional tensor with shape (num_heads,) that is used to mask the attention heads.
                Defaults to None.
            decoder_head_mask (Tensor): An optional tensor with shape (num_heads,) that is used to mask the decoder
                attention heads. Defaults to None.
            cross_attn_head_mask (Tensor): An optional tensor with shape (num_heads,) that is used to mask the
                cross-attention heads. Defaults to None.
            use_cache (bool): A flag indicating whether to use the cache for fast decoding. Defaults to None.
            encoder_outputs (Tuple): A tuple of tensors containing the encoder outputs, used in the cross-attention
                mechanism.

        Returns:
            None.

        Raises:
            None
        """
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: mindspore.Tensor):
        """
        Prepare decoder input IDs from labels for conditional generation.

        Args:
            self (Pop2PianoForConditionalGeneration): The instance of the Pop2PianoForConditionalGeneration class.
            labels (mindspore.Tensor): The labels tensor representing the target sequence.
                It serves as the input to construct the decoder input IDs by shifting the labels to the right.

        Returns:
            None: This method does not return a value explicitly. It prepares the decoder input IDs for the model.

        Raises:
            None: This method does not raise any exceptions.
        """
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorders the cache for the Pop2PianoForConditionalGeneration class.

        This method takes three parameters: self, past_key_values, and beam_idx.

        Args:
            self: An instance of the Pop2PianoForConditionalGeneration class.
            past_key_values: A tuple representing the past key values of the decoder.
                It contains the cached states for each layer of the decoder.
                If None, a warning will be logged and the method will return None.
            beam_idx: A tensor representing the indices of the beams. It is used to reorder the past key values.
        
        Returns:
            reordered_decoder_past: A tuple representing the reordered past key values.
                It contains the reordered states for each layer of the decoder.
        
        Raises:
            ValueError: If the shape of the reordered_layer_past_states[0] and layer_past_states[0] do not match,
                or if the length of reordered_layer_past_states and layer_past_states do not match.
        """
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

__all__ = [
    "Pop2PianoPreTrainedModel",
    "Pop2PianoForConditionalGeneration",
]
