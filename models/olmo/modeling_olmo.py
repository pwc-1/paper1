# coding=utf-8
# Copyright 2024 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""MindSpore OLMo model."""

import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import Normal

from ....modules.functional import finfo
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import ALL_LAYERNORM_LAYERS
from ....utils import logging, get_default_dtype
from .configuration_olmo import OlmoConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "OlmoConfig"


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    """
    Args:
        attention_mask (Tensor): A 2D tensor representing the attention mask. Each element in the tensor
            should be a binary value indicating whether the corresponding token is masked or not.
    
    Returns:
        tuple:
            A tuple containing the following elements:

            - indices (Tensor): A 1D tensor containing the indices of non-zero elements in the flattened attention mask.
            - cu_seqlens (Tensor): A 1D tensor representing the cumulative sum of sequence lengths in the batch.
            - max_seqlen_in_batch (int): The maximum sequence length in the batch.

    Raises:
        None
    """
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype=mindspore.int32)
    indices = ops.nonzero(attention_mask.flatten()).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = ops.pad(ops.cumsum(seqlens_in_batch, axis=0, dtype=mindspore.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class OlmoLayerNorm(nn.Cell):
    """LayerNorm but with no learnable weight or bias."""
    def __init__(self, hidden_size: int) -> None:
        """
        Initializes a new instance of the OlmoLayerNorm class.

        Args:
            self (OlmoLayerNorm): The instance of the class.
            hidden_size (int): The size of the hidden dimension for the layer normalization.
                It determines the shape of the normalized layer. The hidden size must be a positive integer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.normalized_shape = (hidden_size,)
        self.layer_norm = ops.LayerNorm(begin_norm_axis=-1,
                                      begin_params_axis=-1,
                                      epsilon=1e-5)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the OlmoLayerNorm for the given hidden states.

        Args:
            self (OlmoLayerNorm): An instance of the OlmoLayerNorm class.
            hidden_states (mindspore.Tensor): The input hidden states to be normalized.

        Returns:
            mindspore.Tensor: The normalized hidden states.

        Raises:
            TypeError: If the input hidden states are not of type mindspore.Tensor.
            ValueError: If the input hidden states are empty or have incompatible shape.

        Note:
            - The input hidden states should have a valid shape compatible with the layer normalization operation.
            - The hidden states are expected to be of a specific data type.

        Example:
            ```python
            >>> norm = OlmoLayerNorm()
            >>> input_states = mindspore.Tensor([1, 2, 3], mindspore.float32)
            >>> output_states = norm.construct(input_states)
            ```
        """
        orig_dtype = hidden_states.dtype
        y, _, _ = self.layer_norm(hidden_states.to(dtype=mindspore.float32),
                                  ops.ones(self.normalized_shape, mindspore.float32),
                                  ops.zeros(self.normalized_shape, mindspore.float32))
        return y.to(orig_dtype)


ALL_LAYERNORM_LAYERS.append(OlmoLayerNorm)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Olmo
class OlmoRotaryEmbedding(nn.Cell):

    """
    This class represents an implementation of Olmo Rotary Embedding for neural networks.
    It provides methods to calculate and cache cosine and sine values based on positional embeddings for efficient
    computation in attention mechanisms. The class inherits from nn.Cell and includes initialization parameters for
    dimensionality, maximum position embeddings, base value, and scaling factor.
    The class also includes methods to calculate cosine and sine values based on positional embeddings, and provides
    warnings for deprecated attributes.

    Note:
        The 'sin_cached' and 'cos_cached' attributes will be removed in version 4.39 and their contents changed in
        version 4.38.  It is recommended to use the 'forward' method of RoPE instead of accessing these attributes
        directly.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        """
        Initializes an instance of the OlmoRotaryEmbedding class.

        Args:
            self: The object itself.
            dim (int): The dimensionality of the rotary embeddings.
            max_position_embeddings (int, optional): The maximum number of position embeddings. Defaults to 2048.
            base (int, optional): The base value used for calculating inverse frequencies. Defaults to 10000.
            scaling_factor (float, optional): The scaling factor applied to the sequence length. Defaults to 1.0.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (ops.arange(0, self.dim, 2, dtype=mindspore.int64).float() / self.dim))
        self.inv_freq = inv_freq
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings
        t = ops.arange(self.max_seq_len_cached, dtype=mindspore.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self._cos_cached = emb.cos().to(get_default_dtype())
        self._sin_cached = emb.sin().to(get_default_dtype())

    @property
    def sin_cached(self):
        """
        Returns the cached value of the sine of the input.

        Args:
            self: An instance of the OlmoRotaryEmbedding class.

        Returns:
            Conditional return:
                This method returns the cached value of the sine of the input, or None if the cache is empty.

        Raises:
            None.
        """
        logger.warning_once(
            "The sin_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `OlmoAttention` class"
        )
        return self._sin_cached

    @property
    def cos_cached(self):
        """
        This method 'cos_cached' in the class 'OlmoRotaryEmbedding' retrieves the cached cosine similarity value.

        Args:
            self: An instance of the 'OlmoRotaryEmbedding' class.

        Returns:
            None.

        Raises:
            None
        """
        logger.warning_once(
            "The cos_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `OlmoAttention` class"
        )
        return self._cos_cached

    def construct(self, x, position_ids):
        """
        Constructs the OlmoRotaryEmbedding.

        Args:
            self: OlmoRotaryEmbedding
                The instance of the OlmoRotaryEmbedding class.
            x: torch.Tensor
                The input tensor.
            position_ids: torch.Tensor
                The position IDs tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
                The tuple containing the cosine and sine values computed based on the input and position IDs.

        Raises:
            None
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).swapaxes(1, 2)
        emb = ops.cat((freqs, freqs), axis=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->Olmo
class OlmoLinearScalingRotaryEmbedding(OlmoRotaryEmbedding):
    """OlmoRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""
    def construct(self, x, position_ids):
        """
        Constructs the cosine and sine embeddings for the given input tensor 'x' with positional encoding.

        Args:
            self (OlmoLinearScalingRotaryEmbedding): The instance of the OlmoLinearScalingRotaryEmbedding class.
            x (Tensor): The input tensor for which the positional embeddings are constructed.
            position_ids (Tensor): The tensor containing positional indices.

        Returns:
            Tuple[Tensor, Tensor]:
                A tuple containing the cosine and sine embeddings constructed based on the input 'x' and 'position_ids'.

        Raises:
            TypeError: If the input 'position_ids' is not a tensor.
            ValueError: If the scaling factor 'self.scaling_factor' is not valid for the division operation.
            NotImplementedError: If the superclass method 'construct' is not implemented.
        """
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().construct(x, position_ids)
        return cos, sin


# Copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->Olmo
class OlmoDynamicNTKScalingRotaryEmbedding(OlmoRotaryEmbedding):
    """OlmoRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""
    def construct(self, x, position_ids):
        """Constructs the OlmoDynamicNTKScalingRotaryEmbedding.

        This method initializes the OlmoDynamicNTKScalingRotaryEmbedding object by constructing the positional
        encodings for the input tensor.

        Args:
            self (OlmoDynamicNTKScalingRotaryEmbedding): An instance of the OlmoDynamicNTKScalingRotaryEmbedding class.
            x (Tensor): The input tensor.
            position_ids (Tensor): The tensor containing the positional indices.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the cosine and sine of the positional encodings.

        Raises:
            None.
        """
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = ops.max(position_ids)[0] + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (ops.arange(0, self.dim, 2, dtype=mindspore.int64).float() / self.dim)
            )
            self.inv_freq = inv_freq
        cos, sin = super().construct(x, position_ids)
        return cos, sin


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # x1 = x[..., : x.shape[-1] // 2]
    # x2 = x[..., x.shape[-1] // 2 :]
    x1, x2 = x.tensor_split(2, -1)
    return ops.cat((-x2, x1), axis=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`mindspore.Tensor`): The query tensor.
        k (`mindspore.Tensor`): The key tensor.
        cos (`mindspore.Tensor`): The cosine part of the rotary embedding.
        sin (`mindspore.Tensor`): The sine part of the rotary embedding.
        position_ids (`mindspore.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.

    Returns:
        `tuple(mindspore.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class OlmoMLP(nn.Cell):

    """
    The 'OlmoMLP' class represents a multi-layer perceptron (MLP) with customized operations for gating, projection,
    and activation functions. This class inherits from the 'nn.Cell' class.

    Attributes:
        config (object): The configuration object that stores the parameters for the MLP.
        hidden_size (int): The size of the hidden layer in the MLP.
        intermediate_size (int): The size of the intermediate layer in the MLP.
        gate_proj (nn.Dense): The dense layer used for projecting the input into the intermediate size for gating.
        up_proj (nn.Dense): The dense layer used for projecting the input into the intermediate size for the up projection.
        down_proj (nn.Dense): The dense layer used for projecting the intermediate size back to the hidden size.
        act_fn (function): The activation function applied to the output of the gating and up projection.

    Methods:
        __init__: Initializes the 'OlmoMLP' class with the given configuration object.
        construct: Constructs the MLP by applying the necessary operations to the input 'x' and returning the result.

    Example:
        ```python
        >>> # Create a configuration object
        >>> config = MLPConfig(hidden_size=128, intermediate_size=64, hidden_act='relu')
        ...
        >>> # Create an instance of the 'OlmoMLP' class
        >>> mlp = OlmoMLP(config)
        ...
        >>> # Construct the MLP
        >>> output = mlp.construct(input_data)
        ```

    Note:
        The 'OlmoMLP' class assumes that the 'ACT2FN' dictionary is defined, which maps the activation function names
        to their corresponding functions.
    """
    def __init__(self, config):
        """
        Initializes an instance of the OlmoMLP class.

        Args:
            self: The instance of the OlmoMLP class.
            config: An object of type 'Config' that contains the configuration settings for the OlmoMLP model.
                It must have the following attributes:

                - hidden_size: An integer representing the size of the hidden layers.
                - intermediate_size: An integer representing the size of the intermediate layers.
                - hidden_act: A string representing the activation function to be used in the hidden layers.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.up_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(self.intermediate_size, self.hidden_size, has_bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def construct(self, x):
        """
        Constructs a multi-layer perceptron using the specified input data.

        Args:
            self (OlmoMLP): An instance of the OlmoMLP class.
            x: Input data for constructing the MLP.

        Returns:
            None: The method modifies the MLP model in-place.

        Raises:
            TypeError: If the input data is not in the expected format.
            ValueError: If the input data is invalid or incompatible with the model.
            RuntimeError: If there is an issue during the construction process.
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: mindspore.Tensor, n_rep: int) -> mindspore.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class OlmoAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # Copied from transformers.models.llama.modeling_llama.LlamaAttention.__init__ with Llama->Olmo
    def __init__(self, config: OlmoConfig, layer_idx: Optional[int] = None):
        """
        Initializes an instance of the OlmoAttention class.

        Args:
            self: The instance of the class itself.
            config: An instance of the OlmoConfig class, containing configuration parameters for the attention layer.
            layer_idx (Optional[int]): The index of the layer. If not provided, it is set to None.
                Not providing a `layer_idx` is not recommended and may lead to errors during the forward call
                if caching is used. Please make sure to provide a `layer_idx` when creating this class.

        Returns:
            None

        Raises:
            ValueError: If `hidden_size` is not divisible by `num_heads`.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=config.attention_bias)
        self.k_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=config.attention_bias)
        self.v_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=config.attention_bias)
        self.o_proj = nn.Dense(self.hidden_size, self.hidden_size, has_bias=config.attention_bias)
        self._init_rope()

    # Copied from transformers.models.llama.modeling_llama.LlamaAttention._init_rope with Llama->Olmo
    def _init_rope(self):
        """
        Initializes the RoPE (Rotary Positional Encoding) for the OlmoAttention class.

        Args:
            self: An instance of the OlmoAttention class.

        Returns:
            None

        Raises:
            ValueError: If the RoPE scaling type is unknown.

        This method initializes the RoPE based on the provided configuration. The RoPE is used to incorporate positional
        information into the attention mechanism of the OlmoAttention model.

        If the 'rope_scaling' configuration parameter is not specified, the RoPE is initialized with the
        OlmoRotaryEmbedding class using the default parameters.

        If the 'rope_scaling' configuration parameter is specified, the RoPE is initialized with a specific scaling
        type and factor. The 'scaling_type' parameter determines the type of scaling to be used, and the
        'scaling_factor' parameter determines the scaling factor to be applied.
        The available scaling types are 'linear' and 'dynamic'.

        - For 'linear' scaling type, the RoPE is initialized with the OlmoLinearScalingRotaryEmbedding class using the
        specified scaling factor.
        - For 'dynamic' scaling type, the RoPE is initialized with the OlmoDynamicNTKScalingRotaryEmbedding class using
        the specified scaling factor.

        Note:
            The 'scaling_factor' parameter is used to adjust the scale of the RoPE embeddings.
            A higher scaling factor results in more distinct embeddings for different positions.

        If the 'scaling_type' provided is not one of the available options, a ValueError is raised.

        Example:
            ```python
            >>> olmo_attention = OlmoAttention()
            >>> olmo_attention._init_rope()
            ```
            or
            ``` python
            >>> config = {'rope_scaling': {'type': 'linear', 'factor': 2.0}}
            >>> olmo_attention = OlmoAttention(config)
            >>> olmo_attention._init_rope()
            ```
        """
        if self.config.rope_scaling is None:
            self.rotary_emb = OlmoRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = OlmoLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = OlmoDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[mindspore.Tensor] = None,
        **kwargs,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        '''
        Constructs the attention mechanism for OlmoAttention.

        Args:
            self (OlmoAttention): An instance of the OlmoAttention class.
            hidden_states (mindspore.Tensor): The hidden states input tensor of shape
                (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor of shape
                (batch_size, num_heads, sequence_length, sequence_length), where each element is either 0 or 1.
            position_ids (Optional[mindspore.Tensor]): The position ids tensor of shape (batch_size, sequence_length).
            past_key_value (Optional[Cache]): The past key-value cache for efficient attention computation.
            output_attentions (bool): Flag indicating whether to output the attention weights.
            use_cache (bool): Flag indicating whether to use the past key-value cache.
            cache_position (Optional[mindspore.Tensor]): The position tensor for the key-value cache.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
                A tuple containing the attention output tensor of shape (batch_size, sequence_length, hidden_size), the
                attention weights tensor of shape (batch_size, num_heads, sequence_length, sequence_length), and the
                updated past key-value cache.

        Raises:
            ValueError: If the shape of the attention output tensor is not
                (batch_size, num_heads, sequence_length, hidden_size).
        '''
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states = query_states.clamp(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states = key_states.clamp(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states = value_states.clamp(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = ops.matmul(query_states, key_states.swapaxes(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = ops.softmax(attn_weights, axis=-1, dtype=mindspore.float32).to(query_states.dtype)
        attn_weights = ops.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = ops.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.swapaxes(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


OLMO_ATTENTION_CLASSES = {
    "eager": OlmoAttention,
}


class OlmoDecoderLayer(nn.Cell):

    """
    This class represents a decoder layer in the Olmo model. It inherits from the nn.Cell class.

    Attributes:
        hidden_size (int): The size of the hidden state.
        self_attn: An instance of the OLMO_ATTENTION_CLASSES['eager'] class for self-attention.
        mlp: An instance of the OlmoMLP class.
        input_layernorm: An instance of the OlmoLayerNorm class for input layer normalization.
        post_attention_layernorm: An instance of the OlmoLayerNorm class for post-attention layer normalization.

    Warnings:
        Passing `padding_mask` is deprecated and will be removed in v4.37.
        Please make sure to use `attention_mask` instead.

    Note:
        The construct method is the entry point for the decoder layer.
    """
    def __init__(self, config: OlmoConfig, layer_idx: int):
        """
        Initializes an instance of OlmoDecoderLayer.

        Args:
            self (OlmoDecoderLayer): The instance of the OlmoDecoderLayer class.
            config (OlmoConfig): An instance of OlmoConfig that contains configuration settings for the decoder layer.
            layer_idx (int): An integer representing the index of the layer.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not an instance of OlmoConfig.
            ValueError: If the layer_idx parameter is not an integer.
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = OLMO_ATTENTION_CLASSES["eager"](config=config, layer_idx=layer_idx)

        self.mlp = OlmoMLP(config)
        self.input_layernorm = OlmoLayerNorm(config.hidden_size)
        self.post_attention_layernorm = OlmoLayerNorm(config.hidden_size)

    # Copied from transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward
    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[mindspore.Tensor] = None,
        **kwargs,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(mindspore.Tensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

# Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel with Llama->Olmo
class OlmoPreTrainedModel(PreTrainedModel):

    """
    This class represents a pre-trained model for Olmo, which is a subclass of the PreTrainedModel class.

    OlmoPreTrainedModel provides methods for initializing weights, setting up cache, and resetting cache.

    Methods:
        _init_weights:
            Initializes the weights of the given cell.

            - If the cell is of type nn.Dense, the weight is initialized using a normal distribution with a
            standard deviation of self.config.initializer_range.
            - If the cell has a bias, the bias is initialized to zeros.
            - If the cell is of type nn.Embedding, the weight is initialized using a normal distribution with a
            standard deviation of self.config.initializer_range.
            - If the cell has a padding index, the weight at the padding index is set to 0.

        _setup_cache: Sets up the cache for the model.
            If the attention implementation is 'flash_attention_2' and the cache class is StaticCache,
            a ValueError is raised. For each layer in the model, the cache is set to an instance of the cache class,
            with the specified maximum batch size, maximum cache length, and data type.

        _reset_cache: Resets the cache for the model. For each layer in the model, the cache is set to None.

    Note:
        The OlmoPreTrainedModel class assumes the existence of a model attribute, which is expected to have a
        layers attribute. Additionally, it checks for the existence of a _pre_quantization_dtype attribute in the 
        config attribute.
        For more information on Olmo, refer to the documentation at https://github.com/huggingface/transformers.
    """
    config_class = OlmoConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["OlmoDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_cache_class = True

    def _init_weights(self, cell):
        """
        Initializes the weights of a cell in the OlmoPreTrainedModel.

        Args:
            self (OlmoPreTrainedModel): The instance of the OlmoPreTrainedModel class.
            cell: The cell to initialize the weights for.

        Returns:
            None.

        Raises:
            None.
        """
        std = self.config.initializer_range
        if isinstance(cell, nn.Dense):
            cell.weight.initialize(Normal(std))
            if cell.bias is not None:
                cell.bias.initialize('zeros')
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, std, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))

    def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):
        """
        This method initializes the cache for the model's past key-value pairs used in self-attention mechanisms.

        Args:
            self (OlmoPreTrainedModel): The instance of the OlmoPreTrainedModel class.
            cache_cls (class): The class representing the cache implementation to be used.
            max_batch_size (int): The maximum batch size for caching past key-value pairs.
            max_cache_len (Optional[int]): The maximum length of the cache. Defaults to None.

        Returns:
            None.

        Raises:
            ValueError: Raised if the `static` cache implementation is selected while using 
                `attn_implementation==flash_attention_2`.  In such cases, it is recommended to use `sdpa` instead and 
                report the issue at https://github.com/huggingface/transformers.
        """
        if self.config._attn_implementation == "flash_attention_2" and cache_cls == StaticCache:
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        for layer in self.model.layers:
            if hasattr(self.config, "_pre_quantization_dtype"):
                dtype = self.config._pre_quantization_dtype
            else:
                dtype = layer.self_attn.o_proj.weight.dtype
            layer.self_attn.past_key_value = cache_cls(
                self.config, max_batch_size, max_cache_len, dtype=dtype
            )

    def _reset_cache(self):
        """
        Resets the cache for the self-attention layers in the OlmoPreTrainedModel.

        Args:
            self (OlmoPreTrainedModel): The instance of the OlmoPreTrainedModel class.

        Returns:
            None.

        Raises:
            None.
        """
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None


class OlmoModel(OlmoPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OlmoDecoderLayer`]

    Args:
        config: OlmoConfig
    """
    def __init__(self, config: OlmoConfig):
        """
        Initializes an instance of the `OlmoModel` class.

        Args:
            self: The instance of the class.
            config (OlmoConfig):
                An object containing the configuration parameters for the model.

                - `pad_token_id` (int): The token ID used for padding sequences.
                - `vocab_size` (int): The size of the vocabulary.
                - `hidden_size` (int): The hidden size of the model.
                - `num_hidden_layers` (int): The number of hidden layers in the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.CellList(
            [OlmoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = OlmoLayerNorm(config.hidden_size)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Get the input embeddings for the OlmoModel class.

        Args:
            self (OlmoModel): The instance of the OlmoModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        This method sets the input embeddings for the OlmoModel.

        Args:
            self (OlmoModel): The instance of the OlmoModel class.
            value (object): The input embeddings to be set for the OlmoModel.
                It should be of type 'object' and can contain the input embeddings data.

        Returns:
            None.

        Raises:
            None.
        """
        self.embed_tokens = value

    # Copied from transformers.models.llama.modeling_llama.LlamaModel.forward
    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Constructs the OlmoModel.

        Args:
            self: The object instance.
            input_ids (mindspore.Tensor, optional): The input tensor containing the token IDs. Default is None.
            attention_mask (mindspore.Tensor, optional): The attention mask tensor. Default is None.
            position_ids (mindspore.Tensor, optional): The tensor containing the position IDs. Default is None.
            past_key_values (List[mindspore.Tensor], optional): The list of tensors containing the past key values.
                Default is None.
            inputs_embeds (mindspore.Tensor, optional): The input tensor containing the embedded inputs. Default is None.
            use_cache (bool, optional): Whether to use cache. Default is None.
            output_attentions (bool, optional): Whether to output attentions. Default is None.
            output_hidden_states (bool, optional): Whether to output hidden states. Default is None.
            return_dict (bool, optional): Whether to return a dictionary instead of tuple. Default is None.
            cache_position (mindspore.Tensor, optional): The tensor containing the cache position. Default is None.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]: A tuple or BaseModelOutputWithPast object containing the model outputs.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified at the same time.
            ValueError: If use_cache is True and gradient checkpointing is enabled.
            ValueError: If cache_position is not specified when using StaticCache.

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = ops.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1])

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: mindspore.Tensor,
        input_tensor: mindspore.Tensor,
        cache_position: mindspore.Tensor,
        past_seen_tokens: int,
    ):
        """
        Update the causal mask for self-attention mechanism.

        Args:
            self (OlmoModel): The instance of the OlmoModel class.
            attention_mask (mindspore.Tensor): A 2D or 4D tensor representing the attention mask.
                This mask is used to exclude certain positions from consideration during attention calculation.
            input_tensor (mindspore.Tensor): The input tensor to the model.
                It is used to determine the dtype and shape information for creating the causal mask.
            cache_position (mindspore.Tensor): A tensor representing the position in the cache to update the causal mask.
            past_seen_tokens (int): The number of tokens seen in the past.

        Returns:
            None: This method updates the causal mask in place and does not return any value.

        Raises:
            ValueError: If the input_tensor dtype is not supported for calculating the min value.
            RuntimeError: If there is an issue in updating the causal mask due to incorrect dimensions or values
                in the input tensors.
        """
        dtype = input_tensor.dtype
        min_dtype = finfo(dtype, 'min')
        sequence_length = input_tensor.shape[1]
        if hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, mindspore.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = ops.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype)
        if sequence_length != 1:
            causal_mask = ops.triu(causal_mask, diagonal=1)
        causal_mask *= ops.arange(target_length) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.copy()  # copy to contiguous memory for in-place edit
            if attention_mask.ndim == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) & attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.ndim == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        return causal_mask


# Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM with LLAMA->OLMO,Llama->Olmo
class OlmoForCausalLM(OlmoPreTrainedModel):

    """
    This class represents a model for Causal Language Modeling using Olmo. It is a subclass of OlmoPreTrainedModel.

    The class contains the following methods:

    - `__init__`: Initializes the class instance with a given configuration.
    - `get_input_embeddings`: Returns the input embeddings of the model.
    - `set_input_embeddings`: Sets the input embeddings of the model.
    - `get_output_embeddings`: Returns the output embeddings of the model.
    - `set_output_embeddings`: Sets the output embeddings of the model.
    - `set_decoder`: Sets the decoder of the model.
    - `get_decoder`: Returns the decoder of the model.
    - `construct`: Constructs the model and returns the output.
    - `prepare_inputs_for_generation`: Prepares the inputs for generation.

    The class also includes a private static method `_reorder_cache(past_key_values, beam_idx)`.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, OlmoForCausalLM
        ...
        >>> model = OlmoForCausalLM.from_pretrained("allenai/OLMo-1B-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
        ...
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        ...
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(generated_text)
        ```
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes an instance of the OlmoForCausalLM class.

        Args:
            self: The current instance of the class.
            config: An instance of the configuration class for OlmoForCausalLM.
                It contains various parameters and settings used for model initialization.

                - Type: config object
                - Purpose: To customize the behavior of the model.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.model = OlmoModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method is implemented in the 'OlmoForCausalLM' class and is used to retrieve the
        input embeddings from the model.

        Args:
            self: An instance of the 'OlmoForCausalLM' class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the OlmoForCausalLM model.

        Args:
            self (OlmoForCausalLM): The instance of the OlmoForCausalLM class.
            value: The input embeddings to be set for the model. It should be a tensor representing the embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """
        This method, 'get_output_embeddings', is defined in the class 'OlmoForCausalLM' and returns the 'lm_head' attribute.

        Args:
            self: An instance of the 'OlmoForCausalLM' class.

        Returns:
            The 'lm_head' attribute: which is of type 'None'. The 'lm_head' is the output embedding layer of the model.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings of the OlmoForCausalLM model.

        Args:
            self (OlmoForCausalLM): The instance of the OlmoForCausalLM class.
            new_embeddings: The new embeddings to be set for the output layer of the model.
                This can be a tensor or any object that can be assigned to `self.lm_head`.
                The shape of the embeddings should match the expected shape of the output layer.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Sets the decoder for the OlmoForCausalLM model.

        Args:
            self (OlmoForCausalLM): The instance of the OlmoForCausalLM class.
            decoder: The decoder to be set for the model. It should be compatible with the OlmoForCausalLM model.

        Returns:
            None.

        Raises:
            None.
        """
        self.model = decoder

    def get_decoder(self):
        """
        This method returns the decoder model used for OlmoForCausalLM.

        Args:
            self: The instance of the OlmoForCausalLM class.

        Returns:
            model: The decoder model associated with the OlmoForCausalLM instance.

        Raises:
            None.
        """
        return self.model

    # Ignore copy
    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Union[Tuple, CausalLMOutputWithPast]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, OlmoForCausalLM
            ...
            >>> model = OlmoForCausalLM.from_pretrained("allenai/OLMo-1B-hf")
            >>> tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
            ...
            >>> prompt = "Hey, are you conscious? Can you talk to me?"
            >>> inputs = tokenizer(prompt, return_tensors="pt")
            ...
            >>> # Generate
            >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
            >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            'Hey, are you conscious? Can you talk to me?\nI’m not sure if you’re conscious of this, but I’m'
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            loss = ops.cross_entropy(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        """
        This method prepares inputs for text generation for OlmoForCausalLM model.
        
        Args:
            self (object): The instance of the class.
            input_ids (tensor): The input tensor containing tokenized input sequence.
            past_key_values (tensor, optional): The tensor of cached key values for previous time steps.
                Defaults to None.
            attention_mask (tensor, optional): The attention mask tensor to avoid attending to padding tokens.
                Defaults to None.
            inputs_embeds (tensor, optional): The tensor of embeddings for input tokens. Defaults to None.
            cache_position (tensor, optional): The tensor specifying the position in the cache. Defaults to None.
            **kwargs: Additional keyword arguments.
        
        Returns:
            None.
        
        Raises:
            ValueError: If attention_mask and input_ids have incompatible shapes.
            ValueError: If past_key_values and inputs_embeds are both provided.
        
        """
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    mindspore.tensor(past_key_values.get_max_length())
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else ops.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = ops.arange(past_length, past_length + input_length)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Method to reorder cached states for beam search in OlmoForCausalLM.
        
        Args:
            past_key_values (tuple): A tuple containing cached states from previous layers.
                Each element in the tuple represents the cached states for a layer.
                These states are used during inference for generating the next tokens.
            beam_idx (Tensor): A 1D tensor containing the indices of beams to reorder the cached states.
                This tensor specifies the new order in which the cached states should be arranged.
        
        Returns:
            None: This method does not return any value but updates the order of the cached states based on the
                given beam indices.
        
        Raises:
            IndexError: If the provided beam indices are out of range or incompatible with the cached states.
            TypeError: If the input parameters are not of the expected types
                (tuple for past_key_values, Tensor for beam_idx).
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past

__all__ = [
    "OlmoForCausalLM",
    "OlmoModel",
    "OlmoPreTrainedModel",
]
