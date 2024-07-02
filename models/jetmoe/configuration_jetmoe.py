# coding=utf-8
# Copyright 2024 JetMoE AI and the HuggingFace Inc. team. All rights reserved.
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
"""JetMoE model configuration"""

from ...configuration_utils import PretrainedConfig
from ....utils import logging


logger = logging.get_logger(__name__)


class JetMoEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`JetMoEModel`]. It is used to instantiate an
    JetMoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a configuration of the JetMoE-4B.

    [jetmoe/jetmoe-8b](https://huggingface.co/jetmoe/jetmoe-8b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the JetMoE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`JetMoEModel`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 12): Defines the number of blocks.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each key and value in the Transformer encoder.
        kv_channels (`int`, *optional*, defaults to 128): Defines the number of channels for the key and value tensors.
        ffn_hidden_size (`int`, *optional*, defaults to 5632): Defines the hidden size of the feed-forward layer.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with. JetMoE's sliding window attention
            allows sequence of up to 4096*32 tokens.
        activation_function (`string`, *optional*, defaults to `"silu"`): Defines the activation function for MLP experts.
        glu (`bool`, *optional*, defaults to `True`): Whether to use Gated Linear Units in the MLP experts.
        moe_num_experts (`int`, *optional*, defaults to 8): Defines the number of experts in the mixture of experts.
        moe_top_k (`int, *optional*, defaults to 2): Defines the number of experts to use for each token.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.
        bias (`bool`, *optional*, defaults to `True`): Whether to use bias in the feed-forward and attention layer.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:
        ```python
        >>> from transformers import JetMoEModel, JetMoEConfig
        ...
        >>> # Initializing a JetMoE 4B style configuration
        >>> configuration = JetMoEConfig()
        ...
        >>> # Initializing a model from the JetMoE 4B style configuration
        >>> model = JetMoEModel(configuration)
        ...
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """
    model_type = "jetmoe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=2048,
        num_hidden_layers=12,
        num_attention_heads=32,
        num_key_value_heads=16,
        kv_channels=128,
        ffn_hidden_size=5632,
        max_position_embeddings=4096,
        activation_function="silu",
        glu=True,
        moe_num_experts=8,
        moe_top_k=2,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        bias=True,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        initializer_range=0.01,
        **kwargs,
    ):
        """
        __init__
        
        Initializes a new instance of JetMoEConfig.
        
        Args:
            self: The instance of the class.
            vocab_size (int, optional): The size of the vocabulary. Defaults to 32000.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 2048.
            num_hidden_layers (int, optional): The number of hidden layers. Defaults to 12.
            num_attention_heads (int, optional): The number of attention heads. Defaults to 32.
            num_key_value_heads (int, optional): The number of key-value heads. Defaults to 16.
            kv_channels (int, optional): The number of channels for key and value computation. Defaults to 128.
            ffn_hidden_size (int, optional): The size of the feed-forward network hidden layers. Defaults to 5632.
            max_position_embeddings (int, optional): The maximum position for positional embeddings. Defaults to 4096.
            activation_function (str, optional): The activation function to be used. Defaults to 'silu'.
            glu (bool, optional): Whether to use Gated Linear Unit (GLU). Defaults to True.
            moe_num_experts (int, optional): The number of experts for Mixture of Experts (MoE). Defaults to 8.
            moe_top_k (int, optional): The top K experts to be used in MoE. Defaults to 2.
            use_cache (bool, optional): Whether to use cache for decoding. Defaults to True.
            bos_token_id (int, optional): The ID of the beginning of sequence token. Defaults to 1.
            eos_token_id (int, optional): The ID of the end of sequence token. Defaults to 2.
            tie_word_embeddings (bool, optional): Whether to tie word embeddings. Defaults to True.
            bias (bool, optional): Whether to use bias. Defaults to True.
            rope_theta (float, optional): The theta value for Relative Positional Embeddings (RoPE). Defaults to 10000.0.
            rms_norm_eps (float, optional): The epsilon value for RMSNorm. Defaults to 1e-06.
            initializer_range (float, optional): The range for weight initialization. Defaults to 0.01.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.kv_channels = kv_channels
        self.ffn_hidden_size = ffn_hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.activation_function = activation_function
        self.glu = glu
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.bias = bias
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps

        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )

__all__ = ['JetMoEConfig']
