# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
model nezha config
"""

from ...configuration_utils import PretrainedConfig

__all__ = ["NezhaConfig"]

NEZHA_SUPPORT_LIST = [
    "nezha-cn-base",
    "nezha-cn-large",
    "nezha-base-wwm",
    "nezha-large-wwm"
]


class NezhaConfig(PretrainedConfig):
    """
    Configuration for Nezha
    """
    def __init__(
        self,
        vocab_size=21128,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        max_relative_position=64,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        classifier_dropout=0.1,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        use_cache=True,
        **kwargs,
    ):
        '''
        Initializes a new instance of the NezhaConfig class.
        
        Args:
            self: The instance of the class.
            vocab_size (int, optional): The size of the vocabulary. Defaults to 21128.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 768.
            num_hidden_layers (int, optional): The number of hidden layers. Defaults to 12.
            num_attention_heads (int, optional): The number of attention heads. Defaults to 12.
            intermediate_size (int, optional): The size of the intermediate layers. Defaults to 3072.
            hidden_act (str, optional): The activation function for the hidden layers. Defaults to 'gelu'.
            hidden_dropout_prob (float, optional): The dropout probability for the hidden layers. Defaults to 0.1.
            attention_probs_dropout_prob (float, optional): The dropout probability for the attention probabilities.
                Defaults to 0.1.
            max_position_embeddings (int, optional): The maximum number of positional embeddings. Defaults to 512.
            max_relative_position (int, optional): The maximum relative position. Defaults to 64.
            type_vocab_size (int, optional): The size of the type vocabulary. Defaults to 2.
            initializer_range (float, optional): The range for the initializer. Defaults to 0.02.
            layer_norm_eps (float, optional): The epsilon value for layer normalization. Defaults to 1e-12.
            classifier_dropout (float, optional): The dropout probability for the classifier. Defaults to 0.1.
            pad_token_id (int, optional): The ID of the padding token. Defaults to 0.
            bos_token_id (int, optional): The ID of the beginning-of-sentence token. Defaults to 2.
            eos_token_id (int, optional): The ID of the end-of-sentence token. Defaults to 3.
            use_cache (bool, optional): Whether to use caching. Defaults to True.
        
        Returns:
            None.
        
        Raises:
            None.
        '''
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_relative_position = max_relative_position
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
