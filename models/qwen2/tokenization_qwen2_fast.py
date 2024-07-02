# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for Qwen2."""

from typing import Optional, Tuple

from mindnlp.utils import logging
from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from .tokenization_qwen2 import Qwen2Tokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"qwen/qwen-tokenizer": "https://hf-mirror.com/qwen/qwen-tokenizer/resolve/main/vocab.json"},
    "merges_file": {"qwen/qwen-tokenizer": "https://hf-mirror.com/qwen/qwen-tokenizer/resolve/main/merges.txt"},
    "tokenizer_file": {
        "qwen/qwen-tokenizer": "https://hf-mirror.com/qwen/qwen-tokenizer/resolve/main/tokenizer.json"
    },
}

MAX_MODEL_INPUT_SIZES = {"qwen/qwen-tokenizer": 32768}


class Qwen2TokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Qwen2 tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    Same with GPT2Tokenizer, this tokenizer has been trained to treat spaces like parts of the tokens so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    Example:
        ```python
        >>> from transformers import Qwen2TokenizerFast
        ...
        >>> tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen-tokenizer")
        >>> tokenizer("Hello world")["input_ids"]
        [9707, 1879]
        >>> tokenizer(" Hello world")["input_ids"]
        [21927, 1879]
        ```
    This is expected.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. Not applicable to this tokenizer.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not applicable for this tokenizer.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = MAX_MODEL_INPUT_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = Qwen2Tokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        **kwargs,
    ):
        """
        Initializes a new instance of the Qwen2TokenizerFast class.
        
        Args:
            self: The instance of the class.
            vocab_file (str, optional): The path to the vocabulary file. Default is None.
            merges_file (str, optional): The path to the merges file. Default is None.
            tokenizer_file (str, optional): The path to the tokenizer file. Default is None.
            unk_token (str, optional): The unknown token. Default is 'endoftext'.
            bos_token (str or AddedToken, optional): The beginning of sequence token. Default is None.
            eos_token (str or AddedToken, optional): The end of sequence token. Default is 'endoftext'.
            pad_token (str or AddedToken, optional): The padding token. Default is 'endoftext'.
            
        Returns:
            None.
        
        Raises:
            None.
        
        Note:
            - The bos_token, eos_token, unk_token, and pad_token parameters can be either a string or an instance of
            the AddedToken class.
            - If any of the bos_token, eos_token, unk_token, or pad_token parameters are provided as strings,
            they will be converted to AddedToken instances with default properties.
            - The vocab_file, merges_file, and tokenizer_file parameters are used to load the respective files
            for the tokenizer.
            - The unk_token, bos_token, eos_token, and pad_token parameters are used to set the respective tokens
            in the tokenizer.
            - Additional keyword arguments can be provided and will be passed to the base class constructor.
        """
        # We need to at least pass vocab_file and merges_file to base class
        # in case a slow tokenizer needs to be initialized; other can be
        # configured through files.
        # following GPT2TokenizerFast, also adding unk_token, bos_token, and eos_token

        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(pad_token, str)
            else pad_token
        )

        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    # Copied from transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary of the Qwen2TokenizerFast model to the specified directory.
        
        Args:
            self: The instance of the Qwen2TokenizerFast class.
            save_directory (str): The directory where the vocabulary files will be saved.
            filename_prefix (Optional[str]): An optional prefix to be added to the vocabulary filenames. Default is None.
        
        Returns:
            Tuple[str]: A tuple containing the filenames of the saved vocabulary files.
        
        Raises:
            This method does not explicitly raise any exceptions.
        """
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

__all__ = ['Qwen2TokenizerFast']
