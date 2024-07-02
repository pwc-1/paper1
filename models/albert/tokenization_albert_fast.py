# coding=utf-8
# Copyright 2022 Huawei Technologies Co., Ltd
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
""" Tokenization classes for ALBERT model."""


import os
from shutil import copyfile
from typing import List, Optional, Tuple

from mindnlp.utils import is_sentencepiece_available, logging
from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast


if is_sentencepiece_available():
    from .tokenization_albert import AlbertTokenizer
else:
    AlbertTokenizer = None

logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "albert-base-v1": "https://hf-mirror.com/albert-base-v1/resolve/main/spiece.model",
        "albert-large-v1": "https://hf-mirror.com/albert-large-v1/resolve/main/spiece.model",
        "albert-xlarge-v1": "https://hf-mirror.com/albert-xlarge-v1/resolve/main/spiece.model",
        "albert-xxlarge-v1": "https://hf-mirror.com/albert-xxlarge-v1/resolve/main/spiece.model",
        "albert-base-v2": "https://hf-mirror.com/albert-base-v2/resolve/main/spiece.model",
        "albert-large-v2": "https://hf-mirror.com/albert-large-v2/resolve/main/spiece.model",
        "albert-xlarge-v2": "https://hf-mirror.com/albert-xlarge-v2/resolve/main/spiece.model",
        "albert-xxlarge-v2": "https://hf-mirror.com/albert-xxlarge-v2/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "albert-base-v1": "https://hf-mirror.com/albert-base-v1/resolve/main/tokenizer.json",
        "albert-large-v1": "https://hf-mirror.com/albert-large-v1/resolve/main/tokenizer.json",
        "albert-xlarge-v1": "https://hf-mirror.com/albert-xlarge-v1/resolve/main/tokenizer.json",
        "albert-xxlarge-v1": "https://hf-mirror.com/albert-xxlarge-v1/resolve/main/tokenizer.json",
        "albert-base-v2": "https://hf-mirror.com/albert-base-v2/resolve/main/tokenizer.json",
        "albert-large-v2": "https://hf-mirror.com/albert-large-v2/resolve/main/tokenizer.json",
        "albert-xlarge-v2": "https://hf-mirror.com/albert-xlarge-v2/resolve/main/tokenizer.json",
        "albert-xxlarge-v2": "https://hf-mirror.com/albert-xxlarge-v2/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "albert-base-v1": 512,
    "albert-large-v1": 512,
    "albert-xlarge-v1": 512,
    "albert-xxlarge-v1": 512,
    "albert-base-v2": 512,
    "albert-large-v2": 512,
    "albert-xlarge-v2": 512,
    "albert-xxlarge-v2": 512,
}

SPIECE_UNDERLINE = "▁"


class AlbertTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" ALBERT tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://hf-mirror.com/docs/tokenizers/python/latest/components.html?highlight=unigram#models). This
    tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        remove_space (`bool`, *optional*, defaults to `True`):
            Whether or not to strip the text when tokenizing (removing excess spaces before and after the string).
        keep_accents (`bool`, *optional*, defaults to `False`):
            Whether or not to keep accents when tokenizing.
        bos_token (`str`, *optional*, defaults to `"[CLS]"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token. .. note:: When building a sequence using special tokens, this is not the token
            that is used for the end of sequence. The token used is the `sep_token`.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = AlbertTokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        remove_space=True,
        keep_accents=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="<unk>",
        sep_token="[SEP]",
        pad_token="<pad>",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs,
    ):
        """
        Initialize the AlbertTokenizerFast class.
        
        Args:
            self (object): The instance of the class.
            vocab_file (str, optional): The file containing the vocabulary. Defaults to None.
            tokenizer_file (str, optional): The file containing the tokenizer. Defaults to None.
            do_lower_case (bool, optional): Flag to indicate if text should be lowercased. Defaults to True.
            remove_space (bool, optional): Flag to indicate if spaces should be removed. Defaults to True.
            keep_accents (bool, optional): Flag to indicate if accents should be kept. Defaults to False.
            bos_token (str, optional): The beginning of sequence token. Defaults to '[CLS]'.
            eos_token (str, optional): The end of sequence token. Defaults to '[SEP]'.
            unk_token (str, optional): The unknown token. Defaults to '<unk>'.
            sep_token (str, optional): The separator token. Defaults to '[SEP]'.
            pad_token (str, optional): The padding token. Defaults to '<pad>'.
            cls_token (str, optional): The classification token. Defaults to '[CLS]'.
            mask_token (str or AddedToken, optional): The masking token. Defaults to '[MASK]'.
        
        Returns:
            None.
        
        Raises:
            TypeError: If mask_token is not a string or an AddedToken.
        """
        # Mask token behave like a normal word, i.e. include the space before it and
        # is included in the raw text, there should be a match in a non-normalized sentence.
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        """
        Method to check if the slow tokenizer can be saved.
        
        Args:
            self (AlbertTokenizerFast): The instance of the AlbertTokenizerFast class.
                It is the current instance of the class to which this method belongs.
        
        Returns:
            bool: Returns a boolean value indicating whether the slow tokenizer can be saved.
                True if the vocab file exists, False if the vocab file does not exist.
        
        Raises:
            None
        """
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An ALBERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary file for a fast tokenizer.
        
        Args:
            self: Instance of the AlbertTokenizerFast class.
            save_directory (str): The directory where the vocabulary file will be saved.
            filename_prefix (Optional[str]): An optional prefix to be added to the vocabulary file name. Default is None.
        
        Returns:
            Tuple[str]: A tuple containing the path to the saved vocabulary file.
        
        Raises:
            ValueError: If the fast tokenizer does not have the necessary information to save the vocabulary for a slow tokenizer.
            FileNotFoundError: If the specified save_directory does not exist.
            OSError: If an error occurs while copying the vocabulary file to the save_directory.
        """
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

__all__ = ['AlbertTokenizerFast']
