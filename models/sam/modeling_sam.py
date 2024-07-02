# coding=utf-8
# Copyright 2023 The Meta AI Authors and The HuggingFace Team. All rights reserved.
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
""" MindSpore SAM model."""

import collections
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import Normal

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ....utils import ModelOutput, logging
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "SamConfig"
_CHECKPOINT_FOR_DOC = "facebook/sam-vit-huge"


@dataclass
class SamVisionEncoderOutput(ModelOutput):
    """
    Base class for sam vision model's outputs that also contains image embeddings obtained by applying the projection
    layer to the pooler_output.

    Args:
        image_embeds (`mindspore.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model
            is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    image_embeds: Optional[mindspore.Tensor] = None
    last_hidden_state: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class SamImageSegmentationOutput(ModelOutput):
    """
    Base class for Segment-Anything model's output

    Args:
        iou_scores (`mindspore.Tensor` of shape `(batch_size, num_masks)`):
            The iou scores of the predicted masks.
        pred_masks (`mindspore.Tensor` of shape `(batch_size, num_masks, height, width)`):
            The predicted low resolutions masks. Needs to be post-processed by the processor
        vision_hidden_states  (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True`
            is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the vision model at the output of each layer plus the optional initial embedding outputs.
        vision_attentions  (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        mask_decoder_attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    iou_scores: mindspore.Tensor = None
    pred_masks: mindspore.Tensor = None
    vision_hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    vision_attentions: Optional[Tuple[mindspore.Tensor, ...]] = None
    mask_decoder_attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


class SamPatchEmbeddings(nn.Cell):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    def __init__(self, config):
        """
        Initializes an instance of the SamPatchEmbeddings class.
        
        Args:
            self: The object instance.
            config:
                An object that stores configuration parameters for the SamPatchEmbeddings class.

                - image_size: The size of the input image as a tuple or a single integer.
                - patch_size: The size of each patch as a tuple or a single integer.
                - num_channels: The number of channels in the input image.
                - hidden_size: The size of the hidden layer in the projection operation.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size, pad_mode='valid', has_bias=True)

    def construct(self, pixel_values):
        """
        Construct method in the SamPatchEmbeddings class.

        This method constructs embeddings for a given set of pixel values.

        Args:
            self: An instance of the SamPatchEmbeddings class.
            pixel_values (ndarray):
                A 4-dimensional array representing the pixel values of the input images.

                - The shape of the array should be (batch_size, num_channels, height, width).
                - The batch_size represents the number of images in the batch.
                - The num_channels represents the number of color channels in each image.
                - The height and width represent the dimensions of each image.

        Returns:
            None

        Raises:
            ValueError:
                - If the number of channels in the pixel values does not match the number of channels set in the
                configuration.
                - If the height or width of the input images do not match the expected image size defined in the model.
        """
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        return embeddings


class SamMLPBlock(nn.Cell):

    """
    This class represents a Multi-Layer Perceptron (MLP) block used in a neural network.
    It inherits from the nn.Cell class, a base class for all neural network modules in MindSpore.

    Attributes:
        lin1 (nn.Dense): The first dense layer of the MLP block.
        lin2 (nn.Dense): The second dense layer of the MLP block.
        act (function): The activation function used in the hidden layer of the MLP block.

    Methods:
        __init__: Initializes the SamMLPBlock instance.
        construct: Constructs the forward pass of the MLP block.

    """
    def __init__(self, config):
        """
        Initializes an instance of the SamMLPBlock class.

        Args:
            self: The instance of the SamMLPBlock class.
            config: An object containing configuration parameters for the MLP block.
                It is expected to have the following attributes:

                - hidden_size (int): The size of the hidden layer.
                - mlp_dim (int): The dimension of the MLP layer.
                - hidden_act (str): The activation function to be applied to the hidden layers.

        Returns:
            None.

        Raises:
            KeyError: If the 'hidden_act' attribute in the 'config' parameter does not correspond to
                any activation function in ACT2FN.
            AttributeError: If the 'config' parameter is missing any of the required attributes.
        """
        super().__init__()
        self.lin1 = nn.Dense(config.hidden_size, config.mlp_dim)
        self.lin2 = nn.Dense(config.mlp_dim, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs a multi-layer perceptron block.

        Args:
            self (SamMLPBlock): The instance of the SamMLPBlock class.
            hidden_states (mindspore.Tensor): The input hidden states tensor to be processed.

        Returns:
            mindspore.Tensor: The processed hidden states tensor after passing through the MLP block.

        Raises:
            None
        """
        hidden_states = self.lin1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.lin2(hidden_states)
        return hidden_states


# Copied from transformers.models.convnext.modeling_convnext.ConvNextLayerNorm with ConvNext->Sam
class SamLayerNorm(nn.Cell):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        """
        Initializes a new instance of the SamLayerNorm class.

        Args:
            self: The object itself.
            normalized_shape (tuple): The shape of the input tensor, indicating the size of each dimension.
            eps (float, optional): A small value to prevent division by zero when normalizing the input tensor.
                Defaults to 1e-06.
            data_format (str, optional): The format of the input tensor. Accepted values are 'channels_last' and
                'channels_first'. Defaults to 'channels_last'.

        Returns:
            None

        Raises:
            NotImplementedError: If the specified data format is not supported.

        This method initializes the SamLayerNorm object with the provided parameters. It sets the weight and bias
        parameters as trainable variables, initializes the epsilon value for numerical stability, and validates the
        data format. The normalized_shape parameter represents the size of each dimension of the input tensor.
        The eps parameter is used to avoid division by zero when normalizing the input tensor. The data_format parameter
        specifies the layout of the input tensor, which can be either 'channels_last' or 'channels_first'.
        If an unsupported data format is provided, a NotImplementedError is raised.
        """
        super().__init__()
        self.weight = Parameter(ops.ones(normalized_shape))
        self.bias = Parameter(ops.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)
        self.layer_norm = ops.LayerNorm(begin_norm_axis=-1,
                                      begin_params_axis=-1,
                                      epsilon=self.eps)

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs a layer normalization operation for the SamLayerNorm class.

        Args:
            self (SamLayerNorm): The instance of the SamLayerNorm class.
            x (mindspore.Tensor): The input tensor to be normalized.

        Returns:
            mindspore.Tensor: A normalized tensor based on the input tensor 'x'.

        Raises:
            ValueError: If the data format is not supported.
            TypeError: If the input tensor 'x' is of an unsupported type.
            RuntimeError: If any runtime error occurs during the normalization process.
        """
        if self.data_format == "channels_last":
            x, _, _ = self.layer_norm(x, self.weight, self.bias)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keep_dims=True)
            s = (x - u).pow(2).mean(1, keep_dims=True)
            x = (x - u) / ops.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SamAttention(nn.Cell):
    """
    SAM's attention layer that allows for downscaling the size of the embedding after projection to queries, keys, and
    values.
    """
    def __init__(self, config, downsample_rate=None):
        """
        Initializes a new instance of the SamAttention class.

        Args:
            self: The object itself.
            config: An object of the configuration class containing various parameters.
            downsample_rate (optional): An integer representing the downsample rate.
                If not provided, it defaults to None. (default: None)

        Returns:
            None.

        Raises:
            ValueError: If the number of attention heads is not a divisor of hidden_size.
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        downsample_rate = config.attention_downsample_rate if downsample_rate is None else downsample_rate

        self.internal_dim = config.hidden_size // downsample_rate
        self.num_attention_heads = config.num_attention_heads
        if self.internal_dim % config.num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size.")

        self.q_proj = nn.Dense(self.hidden_size, self.internal_dim)
        self.k_proj = nn.Dense(self.hidden_size, self.internal_dim)
        self.v_proj = nn.Dense(self.hidden_size, self.internal_dim)
        self.out_proj = nn.Dense(self.internal_dim, self.hidden_size)

    def _separate_heads(self, hidden_states: Tensor, num_attention_heads: int) -> Tensor:
        """
        Method to separate heads in the attention mechanism.

        Args:
            self (SamAttention): The instance of the SamAttention class.
            hidden_states (Tensor): The input hidden states tensor of shape (batch, point_batch_size, n_tokens, channel).
                This tensor represents the input feature map.
            num_attention_heads (int): The number of attention heads to split the hidden states into.

        Returns:
            Tensor: The tensor resulting from splitting the hidden states into multiple heads.
                The shape of the returned tensor is (batch * point_batch_size, num_attention_heads, n_tokens, c_per_head),
                where c_per_head is the channel size divided by the number of attention heads.

        Raises:
            None
        """
        batch, point_batch_size, n_tokens, channel = hidden_states.shape
        c_per_head = channel // num_attention_heads
        hidden_states = hidden_states.reshape(batch * point_batch_size, n_tokens, num_attention_heads, c_per_head)
        return hidden_states.swapaxes(1, 2)

    def _recombine_heads(self, hidden_states: Tensor, point_batch_size: int) -> Tensor:
        """
        Recombines the heads in the hidden states tensor for self attention in the SamAttention class.

        Args:
            self (SamAttention): The instance of the SamAttention class.
            hidden_states (Tensor): A 4D tensor representing the hidden states with shape (batch, n_heads, n_tokens, c_per_head).
                It contains the intermediate representations of the input tokens.

                - batch: The number of sequences in the batch.
                - n_heads: The number of attention heads.
                - n_tokens: The number of input tokens.
                - c_per_head: The size of each head's output.
            point_batch_size (int): The size of the batch for each point.
                Determines how the hidden states should be reshaped.

        Returns:
            Tensor: A reshaped tensor of the hidden states after recombining the heads.
                The shape of the returned tensor is (batch // point_batch_size, point_batch_size, n_tokens,
                n_heads * c_per_head).
                This reshaped tensor is used for further processing in self attention mechanisms.

        Raises:
            None
        """
        batch, n_heads, n_tokens, c_per_head = hidden_states.shape
        hidden_states = hidden_states.swapaxes(1, 2)
        return hidden_states.reshape(batch // point_batch_size, point_batch_size, n_tokens, n_heads * c_per_head)

    def construct(self, query: Tensor, key: Tensor, value: Tensor, attention_similarity: Tensor = None) -> Tensor:
        '''
        Constructs a self-attention mechanism for the SamAttention class.

        Args:
            self (SamAttention): An instance of the SamAttention class.
            query (Tensor): The query tensor of shape (batch_size, seq_length, embedding_dim) representing
                the query values.
            key (Tensor): The key tensor of shape (batch_size, seq_length, embedding_dim) representing the key values.
            value (Tensor): The value tensor of shape (batch_size, seq_length, embedding_dim) representing
                the value values.
            attention_similarity (Tensor, optional): The attention similarity tensor of shape
                (batch_size, num_attention_heads, seq_length, seq_length) representing the similarity scores between
                tokens. Defaults to None.

        Returns:
            Tensor: The output tensor of shape (batch_size, seq_length, embedding_dim) representing the attended values.

        Raises:
            None.
        '''
        # Input projections
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        point_batch_size = query.shape[1]
        # Separate into heads
        query = self._separate_heads(query, self.num_attention_heads)
        key = self._separate_heads(key, self.num_attention_heads)
        value = self._separate_heads(value, self.num_attention_heads)

        # SamAttention
        _, _, _, c_per_head = query.shape
        attn = query @ key.permute(0, 1, 3, 2)  # batch_size * point_batch_size  x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = ops.softmax(attn, axis=-1)

        if attention_similarity is not None:
            attn = attn + attention_similarity
            attn = ops.softmax(attn, axis=-1)

        # Get output
        out = attn @ value
        out = self._recombine_heads(out, point_batch_size)
        out = self.out_proj(out)

        return out


class SamTwoWayAttentionBlock(nn.Cell):

    """
    A transformer block with four layers:

    1. self-attention of sparse inputs
    2. cross attention of sparse inputs -> dense inputs
    3. mlp block on sparse inputs
    4. cross attention of dense inputs -> sparse inputs

    This class represents a SamTwoWayAttentionBlock that implements a transformer block with the above-described layers.
    It inherits from nn.Cell and is designed to handle attention mechanisms between sparse and dense inputs.

    Arguments:
        config (`SamMaskDecoderConfig`): The configuration file used to instantiate the block.
        attention_downsample_rate (*optional*, int, defaults to 2): The downsample ratio of the block used to reduce
            the inner dimension of the attention.
        skip_first_layer_pe (*optional*, bool, defaults to `False`): Whether or not to skip the addition of the
            query_point_embedding on the first layer.

    Attributes:
        hidden_size (int): The size of the hidden layers in the block.
        layer_norm_eps (float): The epsilon value for layer normalization.
        self_attn (SamAttention): The self-attention mechanism for sparse inputs.
        layer_norm1 (nn.LayerNorm): Layer normalization for the first layer.
        cross_attn_token_to_image (SamAttention): Cross-attention from token to image inputs.
        layer_norm2 (nn.LayerNorm): Layer normalization for the second layer.
        mlp (SamMLPBlock): Multi-Layer Perceptron block for processing inputs.
        layer_norm3 (nn.LayerNorm): Layer normalization for the third layer.
        layer_norm4 (nn.LayerNorm): Layer normalization for the fourth layer.
        cross_attn_image_to_token (SamAttention): Cross-attention from image to token inputs.

    Note:
        This class is specialized for two-way attention mechanisms in transformer architectures and is used to process
        sparse and dense inputs efficiently.
    """
    def __init__(self, config, attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False):
        """
        A transformer block with four layers:
            (1) self-attention of sparse inputs (2) cross attention of sparse inputs -> dense inputs (3) mlp block on
            sparse inputs (4) cross attention of dense inputs -> sparse inputs

        Arguments:
            config (`SamMaskDecoderConfig`):
                The configuration file used to instantiate the block
            attention_downsample_rate (*optionalk*, int, defaults to 2):
                The downsample ratio of the block used to reduce the inner dim of the attention.
            skip_first_layer_pe (*optional*, bool, defaults to `False`):
                Whether or not to skip the addition of the query_point_embedding on the first layer.
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps

        self.self_attn = SamAttention(config, downsample_rate=1)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, epsilon=self.layer_norm_eps)

        self.cross_attn_token_to_image = SamAttention(config, downsample_rate=attention_downsample_rate)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, epsilon=self.layer_norm_eps)

        self.mlp = SamMLPBlock(config)
        self.layer_norm3 = nn.LayerNorm(self.hidden_size, epsilon=self.layer_norm_eps)

        self.layer_norm4 = nn.LayerNorm(self.hidden_size, epsilon=self.layer_norm_eps)
        self.cross_attn_image_to_token = SamAttention(config, downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def construct(
        self,
        queries: Tensor,
        keys: Tensor,
        query_point_embedding: Tensor,
        key_point_embedding: Tensor,
        attention_similarity: Tensor,
        output_attentions: bool = False,
    ):
        """
        This method constructs a two-way attention block for processing queries and keys in a neural network model.

        Args:
            self: The instance of the class.
            queries (Tensor): The input tensor representing queries for attention mechanism.
            keys (Tensor): The input tensor representing keys for attention mechanism.
            query_point_embedding (Tensor): The embedding tensor for query points.
            key_point_embedding (Tensor): The embedding tensor for key points.
            attention_similarity (Tensor): The tensor representing attention similarity scores.
            output_attentions (bool, optional): A flag indicating whether to output attention values. Defaults to False.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing processed queries and keys, and optionally attention values.

        Raises:
            None
        """
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(query=queries, key=queries, value=queries)
        else:
            query = queries + query_point_embedding
            attn_out = self.self_attn(query=query, key=query, value=queries)
            queries = queries + attn_out
        queries = self.layer_norm1(queries)

        # Cross attention block, tokens attending to image embedding
        query = queries + query_point_embedding
        key = keys + key_point_embedding

        attn_out = self.cross_attn_token_to_image(
            query=query, key=key, value=keys, attention_similarity=attention_similarity
        )
        queries = queries + attn_out

        queries = self.layer_norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        # Cross attention block, image embedding attending to tokens
        query = queries + query_point_embedding
        key = keys + key_point_embedding

        attn_out = self.cross_attn_image_to_token(query=key, key=query, value=queries)
        keys = keys + attn_out

        keys = self.layer_norm4(keys)

        outputs = (queries, keys)

        if output_attentions:
            outputs = outputs + (attn_out,)
        else:
            outputs = outputs + (None,)

        return outputs


class SamTwoWayTransformer(nn.Cell):

    """
    This class represents a two-way transformer model called SamTwoWayTransformer. It is a subclass of nn.Cell.

    SamTwoWayTransformer is designed to perform two-way attention between point embeddings and image embeddings.
    It consists of multiple layers of SamTwoWayAttentionBlock, followed by a final attention step using SamAttention.
    The class also includes a layer normalization step.

    The main purpose of this class is to construct the transformer model and generate the outputs based on the given
    inputs. The inputs include point embeddings, image embeddings, image positional embeddings, attention similarity,
    target embeddings (optional), and various optional parameters to control the output format.

    The constructor (__init__) initializes the SamTwoWayTransformer instance with a configuration object (config) of
    type SamMaskDecoderConfig. It sets the configuration, number of hidden layers, and initializes the list of layers.

    The construct method takes the point embeddings, image embeddings, image positional embeddings, attention similarity,
    target embedding, and optional parameters as inputs. It performs the necessary computations to generate the outputs
    of the transformer model. The method supports optional arguments to control the output format, such as
    output_attentions, output_hidden_states, and return_dict. The method returns a tuple containing the queries, keys,
    and optionally, all the attention outputs.

    Please note that this class requires the image_embeddings parameter to be specified. If it is not provided,
    a ValueError will be raised.

    """
    def __init__(self, config: SamMaskDecoderConfig):
        """
        Initializes a new instance of the SamTwoWayTransformer class.

        Args:
            self: The instance of the class.
            config (SamMaskDecoderConfig): The configuration object containing the parameters for the transformer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config

        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.CellList()

        for i in range(self.num_hidden_layers):
            self.layers.append(SamTwoWayAttentionBlock(config, skip_first_layer_pe=(i == 0)))

        self.final_attn_token_to_image = SamAttention(config)
        self.layer_norm_final_attn = nn.LayerNorm(config.hidden_size)

    def construct(
        self,
        point_embeddings: Tensor,
        image_embeddings: Tensor,
        image_positional_embeddings: Tensor,
        attention_similarity: Tensor,
        target_embedding=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Constructs the SamTwoWayTransformer.

        This method initializes and constructs the SamTwoWayTransformer model by taking in various input parameters.

        Args:
            self: The object instance.
            point_embeddings (Tensor): The tensor representing the point embeddings.
            image_embeddings (Tensor): The tensor representing the image embeddings.
            image_positional_embeddings (Tensor): The tensor representing the positional embeddings of the images.
            attention_similarity (Tensor): The tensor representing the attention similarity.
            target_embedding (Optional[Tensor]): The tensor representing the target embedding (default: None).
            output_attentions (Optional[bool]): Whether to output attentions (default: None).
            output_hidden_states (Optional[bool]): Whether to output hidden states (default: None).
            return_dict (Optional[bool]): Whether to use return dict (default: None).

        Returns:
            Union[Tuple, BaseModelOutput]: The output of the SamTwoWayTransformer model.

        Raises:
            ValueError: This exception is raised if the image_embeddings parameter is not specified.

        Note:
            - The output_attentions, output_hidden_states, and return_dict parameters will take values from the self.config
              object if not explicitly provided.
            - This method performs various computations and transformations on the input tensors to construct the
              SamTwoWayTransformer model.
            - The constructed model is returned as an output.

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_attentions = ()

        if image_embeddings is None:
            raise ValueError("You have to specify an image_embedding")

        image_embeddings = image_embeddings.flatten(start_dim=2).permute(0, 2, 1).unsqueeze(1)
        image_positional_embeddings = image_positional_embeddings.flatten(start_dim=2).permute(0, 2, 1).unsqueeze(1)

        # Prepare queries
        queries = point_embeddings
        keys = image_embeddings

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            if target_embedding is not None:
                queries += target_embedding

            queries, keys, attention_outputs = layer(
                queries=queries,
                keys=keys,
                query_point_embedding=point_embeddings,
                key_point_embedding=image_positional_embeddings,
                attention_similarity=attention_similarity,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions = all_attentions + (attention_outputs,)

        # Apply the final attenion layer from the points to the image
        query = queries + point_embeddings
        key = keys + image_positional_embeddings

        attn_out = self.final_attn_token_to_image(query=query, key=key, value=keys)

        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)
        return queries, keys, all_attentions


class SamFeedForward(nn.Cell):

    """
    SamFeedForward is a class representing a feedforward neural network model with customizable parameters for input,
    hidden, and output dimensions, as well as the number of layers. The class allows for the option of applying a
    sigmoid activation function to the output layer.

    Parameters:
        input_dim (int): The dimension of the input data.
        hidden_dim (int): The dimension of the hidden layers.
        output_dim (int): The dimension of the output data.
        num_layers (int): The number of hidden layers in the network.
        sigmoid_output (bool, optional): If True, applies a sigmoid activation function to the output layer.
            Defaults to False.

    Attributes:
        num_layers (int): The number of hidden layers in the network.
        activation (nn.ReLU): The rectified linear unit (ReLU) activation function.
        proj_in (nn.Dense): The linear transformation for input data to the hidden layer.
        proj_out (nn.Dense): The linear transformation for the last hidden layer to the output data.
        layers (nn.CellList): List of Dense layers for the hidden layers in the network.

    Methods:
        construct: Constructs the feedforward network by applying linear transformations and activation functions
            to the input data through the hidden layers, and finally to the output data.

    Returns:
        hidden_states: The output data after passing through the feedforward network, with optional sigmoid activation
            applied.

    """
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False
    ):
        """
        Initializes an instance of the SamFeedForward class.

        Args:
            self: The instance of the class.
            input_dim (int): The dimension of the input data.
            hidden_dim (int): The dimension of the hidden layers.
            output_dim (int): The dimension of the output data.
            num_layers (int): The number of hidden layers in the network.
            sigmoid_output (bool, optional): Flag to indicate whether the output should be passed through a sigmoid
                activation function. Default is False.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.num_layers = num_layers
        self.activation = nn.ReLU()
        self.proj_in = nn.Dense(input_dim, hidden_dim)
        self.proj_out = nn.Dense(hidden_dim, output_dim)
        self.layers = nn.CellList([nn.Dense(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        self.sigmoid_output = sigmoid_output

    def construct(self, hidden_states):
        """
        This method constructs a feedforward neural network using the provided hidden states.

        Args:
            self (SamFeedForward): The instance of the SamFeedForward class.
            hidden_states (tensor): The input hidden states to be processed by the neural network.

        Returns:
            None.

        Raises:
            None.
        """
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        for layer in self.layers:
            hidden_states = self.activation(layer(hidden_states))

        hidden_states = self.proj_out(hidden_states)
        if self.sigmoid_output:
            hidden_states = ops.sigmoid(hidden_states)
        return hidden_states


class SamMaskDecoder(nn.Cell):

    """
    A class representing a Mask Decoder module for generating masks based on image and prompt embeddings.

    This class inherits from nn.Cell and contains methods for initializing the decoder and constructing the masks
    based on input embeddings. The decoder architecture includes components such as transformers, convolutional layers,
    embeddings, and feedforward networks to generate masks with optional attentions and predictions.

    Attributes:
        hidden_size (int): The size of the hidden layers in the decoder.
        num_multimask_outputs (int): The number of multimask outputs to be generated.
        num_mask_tokens (int): The total number of mask tokens used in the decoder.
        iou_token (nn.Embedding): Embedding layer for IOU tokens.
        mask_tokens (nn.Embedding): Embedding layer for mask tokens.
        transformer (SamTwoWayTransformer): Transformer network used in the decoding process.
        upscale_conv1 (nn.Conv2dTranspose): Transposed convolutional layer for upscaling.
        upscale_conv2 (nn.Conv2dTranspose): Additional transposed convolutional layer for upscaling.
        upscale_layer_norm (SamLayerNorm): Layer normalization applied after upscaling.
        activation (nn.GELU): Activation function used in the decoder.
        output_hypernetworks_mlps (nn.CellList): List of feedforward networks for output hypernetworks.
        iou_prediction_head (SamFeedForward): Feedforward network for IOU prediction.

    Methods:
        __init__: Initializes the Mask Decoder with the provided configuration.
        construct: Predicts masks based on input embeddings and returns the generated masks along with optional attentions.

    For more details on the functionality and usage of the Mask Decoder class, refer to the method descriptions and class attributes above.
    """
    def __init__(self, config: SamMaskDecoderConfig):
        """
        Initialize the SamMaskDecoder class.

        Args:
            self: The instance of the SamMaskDecoder class.
            config (SamMaskDecoderConfig): An instance of the SamMaskDecoderConfig class containing the configuration
                parameters for the SamMaskDecoder.
                It includes the following attributes:

                - hidden_size (int): The size of the hidden state.
                - num_multimask_outputs (int): The number of multimask outputs.
                - num_mask_tokens (int): The number of mask tokens, which is calculated as
                config.num_multimask_outputs + 1.
                - iou_token (nn.Embedding): An embedding for the intersection over union (IOU) token with a shape of
                (1, hidden_size).
                - mask_tokens (nn.Embedding): An embedding for the mask tokens with a shape of
                (num_mask_tokens, hidden_size).
                - transformer (SamTwoWayTransformer): The SamTwoWayTransformer instance.
                - upscale_conv1 (nn.Conv2dTranspose): The first convolution layer for upscaling.
                - upscale_conv2 (nn.Conv2dTranspose): The second convolution layer for upscaling.
                - upscale_layer_norm (SamLayerNorm): The layer normalization for upscaling.
                - activation (nn.GELU): The activation function.
                - output_hypernetworks_mlps (nn.CellList): A list of SamFeedForward instances for output hypernetworks.
                - iou_prediction_head (SamFeedForward): The SamFeedForward instance for IOU prediction head.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()

        self.hidden_size = config.hidden_size

        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, self.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)

        self.transformer = SamTwoWayTransformer(config)

        # should we create a new class for this?
        self.upscale_conv1 = nn.Conv2dTranspose(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2, pad_mode='valid', has_bias=True)
        self.upscale_conv2 = nn.Conv2dTranspose(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2, pad_mode='valid', has_bias=True)
        self.upscale_layer_norm = SamLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.activation = nn.GELU()

        mlps_list = []
        for _ in range(self.num_mask_tokens):
            mlps_list += [SamFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)]
        self.output_hypernetworks_mlps = nn.CellList(mlps_list)

        self.iou_prediction_head = SamFeedForward(
            self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )

    def construct(
        self,
        image_embeddings: mindspore.Tensor,
        image_positional_embeddings: mindspore.Tensor,
        sparse_prompt_embeddings: mindspore.Tensor,
        dense_prompt_embeddings: mindspore.Tensor,
        multimask_output: bool,
        output_attentions: Optional[bool] = None,
        attention_similarity: mindspore.Tensor = None,
        target_embedding: mindspore.Tensor = None,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (`mindspore.Tensor`):
                the embeddings from the image encoder
            image_positional_embedding (`mindspore.Tensor`):
                positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (`mindspore.Tensor`):
                The embeddings of the points and boxes
            dense_prompt_embeddings (`mindspore.Tensor`):
                the embeddings of the mask inputs
            multimask_output (bool):
                Whether to return multiple masks or a single mask.
            output_attentions (bool, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = ops.cat([self.iou_token.weight, self.mask_tokens.weight], axis=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = ops.cat((output_tokens, sparse_prompt_embeddings), axis=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)

        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings, attentions = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.swapaxes(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = ops.stack(hyper_in_list, axis=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        outputs = (masks, iou_pred)

        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)

        return outputs


class SamPositionalEmbedding(nn.Cell):

    """
    The SamPositionalEmbedding class represents a positional encoding module that inherits from nn.Cell.
    It provides functionality to positionally encode points normalized to the range [0,1] using sinusoidal
    and cosine functions.

    Attributes:
        scale (int): The scale value calculated as config.hidden_size // 2.
        positional_embedding (Parameter): The positional embedding parameter calculated using random values with
            specified shape and no gradient.

    Methods:
        construct: Positionally encodes normalized points and returns the encoded
            coordinates as a concatenation of sinusoidal and cosine functions.
    """
    def __init__(self, config):
        """
        Initializes an instance of the SamPositionalEmbedding class.

        Args:
            self (SamPositionalEmbedding): The current instance of the SamPositionalEmbedding class.
            config (object): The configuration object that holds various settings.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.scale = config.hidden_size // 2
        self.positional_embedding = Parameter(self.scale * ops.randn((2, config.num_pos_feats)), requires_grad=False)

    def construct(self, input_coords, input_shape=None):
        """Positionally encode points that are normalized to [0,1]."""
        coordinates = input_coords.copy()

        if input_shape is not None:
            coordinates[:, :, :, 0] = coordinates[:, :, :, 0] / input_shape[1]
            coordinates[:, :, :, 1] = coordinates[:, :, :, 1] / input_shape[0]

        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coordinates = 2 * coordinates - 1
        coordinates = coordinates.to(self.positional_embedding.dtype)
        coordinates = coordinates @ self.positional_embedding
        coordinates = 2 * np.pi * coordinates
        # outputs d_1 x ... x d_n x channel shape
        return ops.cat([ops.sin(coordinates), ops.cos(coordinates)], axis=-1)


class SamMaskEmbedding(nn.Cell):

    """
    This class represents a mask embedding module used for generating dense embeddings from input masks.
    It consists of several convolutional and normalization layers for processing the input masks and
    producing dense embeddings. The class inherits from nn.Cell.

    Attributes:
        mask_input_channels (int): Number of input channels for the mask
        activation (function): Activation function used in the module
        conv1 (nn.Conv2d): Convolutional layer 1
        conv2 (nn.Conv2d): Convolutional layer 2
        conv3 (nn.Conv2d): Convolutional layer 3
        layer_norm1 (SamLayerNorm): Layer normalization for the first layer
        layer_norm2 (SamLayerNorm): Layer normalization for the second layer

    Methods:
        construct: Processes the input masks through the convolutional and normalization layers to generate dense
            embeddings
    """
    def __init__(self, config: SamPromptEncoderConfig):
        """
        Initializes the SamMaskEmbedding object with the provided configuration.

        Args:
            self: The instance of the SamMaskEmbedding class.
            config (SamPromptEncoderConfig): An instance of the SamPromptEncoderConfig class containing the
                configuration settings for the SamMaskEmbedding.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.mask_input_channels = config.mask_input_channels // 4
        self.activation = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2, pad_mode='valid', has_bias=True)
        self.conv2 = nn.Conv2d(self.mask_input_channels, config.mask_input_channels, kernel_size=2, stride=2, pad_mode='valid', has_bias=True)
        self.conv3 = nn.Conv2d(config.mask_input_channels, config.hidden_size, kernel_size=1, pad_mode='valid', has_bias=True)
        self.layer_norm1 = SamLayerNorm(
            self.mask_input_channels, eps=config.layer_norm_eps, data_format="channels_first"
        )
        self.layer_norm2 = SamLayerNorm(
            self.mask_input_channels * 4, eps=config.layer_norm_eps, data_format="channels_first"
        )

    def construct(self, masks):
        """
        Constructs dense embeddings from masks using convolutional layers.

        Args:
            self: An instance of the SamMaskEmbedding class.
            masks: A tensor of shape (batch_size, channels, height, width) representing the input masks.

        Returns:
            None: The method modifies the state of the object by updating the dense embeddings attribute.

        Raises:
            None.

        This method applies a series of convolutional layers to the input masks to generate dense embeddings.
        The process involves the following steps:

        1. Convolution 1: Applies a 2D convolutional operation to the masks tensor, resulting in hidden states.
        2. Layer Normalization 1: Performs layer normalization on the hidden states.
        3. Activation: Applies an activation function to the normalized hidden states.
        4. Convolution 2: Applies another 2D convolutional operation to the activated hidden states.
        5. Layer Normalization 2: Performs layer normalization on the hidden states from the second convolution.
        6. Activation: Applies the activation function to the normalized hidden states from the second convolution.
        7. Convolution 3: Applies a final 2D convolutional operation to the normalized hidden states from the
        second convolution.

        After these steps, the method returns the dense embeddings.
        """
        hidden_states = self.conv1(masks)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.activation(hidden_states)
        dense_embeddings = self.conv3(hidden_states)
        return dense_embeddings


class SamPromptEncoder(nn.Cell):

    """
    A prompt encoder for sparse and dense embeddings.

    This class represents a prompt encoder that embeds different types of prompts, returning both sparse
    and dense embeddings.

    Args:
        config (SamPromptEncoderConfig): The configuration for the prompt encoder.
        shared_patch_embedding : A shared patch embedding.

    Attributes:
        shared_embedding: The shared patch embedding for the prompt encoder.
        mask_embed: The mask embedding for the prompt encoder.
        no_mask_embed: A tensor for no mask embedding.
        image_embedding_size: The size of the image embedding.
        input_image_size: The size of the input image.
        point_embed: A list of point embeddings.
        hidden_size: The hidden size for the prompt encoder.
        not_a_point_embed: The embedding for non-point prompts.

    Methods:
        _embed_points(: Embeds point prompts.
        _embed_boxes: Embeds box prompts.
        construct: Embeds different types of prompts,
        returning both sparse and dense embeddings.

    Raises:
        ValueError: If points are provided, labels must also be provided.

    """
    def __init__(self, config: SamPromptEncoderConfig, shared_patch_embedding):
        """
        Initializes a new instance of SamPromptEncoder.

        Args:
            self: The instance of the class.
            config (SamPromptEncoderConfig): An instance of SamPromptEncoderConfig containing configuration parameters.
            shared_patch_embedding: The shared patch embedding used in the encoder.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.shared_embedding = shared_patch_embedding
        self.mask_embed = SamMaskEmbedding(config)
        self.no_mask_embed = nn.Embedding(1, config.hidden_size)

        self.image_embedding_size = (config.image_embedding_size, config.image_embedding_size)
        self.input_image_size = config.image_size

        self.point_embed = nn.CellList(
            [nn.Embedding(1, config.hidden_size) for i in range(config.num_point_embeddings)]
        )
        self.hidden_size = config.hidden_size
        self.not_a_point_embed = nn.Embedding(1, config.hidden_size)

    def _embed_points(self, points: mindspore.Tensor, labels: mindspore.Tensor, pad: bool) -> mindspore.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            target_point_shape = (points.shape[0], points.shape[1], 1, points.shape[-1])
            target_labels_shape = (points.shape[0], points.shape[1], 1)
            padding_point = ops.zeros(target_point_shape)
            padding_label = -ops.ones(target_labels_shape)
            points = ops.cat([points, padding_point], axis=2)
            labels = ops.cat([labels, padding_label.type_as(labels)], axis=2)
        input_shape = (self.input_image_size, self.input_image_size)
        point_embedding = self.shared_embedding(points, input_shape)

        # torch.where and expanding the labels tensor is required by the ONNX export
        point_embedding = ops.where(labels[..., None] == -1, self.not_a_point_embed.weight, point_embedding)

        # specificed as otherwise torch.onnx.export interprets as double
        point_embedding = ops.where(
            labels[..., None] != -10,
            point_embedding,
            mindspore.tensor(0.0, dtype=point_embedding.dtype),
        )

        point_embedding = ops.where(
            (labels == 0)[:, :, :, None],
            point_embedding + self.point_embed[0].weight[None, None, :, :],
            point_embedding,
        )

        point_embedding = ops.where(
            (labels == 1)[:, :, :, None],
            point_embedding + self.point_embed[1].weight[None, None, :, :],
            point_embedding,
        )

        return point_embedding

    def _embed_boxes(self, boxes: mindspore.Tensor) -> mindspore.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        batch_size, nb_boxes = boxes.shape[:2]
        coords = boxes.reshape(batch_size, nb_boxes, 2, 2)
        input_shape = (self.input_image_size, self.input_image_size)
        corner_embedding = self.shared_embedding(coords, input_shape)
        corner_embedding[:, :, 0, :] += self.point_embed[2].weight
        corner_embedding[:, :, 1, :] += self.point_embed[3].weight
        return corner_embedding

    def construct(
        self,
        input_points: Optional[Tuple[mindspore.Tensor, mindspore.Tensor]],
        input_labels: Optional[mindspore.Tensor],
        input_boxes: Optional[mindspore.Tensor],
        input_masks: Optional[mindspore.Tensor],
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (`mindspore.Tensor`, *optional*):
                point coordinates and labels to embed.
            boxes (`mindspore.Tensor`, *optional*):
                boxes to embed
            masks (`mindspore.Tensor`, *optional*):
                masks to embed
        """
        sparse_embeddings = None
        batch_size = 1

        if input_points is not None:
            batch_size, point_batch_size = input_points.shape[:2]
            if input_labels is None:
                raise ValueError("If points are provided, labels must also be provided.")
            point_embeddings = self._embed_points(input_points, input_labels, pad=(input_boxes is None))
            sparse_embeddings = point_embeddings

        if input_boxes is not None:
            batch_size = input_boxes.shape[0]
            box_embeddings = self._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = ops.cat([sparse_embeddings, box_embeddings], axis=2)

        if input_masks is not None:
            dense_embeddings = self.mask_embed(input_masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        if sparse_embeddings is None:
            sparse_embeddings = ops.zeros((batch_size, 1, 1, self.hidden_size))

        return sparse_embeddings, dense_embeddings


class SamVisionAttention(nn.Cell):
    """Multi-head Attention block with relative position embeddings."""
    def __init__(self, config, window_size):
        """
        Initializes a SamVisionAttention object.

        Args:
            self: The object itself.
            config: An instance of a configuration class containing various parameters.
                It specifies the configuration settings for the attention mechanism.
            window_size: An integer representing the size of the window. If set to 0, the window size
                is determined based on the config's image size. It determines the size of the attention window.

        Returns:
            None

        Raises:
            ValueError: If input_size is None and use_rel_pos is True.
        """
        super().__init__()
        input_size = (
            (config.image_size // config.patch_size, config.image_size // config.patch_size)
            if window_size == 0
            else (window_size, window_size)
        )

        self.num_attention_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim**-0.5
        self.dropout = config.attention_dropout

        self.qkv = nn.Dense(config.hidden_size, config.hidden_size * 3, has_bias=config.qkv_bias)
        self.proj = nn.Dense(config.hidden_size, config.hidden_size)

        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError("Input size must be provided if using relative positional encoding.")

            # initialize relative positional embeddings
            self.rel_pos_h = Parameter(ops.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = Parameter(ops.zeros(2 * input_size[1] - 1, head_dim))

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: mindspore.Tensor) -> mindspore.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int):
                size of the query.
            k_size (int):
                size of key k.
            rel_pos (`mindspore.Tensor`):
                relative position embeddings (L, channel).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos.
        rel_pos_resized = ops.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = ops.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = ops.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]

    def add_decomposed_rel_pos(
        self,
        attn: mindspore.Tensor,
        query: mindspore.Tensor,
        rel_pos_h: mindspore.Tensor,
        rel_pos_w: mindspore.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> mindspore.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`mindspore.Tensor`):
                attention map.
            query (`mindspore.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`mindspore.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`mindspore.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`mindspore.Tensor`):
                attention map with added relative positional embeddings.
        """
        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)

        batch_size, _, dim = query.shape
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        rel_h = ops.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        rel_w = ops.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        attn = attn.reshape(batch_size, query_height, query_width, key_height, key_width)
        attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        attn = attn.reshape(batch_size, query_height * query_width, key_height * key_width)
        return attn

    def construct(self, hidden_states: mindspore.Tensor, output_attentions=False) -> mindspore.Tensor:
        """
        Method 'construct' in the class 'SamVisionAttention'.

        Args:
            self: SamVisionAttention object. Represents the instance of the SamVisionAttention class.
            hidden_states: mindspore.Tensor. Input tensor of shape (batch_size, height, width, _),
                where _ represents a dimension. Contains the hidden states to be processed.
            output_attentions: bool. Indicates whether to output the attention weights. Default is False.
                If True, the attention weights will be included in the return value.

        Returns:
            Tuple[Tensor]:
                Tuple of two elements - attn_output: mindspore.Tensor. Output tensor after attention mechanism processing.
                If output_attentions is True, the second element is attn_weights: mindspore.Tensor. Attention weights tensor.
                The return value represents the result of applying the attention mechanism on the hidden_states input.

        Raises:
            None.
        """
        batch_size, height, width, _ = hidden_states.shape
        # qkv with shape (3, batch_size, nHead, height * width, channel)
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, height * width, 3, self.num_attention_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (batch_size * nHead, height * width, channel)
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)

        attn_weights = (query * self.scale) @ key.swapaxes(-2, -1)

        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        attn_weights = ops.softmax(attn_weights, dtype=mindspore.float32, axis=-1).to(query.dtype)

        attn_probs = ops.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)

        attn_output = self.proj(attn_output)

        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)

        return outputs


class SamVisionLayer(nn.Cell):

    """
    This class represents a vision layer in the SamVision model. It inherits from the nn.Cell class and implements
    the necessary methods and functionality for performing attention-based operations on input image tokens.

    Attributes:
        layer_norm1: An instance of nn.LayerNorm which applies layer normalization to the input hidden states.
        attn: An instance of the SamVisionAttention class which performs attention computation on the hidden states.
        layer_norm2: An instance of nn.LayerNorm which applies layer normalization to the output hidden states.
        mlp: An instance of the SamMLPBlock class which applies multi-layer perceptron operations to the output hidden states.
        window_size: An integer representing the size of the attention windows.

    Methods:
        window_partition(hidden_states, window_size): Partitions the input hidden states into non-overlapping windows
            with padding if needed.
        window_unpartition(windows, window_size, padding_shape, original_shape): Unpartitions the windows into original
            sequences, removing padding.
        construct(hidden_states, output_attentions=False): Constructs the output hidden states by applying
            layer normalization, attention, and MLP operations.

    Example:
        ```python
        >>> config = Configuration()
        >>> window_size = 16
        >>> vision_layer = SamVisionLayer(config, window_size)
        >>> hidden_states = torch.randn(batch_size, height, width, channel)
        >>> output = vision_layer.construct(hidden_states)
        ```
    """
    def __init__(self, config, window_size):
        """
        Initializes a new instance of SamVisionLayer.

        Args:
            self: The instance of the class.
            config:
                A configuration object containing the hidden size and layer normalization epsilon.

                - Type: object
                - Purpose: Specifies the configuration for the vision layer.
                - Restrictions: Must contain a 'hidden_size' property and a 'layer_norm_eps' property.
            window_size:
                An integer specifying the size of the vision window.

                - Type: int
                - Purpose: Specifies the size of the vision window used for attention mechanism.
                - Restrictions: Must be a positive integer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.attn = SamVisionAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.mlp = SamMLPBlock(config)
        self.window_size = window_size

    def window_partition(self, hidden_states: mindspore.Tensor, window_size: int) -> Tuple[mindspore.Tensor, Tuple[int, int]]:
        """
        Partition into non-overlapping windows with padding if needed.

        Args:
            hidden_states (tensor): input tokens with [batch_size, height, width, channel]. window_size (int): window
                size.

        Returns:
            windows: windows after partition with [batch_size * num_windows, window_size, window_size, channel].
                (pad_height, pad_width): padded height and width before partition
        """
        batch_size, height, width, channel = hidden_states.shape

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        hidden_states = ops.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))
        pad_height, pad_width = height + pad_h, width + pad_w

        hidden_states = hidden_states.reshape(
            batch_size, pad_height // window_size, window_size, pad_width // window_size, window_size, channel
        )
        windows = hidden_states.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, channel)
        return windows, (pad_height, pad_width)

    def window_unpartition(
        self, windows: mindspore.Tensor, window_size: int, padding_shape: Tuple[int, int], original_shape: Tuple[int, int]
    ) -> mindspore.Tensor:
        """
        Window unpartition into original sequences and removing padding.

        Args:
            hidden_states (tensor):
                input tokens with [batch_size * num_windows, window_size, window_size, channel].
            window_size (int):
                window size.
            padding_shape (Tuple):
                padded height and width (pad_height, pad_width).
            original_shape (Tuple): original height and width (height, width) before padding.

        Returns:
            hidden_states: unpartitioned sequences with [batch_size, height, width, channel].
        """
        pad_height, pad_width = padding_shape
        height, width = original_shape
        batch_size = windows.shape[0] // (pad_height * pad_width // window_size // window_size)
        hidden_states = windows.reshape(
            batch_size, pad_height // window_size, pad_width // window_size, window_size, window_size, -1
        )
        hidden_states = (
            hidden_states.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, pad_height, pad_width, -1)
        )

        hidden_states = hidden_states[:, :height, :width, :]
        return hidden_states

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        '''
        This method constructs the SamVisionLayer by applying attention mechanism and multi-layer perceptron (MLP) to
        the input hidden states.

        Args:
            self: The instance of the SamVisionLayer class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.
                It is expected to be a tensor of shape (batch_size, sequence_length, hidden_size).
            output_attentions (Optional[bool]): A flag indicating whether to output the attention weights.
                Default is False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the output hidden states tensor after applying attention
                mechanism and MLP.

        Raises:
            None
        '''
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        # Window partition
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)

        hidden_states, attn_weights = self.attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )
        # Reverse window partition
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))

        hidden_states = residual + hidden_states
        layernorm_output = self.layer_norm2(hidden_states)
        hidden_states = hidden_states + self.mlp(layernorm_output)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SamVisionNeck(nn.Cell):

    """
    This class represents the SamVisionNeck module, which is a component of a vision model that performs operations
    on hidden states.

    SamVisionNeck inherits from the nn.Cell class and includes two convolutional layers with layer normalization.
    The hidden states are passed through these layers to extract relevant features.

    Attributes:
        config (SamVisionConfig): The configuration object that defines the parameters for the SamVisionNeck module.
        conv1 (nn.Conv2d): The first convolutional layer that processes the hidden states.
        layer_norm1 (SamLayerNorm): The first layer normalization module that normalizes the output of the
            first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer that further processes the hidden states.
        layer_norm2 (SamLayerNorm): The second layer normalization module that normalizes the output of the
            second convolutional layer.

    Methods:
        __init__: Initializes a new instance of the SamVisionNeck class with the given configuration.
        construct: Processes the hidden states through the convolutional and layer normalization layers.

    """
    def __init__(self, config: SamVisionConfig):
        """
        Initialize the SamVisionNeck class.

        Args:
            self: The instance of the class.
            config (SamVisionConfig): An instance of SamVisionConfig containing the configuration for the SamVisionNeck.
                It defines the parameters required for the layers in the network.

        Returns:
            None.

        Raises:
            ValueError: If the configuration provided is invalid or incomplete.
            TypeError: If the configuration data type is not as expected.
            RuntimeError: If there is an issue during the initialization of the layers.
        """
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, has_bias=False, pad_mode='valid')
        self.layer_norm1 = SamLayerNorm(config.output_channels, data_format="channels_first")
        self.conv2 = nn.Conv2d(config.output_channels, config.output_channels, kernel_size=3, padding=1, has_bias=False, pad_mode='pad')
        self.layer_norm2 = SamLayerNorm(config.output_channels, data_format="channels_first")

    def construct(self, hidden_states):
        """Constructs the hidden states in the SamVisionNeck class.

        This method takes in two parameters: self and hidden_states. The hidden_states parameter represents the
        input hidden states and should be a tensor. The purpose of this parameter is to provide the input
        for constructing the hidden states. There are no restrictions on the shape or size of the hidden_states tensor.

        The method performs the following operations on the hidden_states:

        1. Permute the dimensions of the hidden_states tensor using the permute() function,
        with the dimensions permuted as (0, 3, 1, 2).
        2. Apply the conv1 layer to the permuted hidden_states tensor.
        3. Apply the layer_norm1 layer to the output of the conv1 layer.
        4. Apply the conv2 layer to the output of the layer_norm1 layer.
        5. Apply the layer_norm2 layer to the output of the conv2 layer.

        The method returns the final constructed hidden states tensor.

        Args:
            self: An instance of the SamVisionNeck class.
            hidden_states: A tensor representing the input hidden states.

        Returns:
            hidden_states: The method returns the constructed hidden states as a tensor.

        Raises:
            None.
        """
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states


class SamVisionEncoder(nn.Cell):

    """
    The SamVisionEncoder class represents a vision encoder for processing image data using the SAM
    (Self-Attention Model) architecture. It inherits from the nn.Cell class and is designed to be used within the
    MindSpore framework for deep learning applications.

    The class initializes with a SamVisionConfig object and sets various attributes based on the provided configuration.
    It includes methods for retrieving input embeddings and constructing the encoder output based on the input pixel
    values. The construction process involves passing the input through the patch embeddings, applying positional
    embeddings if configured, processing the input through multiple vision layers, and finally passing the output
    through a vision neck module.

    The class also provides options for controlling the output of attentions and hidden states, as well as the ability
    to return the output as a dictionary. Additionally, it supports gradient checkpointing during training for efficient
    memory usage.

    Overall, the SamVisionEncoder class encapsulates the functionality for encoding image data using the SAM architecture,
    providing a flexible and configurable interface for vision processing tasks within the MindSpore framework.
    """
    def __init__(self, config: SamVisionConfig):
        """
        Initializes a SamVisionEncoder object.

        Args:
            self: The object itself.
            config (SamVisionConfig): An instance of SamVisionConfig containing configuration parameters.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        self.patch_embed = SamPatchEmbeddings(config)

        self.pos_embed = None
        if config.use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = Parameter(
                ops.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )

        self.layers = nn.CellList()
        for i in range(config.num_hidden_layers):
            layer = SamVisionLayer(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
            )
            self.layers.append(layer)

        self.neck = SamVisionNeck(config)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings of the SamVisionEncoder.

        Args:
            self: An instance of the SamVisionEncoder class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the input embeddings used by the SamVisionEncoder.
        The input embeddings are obtained from the patch embedding process performed by the 'patch_embed' method.
        """
        return self.patch_embed

    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SamVisionEncoderOutput]:
        """
        Constructs the SamVisionEncoder.

        Args:
            self (SamVisionEncoder): The instance of the SamVisionEncoder class.
            pixel_values (Optional[mindspore.Tensor]): The input pixel values. Default is None.
            output_attentions (Optional[bool]): Whether to output attentions. Default is None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Whether to return a dictionary. Default is None.

        Returns:
            Union[Tuple, SamVisionEncoderOutput]: The output of the SamVisionEncoder.
            If return_dict is False, returns a tuple containing the hidden states,
            all_hidden_states (if output_hidden_states is True), and all_self_attentions (if output_attentions is True).
            If return_dict is True, returns a SamVisionEncoderOutput object containing the last hidden state,
            all_hidden_states, and all_self_attentions.

        Raises:
            ValueError: If pixel_values is None.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.neck(hidden_states)

        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            return outputs

        return SamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class SamPreTrainedModel(PreTrainedModel):

    """A class representing a pretrained model in the Sam library.

    This class, 'SamPreTrainedModel', is a subclass of the 'PreTrainedModel' class in the Sam library.
    It provides functionality for initializing the weights of different types of cells in the model.
    The weights are initialized using a normal distribution with a standard deviation specified in the configuration.
    If a bias term is present in the cell, it is initialized to zeros.
    For embedding cells, the weights are initialized using a normal distribution and a padding index,
    if provided, is set to zero.

    Attributes:
        config (PretrainedConfig): The configuration object for the pretrained model.

    Methods:
        _init_weights: Initializes the weights of different types of cells in the model.

    Note:
        This class assumes that the model is built using cells from the Sam library, such as nn.Dense, nn.Conv2d,
        nn.Conv2dTranspose, and nn.Embedding.

    Example:
        ```python
        >>> model = SamPreTrainedModel(config)
        >>> model._init_weights(cell)
        ```
    """
    config_class = SamConfig
    base_model_prefix = "sam"
    main_input_name = "pixel_values"

    def _init_weights(self, cell):
        '''
        This method initializes the weights and biases of the provided cell based on the specified configuration.

        Args:
            self (SamPreTrainedModel): The instance of the SamPreTrainedModel class.
            cell: The neural network cell for which the weights and biases are initialized.
                It can be an instance of nn.Dense, nn.Conv2d, nn.Conv2dTranspose, or nn.Embedding.

        Returns:
            None.

        Raises:
            TypeError: If the cell parameter is not an instance of supported cell types
                (nn.Dense, nn.Conv2d, nn.Conv2dTranspose, or nn.Embedding).
            ValueError: If the cell parameter is an instance of nn.Embedding and the padding index
                (cell.padding_idx) is out of range.
        '''
        std = self.config.initializer_range
        if isinstance(cell, (nn.Dense, nn.Conv2d, nn.Conv2dTranspose)):
            cell.weight.initialize(Normal(std))
            if cell.bias is not None:
                cell.bias.initialize('zeros')
        elif isinstance(cell, nn.Embedding):
            data = np.random.normal(0.0, std, cell.weight.shape)
            if cell.padding_idx is not None:
                data[cell.padding_idx] = 0
            cell.weight.set_data(Tensor(data, cell.weight.dtype))


class SamModel(SamPreTrainedModel):

    """
    The `SamModel` class is a Python class that represents a model for image segmentation tasks.
    It is a subclass of the `SamPreTrainedModel` class.

    The `SamModel` class is typically used for image segmentation tasks. An example of how to use the `SamModel`
    class is provided in the docstring.

    Note:
        This docstring provides an overview of the `SamModel` class and its methods. For more detailed information
        on the parameters and return types of each method, please refer to the method docstrings.
    """
    _tied_weights_keys = ["prompt_encoder.shared_embedding.positional_embedding"]

    def __init__(self, config):
        """
        Initializes a new instance of the SamModel class.

        Args:
            self (SamModel): The current instance of the SamModel class.
            config (object): Configuration object containing various settings for the model.

        Returns:
            None.

        Raises:
            TypeError: If the provided 'config' parameter is not of type 'object'.
            ValueError: If the 'config' parameter is missing required settings or contains invalid values.
            RuntimeError: If any unexpected runtime error occurs during initialization.
        """
        super().__init__(config)
        self.shared_image_embedding = SamPositionalEmbedding(config.vision_config)

        self.vision_encoder = SamVisionEncoder(config.vision_config)
        self.prompt_encoder = SamPromptEncoder(config.prompt_encoder_config, self.shared_image_embedding)
        self.mask_decoder = SamMaskDecoder(config.mask_decoder_config)

        self.post_init()

    def get_input_embeddings(self):
        """
        This method 'get_input_embeddings' in the class 'SamModel' retrieves the input embeddings from the vision encoder.

        Args:
            self: SamModel instance. Represents the current instance of the class.

        Returns:
            None: This method returns None as it simply retrieves the input embeddings from the vision encoder.

        Raises:
            None
        """
        return self.vision_encoder.get_input_embeddings()

    def get_image_wide_positional_embeddings(self):
        """
        This method calculates wide positional embeddings for an image in the SamModel class.

        Args:
            self: An instance of the SamModel class. It is used to access configuration parameters and shared image embeddings.

        Returns:
            positional_embedding (torch.Tensor): A tensor representing the positional embeddings for the image.
                The tensor is permuted and unsqueezed before returning.

        Raises:
            None
        """
        size = self.config.prompt_encoder_config.image_embedding_size
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        grid = ops.ones((size, size), dtype=target_dtype)
        y_embed = grid.cumsum(axis=0) - 0.5
        x_embed = grid.cumsum(axis=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(ops.stack([x_embed, y_embed], axis=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    def get_image_embeddings(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns the image embeddings by passing the pixel values through the vision encoder.

        Args:
            pixel_values (`mindspore.Tensor` of shape `(batch_size, num_channels, height, width)`):
                Input pixel values
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        vision_output = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeddings = vision_output[0]
        return image_embeddings

    def get_prompt_embeddings(
        self,
        input_points: Optional[mindspore.Tensor] = None,
        input_labels: Optional[mindspore.Tensor] = None,
        input_boxes: Optional[mindspore.Tensor] = None,
        input_masks: Optional[mindspore.Tensor] = None,
    ):
        r"""
        Returns the prompt embeddings by passing the input points, labels, boxes and masks through the prompt encoder.

        Args:
            input_points (`mindspore.Tensor` of shape `(batch_size, point_batch_size, num_points_per_image, 2)`):
                Optional input points for the prompt encoder. The padding of the point is automatically done by the
                processor. `point_batch_size` refers to the number of masks that we want the model to predict per
                point. The model will output `point_batch_size` times 3 masks in total.
            input_labels (`mindspore.Tensor` of shape `(batch_size, point_batch_size, num_points_per_image)`):
                Optional input labels for the prompt encoder. The padding of the labels is automatically done by the
                processor, or can be fed by the user.
            input_boxes (`mindspore.Tensor` of shape `(batch_size, num_boxes_per_image, 4)`):
                Optional input boxes for the prompt encoder. The padding of the boxes is automatically done by the
                processor. users can also pass manually the input boxes.
            input_masks (`mindspore.Tensor` of shape `(batch_size, image_size, image_size)`):
                Optional input masks for the prompt encoder.
        """
        prompt_output = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        return prompt_output

    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        input_points: Optional[mindspore.Tensor] = None,
        input_labels: Optional[mindspore.Tensor] = None,
        input_boxes: Optional[mindspore.Tensor] = None,
        input_masks: Optional[mindspore.Tensor] = None,
        image_embeddings: Optional[mindspore.Tensor] = None,
        multimask_output: bool = True,
        attention_similarity: Optional[mindspore.Tensor] = None,
        target_embedding: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> List[Dict[str, mindspore.Tensor]]:
        r"""

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoModel, AutoProcessor
            ...
            >>> model = AutoModel.from_pretrained("facebook/sam-vit-base")
            >>> processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")
            ...
            >>> img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
            >>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
            >>> input_points = [[[400, 650]]]  # 2D location of a window on the car
            >>> inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")
            ...
            >>> # Get segmentation mask
            >>> outputs = model(**inputs)
            ...
            >>> # Postprocess masks
            >>> masks = processor.post_process_masks(
            ...     outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
            ... )
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and image_embeddings is None:
            raise ValueError("Either pixel_values or image_embeddings must be provided.")

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

        if input_points is not None and len(input_points.shape) != 4:
            raise ValueError(
                "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
                " got {}.".format(input_points.shape),
            )
        if input_boxes is not None and len(input_boxes.shape) != 3:
            raise ValueError(
                "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
                " got {}.".format(input_boxes.shape),
            )
        if input_points is not None and input_boxes is not None:
            point_batch_size = input_points.shape[1]
            box_batch_size = input_boxes.shape[1]
            if point_batch_size != box_batch_size:
                raise ValueError(
                    "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                        point_batch_size, box_batch_size
                    )
                )

        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        # repeat with batch size
        batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        vision_attentions = None
        vision_hidden_states = None

        if pixel_values is not None:
            vision_outputs = self.vision_encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeddings = vision_outputs[0]

            if output_hidden_states:
                vision_hidden_states = vision_outputs[1]
            if output_attentions:
                vision_attentions = vision_outputs[-1]

        if input_points is not None and input_labels is None:
            input_labels = ops.ones_like(input_points[:, :, :, 0], dtype=mindspore.int32)

        if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
            raise ValueError(
                "The batch size of the image embeddings and the input points must be the same. ",
                "Got {} and {} respectively.".format(image_embeddings.shape[0], input_points.shape[0]),
                " if you want to pass multiple points for the same image, make sure that you passed ",
                " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
                " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
            )

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )

        if not return_dict:
            output = (iou_predictions, low_res_masks)
            if output_hidden_states:
                output = output + (vision_hidden_states,)

            if output_attentions:
                output = output + (vision_attentions, mask_decoder_attentions)
            return output

        return SamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
        )

__all__ = [
    "SamModel",
    "SamPreTrainedModel",
]
