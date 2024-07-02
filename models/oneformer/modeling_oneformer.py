# coding=utf-8
# Copyright 2022 SHI Labs and The HuggingFace Inc. team. All rights reserved.
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
"""MindSpore OneFormer model."""

import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import mindspore
from mindspore import ops, nn, Tensor, Parameter
from mindspore.common.initializer import initializer, Normal, XavierUniform, TruncatedNormal

from mindnlp.modules.functional import finfo
from ....amp import autocast

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ....utils import (
    ModelOutput,
    is_scipy_available,
    logging,
    requires_backends,
)
from ...backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig


logger = logging.get_logger(__name__)


if is_scipy_available():
    from scipy.optimize import linear_sum_assignment


def _get_clones(module, N):
    return nn.CellList([copy.deepcopy(module) for i in range(N)])


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.multi_scale_deformable_attention
def multi_scale_deformable_attention(
    value: Tensor, value_spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([height.item() * width.item() for height, width in value_spatial_shapes], axis=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = (
            value_list[level_id].flatten(start_dim=2).swapaxes(1, 2).reshape(batch_size * num_heads, hidden_dim, height.item(), width.item())
        )
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        #sampling_grid_l_ = sampling_grids[:, :, :, level_id].swapaxes(1, 2).flatten(0, 1)
        tmp = sampling_grids[:, :, :, level_id, :, :].swapaxes(1, 2)
        B, H, *others = tmp.shape
        sampling_grid_l_ = tmp.reshape([B*H, *others])
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        sampling_value_l_ = ops.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.swapaxes(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (ops.stack(sampling_value_list, axis=-2).flatten(start_dim=-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.swapaxes(1, 2)


# Copied from transformers.models.maskformer.modeling_maskformer.dice_loss
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    r"""
    Compute the DICE loss, similar to generalized IOU for masks as follows:

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x \cap y }{x \cup y + 1}} $$

    In practice, since `labels` is a binary mask, (only 0s and 1s), dice can be computed as follow

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x * y }{x + y + 1}} $$

    Args:
        inputs (`mindspore.Tensor`):
            A tensor representing a mask.
        labels (`mindspore.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_masks (`int`):
            The number of masks present in the current batch, used for normalization.

    Returns:
        `mindspore.Tensor`: The computed loss.
    """
    probs = inputs.sigmoid().flatten(start_dim=1)
    numerator = 2 * (probs * labels).sum(-1)
    denominator = probs.sum(-1) + labels.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() / num_masks
    return loss


# Copied from transformers.models.mask2former.modeling_mask2former.sigmoid_cross_entropy_loss
def sigmoid_cross_entropy_loss(inputs: mindspore.Tensor, labels: mindspore.Tensor, num_masks: int) -> mindspore.Tensor:
    r"""
    Args:
        inputs (`mindspore.Tensor`):
            A float tensor of arbitrary shape.
        labels (`mindspore.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`mindspore.Tensor`): The computed loss.
    """
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss = criterion(inputs, labels)

    loss = cross_entropy_loss.mean(1).sum() / num_masks
    return loss


# Copied from transformers.models.maskformer.modeling_maskformer.pair_wise_dice_loss
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """
    A pair wise version of the dice loss, see `dice_loss` for usage.

    Args:
        inputs (`mindspore.Tensor`):
            A tensor representing a mask
        labels (`mindspore.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        `mindspore.Tensor`: The computed loss between each pairs.
    """
    inputs = inputs.sigmoid().flatten(start_dim=1)
    numerator = 2 * ops.matmul(inputs, labels.T)
    # using broadcasting to get a [num_queries, NUM_CLASSES] matrix
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


# Copied from transformers.models.mask2former.modeling_mask2former.pair_wise_sigmoid_cross_entropy_loss
def pair_wise_sigmoid_cross_entropy_loss(inputs: mindspore.Tensor, labels: mindspore.Tensor) -> mindspore.Tensor:
    r"""
    A pair wise version of the cross entropy loss, see `sigmoid_cross_entropy_loss` for usage.

    Args:
        inputs (`mindspore.Tensor`):
            A tensor representing a mask.
        labels (`mindspore.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`mindspore.Tensor`): The computed loss between each pairs.
    """

    height_and_width = inputs.shape[1]

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss_pos = criterion(inputs, ops.ones_like(inputs))
    cross_entropy_loss_neg = criterion(inputs, ops.zeros_like(inputs))

    loss_pos = ops.matmul(cross_entropy_loss_pos / height_and_width, labels.T)
    loss_neg = ops.matmul(cross_entropy_loss_neg / height_and_width, (1 - labels).T)
    loss = loss_pos + loss_neg
    return loss


# Copied from transformers.models.mask2former.modeling_mask2former.sample_point
def sample_point(
    input_features: mindspore.Tensor, point_coordinates: mindspore.Tensor, add_dim=False, **kwargs
) -> mindspore.Tensor:
    """
    A wrapper around `ops.grid_sample` to support 3D point_coordinates tensors.

    Args:
        input_features (`mindspore.Tensor` of shape (batch_size, channels, height, width)):
            A tensor that contains features map on a height * width grid
        point_coordinates (`mindspore.Tensor` of shape (batch_size, num_points, 2) or (batch_size, grid_height, grid_width,:
        2)):
            A tensor that contains [0, 1] * [0, 1] normalized point coordinates
        add_dim (`bool`):
            boolean value to keep track of added dimension

    Returns:
        point_features (`mindspore.Tensor` of shape (batch_size, channels, num_points) or (batch_size, channels,
        height_grid, width_grid):
            A tensor that contains features for points in `point_coordinates`.
    """
    if point_coordinates.dim() == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(2)

    # use nn.function.grid_sample to get features for points in `point_coordinates` via bilinear interpolation
    point_features = ops.grid_sample(input_features, 2.0 * point_coordinates - 1.0, **kwargs)
    if add_dim:
        point_features = point_features.squeeze(3)

    return point_features


# Refactored from https://github.com/SHI-Labs/OneFormer/blob/33ebb56ed34f970a30ae103e786c0cb64c653d9a/oneformer/modeling/matcher.py#L93
class OneFormerHungarianMatcher(nn.Cell):
    def __init__(
        self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0, num_points: int = 12544
    ):
        """This class computes an assignment between the labels and the predictions of the network.

        For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more
        predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are
        un-matched (and thus treated as non-objects).

        Params:
            cost_class (float, *optional*, defaults to 1.0):
                This is the relative weight of the classification error in the matching cost.
            cost_mask (float, *optional*,  defaults to 1.0):
                This is the relative weight of the sigmoid ce loss of the binary mask in the matching cost.
            cost_dice (float, *optional*, defaults to 1.0):
                This is the relative weight of the dice loss of the binary mask in the matching cost
            num_points (int, *optional*, defaults to 12544):
                Number of points to be sampled for dice and mask loss matching cost.
        """
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs cant be 0")
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points

    # @ops.no_grad()
    def construct(self, masks_queries_logits, class_queries_logits, mask_labels, class_labels) -> List[Tuple[Tensor]]:
        """Performs the matching

        Params:
            masks_queries_logits (`mindspore.Tensor`):
                A tensor` of dim `batch_size, num_queries, num_labels` with the
                  classification logits.
            class_queries_logits (`mindspore.Tensor`):
                A tensor` of dim `batch_size, num_queries, height, width` with the
                  predicted masks.

            class_labels (`mindspore.Tensor`):
                A tensor` of dim `num_target_boxes` (where num_target_boxes is the number
                  of ground-truth objects in the target) containing the class labels.
            mask_labels (`mindspore.Tensor`):
                A tensor` of dim `num_target_boxes, height, width` containing the target
                  masks.

        Returns:
            `List[Tuple[Tensor]]`: A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected labels (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_targets).
        """
        indices: List[Tuple[np.array]] = []

        num_queries = class_queries_logits.shape[1]

        preds_masks = masks_queries_logits
        preds_probs = class_queries_logits
        # iterate through batch size
        for pred_probs, pred_mask, target_mask, labels in zip(preds_probs, preds_masks, mask_labels, class_labels):
            pred_probs = ops.softmax(pred_probs, -1)
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -pred_probs[:, labels]

            pred_mask = pred_mask[:, None]
            target_mask = target_mask[:, None]

            # all masks share the same set of points for efficient matching!
            point_coords = ops.rand(1, self.num_points, 2)

            # get ground truth labels
            target_mask = sample_point(
                target_mask,
                point_coords.repeat(target_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            pred_mask = sample_point(
                pred_mask,
                point_coords.repeat(pred_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                pred_mask = pred_mask.float()
                target_mask = target_mask.float()

                # compute the sigmoid ce loss
                cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
                # Compute the dice loss
                cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
                # final cost matrix
                cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
                cost_matrix = cost_matrix.reshape(num_queries, -1)
                # do the assigmented using the hungarian algorithm in scipy
                assigned_indices: Tuple[np.array] = linear_sum_assignment(cost_matrix)
                indices.append(assigned_indices)

        # It could be stacked in one tensor
        matched_indices = [
            (Tensor(i, dtype=mindspore.int64), Tensor(j, dtype=mindspore.int64)) for i, j in indices
        ]
        return matched_indices


class OneFormerLoss(nn.Cell):
    def __init__(
        self,
        num_classes: int,
        matcher: OneFormerHungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        contrastive_temperature: float = None,
    ):
        """
        This class computes the losses using the class predictions, mask predictions and the contrastive queries.

        Oneformer calculates the classification CE loss on the class predictions. Mask predictions are used for
        calculating the binary CE loss and dice loss. The contrastive queries are used for calculating the contrastive
        loss.

        Args:
            num_labels (`int`):
                The number of classes.
            matcher (`OneFormerHungarianMatcher`):
                A torch module that computes the assigments between the predictions and labels.
            weight_dict (`Dict[str, float]`):
                A dictionary of weights to be applied to the different losses.
            eos_coef (`float`):
                Weight to apply to the null class.
            num_points (`int`):
                Number of points to be sampled for dice and mask loss calculations.
            oversample_ratio (`float`):
                Required for pointwise loss calculation.
            importance_sample_ratio (`float`):
                Required for pointwise loss calculation.
            contrastive_temperature (`float`):
                Temperature for scaling the contrastive logits.
        """
        requires_backends(self, ["scipy"])
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = ops.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.empty_weight = empty_weight

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.contrastive_temperature = contrastive_temperature
        if self.contrastive_temperature is not None:
            self.logit_scale = Parameter(mindspore.tensor([np.log(1 / contrastive_temperature)]))

    def _max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        # get the maximum size in the batch
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        batch_size = len(tensors)
        # compute finel size
        batch_shape = [batch_size] + max_size
        b, _, h, w = batch_shape
        # get metadata
        dtype = tensors[0].dtype
        padded_tensors = ops.zeros(batch_shape, dtype=dtype)
        padding_masks = ops.ones((b, h, w), dtype=mindspore.bool_)
        # pad the tensors to the size of the biggest one
        for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]] = Tensor.copy(tensor)
            padding_mask[: tensor.shape[1], : tensor.shape[2]] = False

        return padded_tensors, padding_masks

    def loss_contrastive(self, contrastive_queries_logits: Tensor, text_queries: Tensor):
        """Compute the query-text contrastive loss.

        Args:
            contrastive_queries_logits (`mindspore.Tensor`):
                A tensor of shape `batch_size, num_queries, hidden_dim`
            text_queries (`mindspore.Tensor`):
                A tensor of shape `batch_size, num_queries, hidden_dim`
        Returns:
            `Dict[str, Tensor]`: A dict of `mindspore.Tensor` containing the following key:
            - **loss_contrastive** -- The query-text contrastive loss computed using task-guided queries
                                    and text queries derived from input text list.
        """

        image_queries = contrastive_queries_logits.float()

        # [batch_size, hidden_dim]
        normalize = lambda x: x / ops.norm(x, dim=-1, keepdim=True)  # pylint: disable=unnecessary-lambda-assignment
        image_queries = normalize(image_queries.flatten(start_dim=1))
        text_queries = normalize(text_queries.flatten(start_dim=1))

        logit_scale = ops.clamp(self.logit_scale.exp(), max=100)

        logits_per_text = ops.matmul(text_queries, image_queries.t()) * logit_scale
        logits_per_img = logits_per_text.t()

        loss_img = ops.cross_entropy(
            logits_per_img, ops.arange(len(logits_per_img))
        )
        loss_text = ops.cross_entropy(
            logits_per_text, ops.arange(len(logits_per_text))
        )

        loss_contrastive = loss_img + loss_text

        losses = {"loss_contrastive": loss_contrastive}
        return losses

    def loss_labels(
        self, class_queries_logits: Tensor, class_labels: List[Tensor], indices: Tuple[np.array]
    ) -> Dict[str, Tensor]:
        """Compute the losses related to the labels using cross entropy.

        Args:
            class_queries_logits (`mindspore.Tensor`):
                A tensor of shape `batch_size, num_queries, num_labels`
            class_labels (`List[mindspore.Tensor]`):
                List of class labels of shape `(labels)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.

        Returns:
            `Dict[str, Tensor]`: A dict of `mindspore.Tensor` containing the following key:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
        """
        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        idx = self._get_predictions_permutation_indices(indices)

        # shape = (batch_size, num_queries)
        target_classes_o = ops.cat([target[j] for target, (_, j) in zip(class_labels, indices)])
        # shape = (batch_size, num_queries)
        target_classes = ops.full(
            (batch_size, num_queries), fill_value=self.num_classes, dtype=mindspore.int32
        )
        target_classes[idx] = target_classes_o
        # permute pred_logits (batch_size, num_queries, num_labels) -> (batch_size, num_labels, num_queries)
        pred_logits_transposed = pred_logits.swapaxes(1, 2)
        loss_ce = criterion(pred_logits_transposed, target_classes)
        losses = {"loss_cross_entropy": loss_ce}
        return losses

    def loss_masks(
        self, masks_queries_logits: Tensor, mask_labels: List[Tensor], indices: Tuple[np.array], num_masks: int
    ) -> Dict[str, Tensor]:
        """Compute the losses related to the masks using focal and dice loss.

        Args:
            masks_queries_logits (`mindspore.Tensor`):
                A tensor of shape `batch_size, num_queries, height, width`
            mask_labels (`mindspore.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.
            num_masks (`int)`:
                The number of masks, used for normalization.

        Returns:
            `Dict[str, Tensor]`: A dict of `mindspore.Tensor` containing two keys:
            - **loss_mask** -- The loss computed using sigmoid ce loss on the predicted and ground truth masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
        """
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        # shape (batch_size * num_queries, height, width)
        pred_masks = masks_queries_logits[src_idx]
        # shape (batch_size, num_queries, height, width)
        # pad all and stack the targets to the num_labels dimension
        # upsample predictions to the target size, we have to add one dim to use interpolate
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]

        # sample point_coords
        point_coords = self.sample_points_using_uncertainty(
            pred_masks,
            self.calculate_uncertainty,
            self.num_points,
            self.oversample_ratio,
            self.importance_sample_ratio,
        )
        # get ground-truth labels
        point_labels = sample_point(target_masks, point_coords, align_corners=False).squeeze(1)

        point_logits = sample_point(pred_masks, point_coords, align_corners=False).squeeze(1)

        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del pred_masks
        del target_masks
        return losses

    # Copied from transformers.models.mask2former.modeling_mask2former.Mask2FormerLoss.calculate_uncertainty
    def calculate_uncertainty(self, logits: mindspore.Tensor) -> mindspore.Tensor:
        """
        In Mask2Former paper, uncertainty is estimated as L1 distance between 0.0 and the logit prediction in 'logits'
        for the foreground class in `classes`.

        Args:
            logits (`mindspore.Tensor`):
            A tensor of shape (R, 1, ...) for class-specific or class-agnostic, where R is the total number of predicted masks in all images and C is:
            the number of foreground classes. The values are logits.

        Returns:
            scores (`mindspore.Tensor`): A tensor of shape (R, 1, ...) that contains uncertainty scores with the most
            uncertain locations having the highest uncertainty score.
        """
        uncertainty_scores = -(ops.abs(logits))
        return uncertainty_scores

    # Copied from transformers.models.mask2former.modeling_mask2former.Mask2FormerLoss.sample_points_using_uncertainty
    def sample_points_using_uncertainty(
        self,
        logits: mindspore.Tensor,
        uncertainty_function,
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
    ) -> mindspore.Tensor:
        """
        This function is meant for sampling points in [0, 1] * [0, 1] coordinate space based on their uncertainty. The
        uncertainty is calculated for each point using the passed `uncertainty function` that takes points logit
        prediction as input.

        Args:
            logits (`float`):
                Logit predictions for P points.
            uncertainty_function:
                A function that takes logit predictions for P points and returns their uncertainties.
            num_points (`int`):
                The number of points P to sample.
            oversample_ratio (`int`):
                Oversampling parameter.
            importance_sample_ratio (`float`):
                Ratio of points that are sampled via importance sampling.

        Returns:
            point_coordinates (`mindspore.Tensor`):
                Coordinates for P sampled points.
        """

        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)

        # Get random point coordinates
        point_coordinates = ops.rand(num_boxes, num_points_sampled, 2)
        # Get sampled prediction value for the point coordinates
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        # Calculate the uncertainties based on the sampled prediction values of the points
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        idx = ops.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * ops.arange(num_boxes, dtype=mindspore.int64)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

        if num_random_points > 0:
            point_coordinates = ops.cat(
                [point_coordinates, ops.rand(num_boxes, num_random_points, 2)],
                axis=1,
            )
        return point_coordinates

    def _get_predictions_permutation_indices(self, indices):
        # permute predictions following indices
        batch_indices = ops.cat([ops.full_like(src, i) for i, (src, _) in enumerate(indices)])
        predictions_indices = ops.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices

    def _get_targets_permutation_indices(self, indices):
        # permute labels following indices
        batch_indices = ops.cat([ops.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        target_indices = ops.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices

    def construct(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        contrastive_queries_logits: Tensor,
        mask_labels: List[Tensor],
        class_labels: List[Tensor],
        text_queries: Tensor,
        auxiliary_predictions: Optional[Dict[str, Tensor]] = None,
        calculate_contrastive_loss: bool = True,
    ) -> Dict[str, Tensor]:
        """
        This performs the loss computation.

        Args:
            masks_queries_logits (`mindspore.Tensor`):
                A tensor of shape `batch_size, num_queries, height, width`
            class_queries_logits (`mindspore.Tensor`):
                A tensor of shape `batch_size, num_queries, num_labels`
            contrastive_queries_logits (`mindspore.Tensor`):
                A tensor of shape `batch_size, num_queries, hidden_dim`
            mask_labels (`mindspore.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            class_labels (`List[mindspore.Tensor]`):
                List of class labels of shape `(labels)`.
            text_queries (`mindspore.Tensor`):
                A tensor of shape `batch_size, num_queries, hidden_dim`
            auxiliary_predictions (`Dict[str, mindspore.Tensor]`, *optional*):
                if `use_auxiliary_loss` was set to `true` in [`OneFormerConfig`], then it contains the logits from the
                inner layers of the Detr's Decoder.
            calculate_contrastive_loss (`bool`, *optional*, defaults to `True`):
                Whether or not to calculate the contrastive loss.

        Returns:
            `Dict[str, Tensor]`: A dict of `mindspore.Tensor` containing two keys:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
            - **loss_mask** -- The loss computed using sigmoid ce loss on the predicted and ground truth masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
            - **loss_contrastive** -- The query-text contrstive loss computed using object and text queries.
            if `use_auxiliary_loss` was set to `true` in [`OneFormerConfig`], the dictionary contains addional losses
            for each auxiliary predictions.
        """

        # retrieve the matching between the outputs of the last layer and the labels
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        # compute the average number of target masks for normalization purposes
        num_masks = self.get_num_masks(class_labels)
        # get all the losses
        losses: Dict[str, Tensor] = {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }
        if calculate_contrastive_loss:
            losses = {**losses, **self.loss_contrastive(contrastive_queries_logits, text_queries)}

        # in case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs["masks_queries_logits"]
                class_queries_logits = aux_outputs["class_queries_logits"]
                loss_dict = self.construct(
                    masks_queries_logits,
                    class_queries_logits,
                    None,
                    mask_labels,
                    class_labels,
                    None,
                    calculate_contrastive_loss=False,
                )
                loss_dict = {f"{key}_{idx}": value for key, value in loss_dict.items()}
                losses.update(loss_dict)

        return losses

    def get_num_masks(self, class_labels: mindspore.Tensor) -> mindspore.Tensor:
        """
        Computes the average number of target masks across the batch, for normalization purposes.
        """
        num_masks = sum(len(classes) for classes in class_labels)
        num_masks = Tensor([num_masks], dtype=mindspore.float32)
        world_size = 1
        num_masks = ops.clamp(num_masks / world_size, min=1)
        return num_masks


@dataclass
class OneFormerTransformerDecoderOutput(BaseModelOutput):
    """
    Base class for outputs of the Transformer decoder. This class adds attributes for class predictions, mask
    predictions and contrastive logits to BaseModelOutputWithCrossAttentions.

    Args:
        object_logits (`mindspore.Tensor` of shape `(batch_size, num_queries, hidden_dim)`):
            Queries representation for the region proposals.
        contrastive_logits (`mindspore.Tensor` of shape `(batch_size, num_queries, hidden_dim)`):
            Queries representation for the contrastive loss.
        prediction_masks (`mindspore.Tensor` of shape `(batch_size, num_queries, height, width)`):
            Mask predictions from last layer of the transformer decoder.
        prediction_class (`mindspore.Tensor` of shape `(batch_size, num_queries, num_classes+1)`):
            Class predictions from last layer of the transformer decoder.
        auxiliary_predictions (Tuple of Dict of `str, mindspore.Tensor`, *optional*):
            Tuple of class and mask predictions from each layer of the transformer decoder.
    """

    object_queries: mindspore.Tensor = None
    contrastive_logits: Optional[mindspore.Tensor] = None
    prediction_masks: mindspore.Tensor = None
    prediction_class: mindspore.Tensor = None
    auxiliary_predictions: Optional[Tuple[Dict[str, mindspore.Tensor]]] = None


@dataclass
# Copied from transformers.models.mask2former.modeling_mask2former.Mask2FormerPixelDecoderOutput with Mask2->One
class OneFormerPixelDecoderOutput(ModelOutput):
    """
    OneFormer's pixel decoder module output, practically a Multi-Scale Deformable Attention based decoder. It returns
    the mask features and the multiscale features.

    Args:
        multi_scale_features (`tuple(mindspore.Tensor)`):
            Tuple of multi-scale features of scales [1/8, 1/16, 1/32] and shape `(batch_size, num_channels, height,
            width)`from the Multi-Scale Deformable Attenntion based Pixel Decoder.
        mask_features (`mindspore.Tensor`):
            Tensor of shape `(batch_size, num_channels, height, width)`, 1/4 scale features from the last Pixel Decoder
            Layer.
        attentions (`tuple(mindspore.Tensor)`, *optional*):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from pixel decoder. Returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`
    """

    multi_scale_features: Tuple[mindspore.Tensor] = None
    mask_features: mindspore.Tensor = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


@dataclass
class OneFormerPixelLevelModuleOutput(ModelOutput):
    """
    OneFormer's pixel level module output. It returns both the last and (optionally) the hidden states from the
    `encoder` and `decoder`. By default, the `encoder` is a Swin/Dinat Backbone and the `decoder` is a Multi-Scale
    Deformable Attention based decoder.

    Args:
        encoder_features (List of `(mindspore.Tensor)`):
            List of `mindspore.Tensor` of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
            called feature maps) of the model at the output of each stage.
        decoder_features (List of `(mindspore.Tensor)`):
            List of `mindspore.Tensor` of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
            called feature maps) of the model at the output of each stage.
        decoder_last_feature (`mindspore.Tensor` of shape `(batch_size, num_channels, height, width)):
            1/4 scale features from the last Pixel Decoder Layer.
    """

    encoder_features: List[mindspore.Tensor] = None
    decoder_features: List[mindspore.Tensor] = None
    decoder_last_feature: mindspore.Tensor = None


@dataclass
class OneFormerModelOutput(ModelOutput):
    """
    Class for outputs of [`OneFormerModel`]. This class returns all the needed hidden states to compute the logits.

    Args:
        encoder_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage.
        pixel_decoder_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage.
        transformer_decoder_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage.
        transformer_decoder_object_queries (`mindspore.Tensor` of shape `(batch_size, num_queries, hidden_dim)`)
            Output object queries from the last layer in the transformer decoder.
        transformer_decoder_contrastive_queries (`mindspore.Tensor` of shape `(batch_size, num_queries, hidden_dim)`)
            Contrastive queries from the transformer decoder.
        transformer_decoder_mask_predictions (`mindspore.Tensor` of shape `(batch_size, num_queries, height, width)`)
            Mask Predictions from the last layer in the transformer decoder.
        transformer_decoder_class_predictions (`mindspore.Tensor` of shape `(batch_size, num_queries, num_classes+1)`):
            Class Predictions from the last layer in the transformer decoder.
        transformer_decoder_auxiliary_predictions (Tuple of Dict of `str, mindspore.Tensor`, *optional*):
            Tuple of class and mask predictions from each layer of the transformer decoder.
        text_queries (`mindspore.Tensor`, *optional* of shape `(batch_size, num_queries, hidden_dim)`)
            Text queries derived from the input text list used for calculating contrastive loss during training.
        task_token (`mindspore.Tensor` of shape `(batch_size, hidden_dim)`)
            1D task token to condition the queries.
        attentions (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tuple(mindspore.Tensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Self and Cross Attentions weights from transformer decoder.
    """

    encoder_hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    transformer_decoder_hidden_states: Optional[mindspore.Tensor] = None
    transformer_decoder_object_queries: mindspore.Tensor = None
    transformer_decoder_contrastive_queries: Optional[mindspore.Tensor] = None
    transformer_decoder_mask_predictions: mindspore.Tensor = None
    transformer_decoder_class_predictions: mindspore.Tensor = None
    transformer_decoder_auxiliary_predictions: Optional[Tuple[Dict[str, mindspore.Tensor]]] = None
    text_queries: Optional[mindspore.Tensor] = None
    task_token: mindspore.Tensor = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


@dataclass
class OneFormerForUniversalSegmentationOutput(ModelOutput):
    """
    Class for outputs of [`OneFormerForUniversalSegmentationOutput`].

    This output can be directly passed to [`~OneFormerImageProcessor.post_process_semantic_segmentation`] or
    [`~OneFormerImageProcessor.post_process_instance_segmentation`] or
    [`~OneFormerImageProcessor.post_process_panoptic_segmentation`] depending on the task. Please, see
    [`~OneFormerImageProcessor] for details regarding usage.

    Args:
        loss (`mindspore.Tensor`, *optional*):
            The computed loss, returned when labels are present.
        class_queries_logits (`mindspore.Tensor`):
            A tensor of shape `(batch_size, num_queries, num_labels + 1)` representing the proposed classes for each
            query. Note the `+ 1` is needed because we incorporate the null class.
        masks_queries_logits (`mindspore.Tensor`):
            A tensor of shape `(batch_size, num_queries, height, width)` representing the proposed masks for each
            query.
        auxiliary_predictions (List of Dict of `str, mindspore.Tensor`, *optional*):
            List of class and mask predictions from each layer of the transformer decoder.
        encoder_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage.
        pixel_decoder_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage.
        transformer_decoder_hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage.
        transformer_decoder_object_queries (`mindspore.Tensor` of shape `(batch_size, num_queries, hidden_dim)`)
            Output object queries from the last layer in the transformer decoder.
        transformer_decoder_contrastive_queries (`mindspore.Tensor` of shape `(batch_size, num_queries, hidden_dim)`)
            Contrastive queries from the transformer decoder.
        transformer_decoder_mask_predictions (`mindspore.Tensor` of shape `(batch_size, num_queries, height, width)`)
            Mask Predictions from the last layer in the transformer decoder.
        transformer_decoder_class_predictions (`mindspore.Tensor` of shape `(batch_size, num_queries, num_classes+1)`):
            Class Predictions from the last layer in the transformer decoder.
        transformer_decoder_auxiliary_predictions (List of Dict of `str, mindspore.Tensor`, *optional*):
            List of class and mask predictions from each layer of the transformer decoder.
        text_queries (`mindspore.Tensor`, *optional* of shape `(batch_size, num_queries, hidden_dim)`)
            Text queries derived from the input text list used for calculating contrastive loss during training.
        task_token (`mindspore.Tensor` of shape `(batch_size, hidden_dim)`)
            1D task token to condition the queries.
        attentions (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tuple(mindspore.Tensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Self and Cross Attentions weights from transformer decoder.
    """

    loss: Optional[mindspore.Tensor] = None
    class_queries_logits: mindspore.Tensor = None
    masks_queries_logits: mindspore.Tensor = None
    auxiliary_predictions: List[Dict[str, mindspore.Tensor]] = None
    encoder_hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    pixel_decoder_hidden_states: Optional[List[mindspore.Tensor]] = None
    transformer_decoder_hidden_states: Optional[mindspore.Tensor] = None
    transformer_decoder_object_queries: mindspore.Tensor = None
    transformer_decoder_contrastive_queries: Optional[mindspore.Tensor] = None
    transformer_decoder_mask_predictions: mindspore.Tensor = None
    transformer_decoder_class_predictions: mindspore.Tensor = None
    transformer_decoder_auxiliary_predictions: Optional[List[Dict[str, mindspore.Tensor]]] = None
    text_queries: Optional[mindspore.Tensor] = None
    task_token: mindspore.Tensor = None
    attentions: Optional[Tuple[Tuple[mindspore.Tensor]]] = None


# Modified from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrFrozenBatchNorm2d with DeformableDetr->OneFormerPixelDecoder
class OneFormerPixelDecoderFrozenBatchNorm2d(nn.Cell):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n):
        super().__init__()
        self.weight = ops.ones(n)
        self.bias = ops.zeros(n)
        self.running_mean = ops.zeros(n)
        self.running_var = ops.ones(n)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def construct(self, x):
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-5
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


# Modified from transformers.models.detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention with DeformableDetr->OneFormerPixelDecoderEncoder
class OneFormerPixelDecoderEncoderMultiscaleDeformableAttention(nn.Cell):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, embed_dim: int, num_heads: int, n_levels: int, n_points: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {embed_dim} and {num_heads}"
            )
        dim_per_head = embed_dim // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in DeformableDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 128

        self.d_model = embed_dim
        self.n_levels = n_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Dense(embed_dim, num_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Dense(embed_dim, num_heads * n_levels * n_points)
        self.value_proj = nn.Dense(embed_dim, embed_dim)
        self.output_proj = nn.Dense(embed_dim, embed_dim)

    def with_pos_embed(self, tensor: mindspore.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[mindspore.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # we invert the attention_mask
            value = value.masked_fill(attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = ops.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # batch_size, num_queries, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = ops.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")
        # MindSpore implementation
        output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output, attention_weights


class OneFormerPixelDecoderEncoderLayer(nn.Cell):
    def __init__(self, config: OneFormerConfig):
        super().__init__()
        self.embed_dim = config.conv_dim
        self.self_attn = OneFormerPixelDecoderEncoderMultiscaleDeformableAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            n_levels=3,
            n_points=4,
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)
        self.dropout = config.dropout
        self.activation_fn = ops.relu
        self.activation_dropout = config.dropout
        self.fc1 = nn.Dense(self.embed_dim, config.encoder_feedforward_dim)
        self.fc2 = nn.Dense(config.encoder_feedforward_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)

        self.is_training = config.is_training

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        position_embeddings: mindspore.Tensor = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            position_embeddings (`mindspore.Tensor`, *optional*):
                Position embeddings, to be added to `hidden_states`.
            reference_points (`mindspore.Tensor`, *optional*):
                Reference points.
            spatial_shapes (`mindspore.Tensor`, *optional*):
                Spatial shapes of the backbone feature maps.
            level_start_index (`mindspore.Tensor`, *optional*):
                Level start index.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps.
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.is_training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = ops.dropout(hidden_states, p=self.activation_dropout, training=self.is_training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.is_training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.is_training:
            if ops.isinf(hidden_states).any() or ops.isnan(hidden_states).any():
                clamp_value = finfo(hidden_states.dtype, 'max') - 1000
                hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Modified from from transformers.models.detr.modeling_deformable_detr.DeformableDetrEncoder with DeformableDetrEncoder->OneFormerPixelDecoderEncoderOnly
class OneFormerPixelDecoderEncoderOnly(nn.Cell):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`OneFormerPixelDecoderEncoderLayer`].

    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: OneFormerConfig
    """

    def __init__(self, config: OneFormerConfig):
        super().__init__()

        self.config = config
        self.dropout = config.dropout
        self.layers = nn.CellList([OneFormerPixelDecoderEncoderLayer(config) for _ in range(config.encoder_layers)])

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios):
        """
        Get reference points for each feature map. Used in decoder.

        Args:
            spatial_shapes (`mindspore.Tensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            valid_ratios (`mindspore.Tensor` of shape `(batch_size, num_feature_levels, 2)`):
                Valid ratios of each feature map.
        Returns:
            `mindspore.Tensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        for lvl, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = ops.meshgrid(
                ops.linspace(0.5, height - 0.5, height),
                ops.linspace(0.5, width - 0.5, width),
                indexing='ij',
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width)
            ref = ops.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = ops.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def construct(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_embeddings=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            position_embeddings (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes (`mindspore.Tensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            level_start_index (`mindspore.Tensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`mindspore.Tensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = inputs_embeds
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


# Modified from from transformers.models.mask2former.modeling_mask2former.Mask2FormerPixelDecoder with Mask2->One
class OneFormerPixelDecoder(nn.Cell):
    def __init__(self, config: OneFormerConfig, feature_channels):
        super().__init__()

        self.config = config

        #  positional encoding
        self.position_embedding = OneFormerSinePositionEmbedding(num_pos_feats=config.conv_dim // 2, normalize=True)
        self.num_feature_levels = 3
        transformer_in_channels = feature_channels[-self.num_feature_levels :]
        self.transformer_feature_strides = config.strides[-self.num_feature_levels :]
        self.feature_channels = feature_channels
        self.level_embed = Parameter(ops.zeros([self.num_feature_levels, config.conv_dim]))

        # Create input projection layers
        if self.num_feature_levels > 1:
            input_projections_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_projections_list.append(
                    nn.SequentialCell(
                        nn.Conv2d(in_channels, config.conv_dim, kernel_size=1, has_bias=True),
                        nn.GroupNorm(32, config.conv_dim),
                    )
                )
            self.input_projections = nn.CellList(input_projections_list)
        else:
            self.input_projections = nn.CellList(
                [
                    nn.SequentialCell(
                        nn.Conv2d(transformer_in_channels[-1], config.conv_dim, kernel_size=1, has_bias=True),
                        nn.GroupNorm(32, config.conv_dim),
                    )
                ]
            )

        self.encoder = OneFormerPixelDecoderEncoderOnly(config)

        self.mask_projection = nn.Conv2d(
            config.conv_dim,
            config.mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            has_bias=True,
        )

        self.common_stride = config.common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        for idx, in_channels in enumerate(self.feature_channels[: self.num_fpn_levels]):
            lateral_conv = nn.SequentialCell(
                nn.Conv2d(
                    in_channels,
                    config.conv_dim,
                    kernel_size=1,
                    has_bias=False,
                ),
                nn.GroupNorm(32, config.conv_dim),
            )
            output_conv = nn.SequentialCell(
                nn.Conv2d(
                    config.conv_dim,
                    config.conv_dim,
                    kernel_size=3,
                    stride=1,
                    pad_mode='pad',
                    padding=1,
                    has_bias=False,
                ),
                nn.GroupNorm(32, config.conv_dim),
                nn.ReLU(),
            )
            self.insert_child_to_cell("adapter_{}".format(idx + 1), lateral_conv)
            self.insert_child_to_cell("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def get_valid_ratio(self, mask, dtype=mindspore.float32):
        """Get the valid ratio of all feature maps."""

        _, height, width = mask.shape
        valid_height = ops.sum(~mask[:, :, 0], 1)
        valid_width = ops.sum(~mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.to(dtype) / height
        valid_ratio_width = valid_width.to(dtype) / width
        valid_ratio = ops.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def construct(
        self,
        features,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        sources = []
        position_embeddings_list = []
        for level, source in enumerate(features[::-1][: self.num_feature_levels]):
            sources.append(self.input_projections[level](source))
            position_embeddings_list.append(self.position_embedding(source))

        masks = [ops.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=mindspore.bool_) for x in sources]

        # Prepare encoder inputs (by flattening)
        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, position_embeddings_list)):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes.append(spatial_shape)
            source = source.flatten(start_dim=2).swapaxes(1, 2)
            mask = mask.flatten(start_dim=1)
            pos_embed = pos_embed.flatten(start_dim=2).swapaxes(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = ops.cat(source_flatten, 1)
        mask_flatten = ops.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = ops.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = Tensor(spatial_shapes, dtype=mindspore.int64)
        level_start_index = ops.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = ops.stack([self.get_valid_ratio(m, dtype=source_flatten.dtype) for m in masks], 1)

        # Fourth, sent source_flatten + mask_flatten + lvl_pos_embed_flatten (backbone + proj layer output) through encoder
        # Also provide spatial_shapes, level_start_index and valid_ratios
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=source_flatten,
                attention_mask=mask_flatten,
                position_embeddings=lvl_pos_embed_flatten,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        y = encoder_outputs.last_hidden_state
        bs = y.shape[0]

        split_size_or_sections = [None] * self.num_feature_levels
        for i in range(self.num_feature_levels):
            if i < self.num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = ops.split(y, [x.item() for x in split_size_or_sections], axis=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(z.swapaxes(1, 2).view(bs, -1, spatial_shapes[i][0].item(), spatial_shapes[i][1].item()))

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, feats in enumerate(features[: self.num_fpn_levels][::-1]):
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(feats)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + ops.interpolate(
                out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False
            )
            y = output_conv(y)
            out.append(y)

        for o in out:
            if num_cur_levels < self.num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        return OneFormerPixelDecoderOutput(
            mask_features=self.mask_projection(out[-1]),
            multi_scale_features=multi_scale_features,
            attentions=encoder_outputs.attentions,
        )


# Modified from from transformers.models.mask2former.modeling_mask2former.Mask2FormerPixelLevelModule with Mask2->One
class OneFormerPixelLevelModule(nn.Cell):
    def __init__(self, config: OneFormerConfig):
        """
        Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image
        Segmentation](https://arxiv.org/abs/2112.01527). It runs the input image through a backbone and a pixel
        decoder, generating multi-scale feature maps and pixel embeddings.

        Args:
            config ([`OneFormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__()
        self.encoder = load_backbone(config)
        self.decoder = OneFormerPixelDecoder(config, feature_channels=self.encoder.channels)

    def construct(self, pixel_values: Tensor, output_hidden_states: bool = False) -> OneFormerPixelLevelModuleOutput:
        features: List[Tensor] = self.encoder(pixel_values).feature_maps
        decoder_output: OneFormerPixelDecoderOutput = self.decoder(features, output_hidden_states=output_hidden_states)
        return OneFormerPixelLevelModuleOutput(
            encoder_features=tuple(features),
            decoder_features=decoder_output.multi_scale_features,
            decoder_last_feature=decoder_output.mask_features,
        )


# Modified from transformers.models.detr.modeling_detr.DetrAttention with Detr->OneFormer
class OneFormerAttention(nn.Cell):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Here, we add position embeddings to the queries and
    keys (as explained in the DETR paper).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.v_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.q_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.out_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def with_pos_embed(self, tensor: mindspore.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_embeddings: Optional[mindspore.Tensor] = None,
        key_value_states: Optional[mindspore.Tensor] = None,
        key_value_position_embeddings: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        hidden_states = hidden_states.permute(1, 0, 2) if hidden_states is not None else None
        position_embeddings = position_embeddings.permute(1, 0, 2) if position_embeddings is not None else None
        key_value_states = key_value_states.permute(1, 0, 2) if key_value_states is not None else None
        key_value_position_embeddings = (
            key_value_position_embeddings.permute(1, 0, 2) if key_value_position_embeddings is not None else None
        )

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.shape

        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # add key-value position embeddings to the key value states
        if key_value_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(key_value_states, key_value_position_embeddings)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states_original), -1, batch_size)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.shape[1]

        attn_weights = ops.bmm(query_states, key_states.swapaxes(1, 2))

        if attn_weights.shape != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (batch_size * self.num_heads, target_len, source_len):
                raise ValueError(
                    f"Attention mask should be of size {(target_len, batch_size * self.num_heads, source_len)}, but is"
                    f" {attention_mask.shape}"
                )
            attn_weights += attention_mask

        attn_weights = ops.softmax(attn_weights, axis=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None

        attn_probs = ops.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = ops.bmm(attn_probs, value_states)

        if attn_output.shape != (batch_size * self.num_heads, target_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)

        attn_output = self.out_proj(attn_output).permute(1, 0, 2)

        return attn_output, attn_weights_reshaped


class OneFormerTransformerDecoderSelfAttentionLayer(nn.Cell):
    def __init__(
        self, embed_dim, num_heads, dropout=0.0, activation="relu", normalize_before=False, layer_norm_eps=1e-05
    ):
        super().__init__()
        self.self_attn = OneFormerAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, is_decoder=True)

        self.norm = nn.LayerNorm(embed_dim, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(p=dropout)

        self.activation = ACT2FN[activation]
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        output,
        output_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output2, attention_weights = self.self_attn(
            hidden_states=output, position_embeddings=query_pos, attention_mask=output_mask, output_attentions=True
        )
        output = output + self.dropout(output2)
        output = self.norm(output)

        return output, attention_weights

    def forward_pre(
        self,
        output,
        output_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output2 = self.norm(output)
        output2, attention_weights = self.self_attn(
            hidden_states=output2, position_embeddings=query_pos, attention_mask=output_mask, output_attentions=True
        )
        output = output + self.dropout(output2)

        return output, attention_weights

    def construct(
        self,
        output,
        output_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(output, output_mask, output_key_padding_mask, query_pos)
        return self.forward_post(output, output_mask, output_key_padding_mask, query_pos)


class OneFormerTransformerDecoderCrossAttentionLayer(nn.Cell):
    def __init__(
        self, embed_dim, num_heads, dropout=0.0, activation="relu", normalize_before=False, layer_norm_eps=1e-05
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        self.norm = nn.LayerNorm(embed_dim, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(p=dropout)

        self.activation = ACT2FN[activation]
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        output,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output2, attention_weights = self.multihead_attn(
            self.with_pos_embed(output, query_pos),
            self.with_pos_embed(memory, pos),
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        output = output + self.dropout(output2)
        output = self.norm(output)

        return output, attention_weights

    def forward_pre(
        self,
        output,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output2 = self.norm(output)
        output2, attention_weights = self.multihead_attn(
            self.with_pos_embed(output2, query_pos),
            self.with_pos_embed(memory, pos),
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        output = output + self.dropout(output2)

        return output, attention_weights

    def construct(
        self,
        output,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(output, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(output, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class OneFormerTransformerDecoderFFNLayer(nn.Cell):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        layer_norm_eps=1e-05,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Dense(d_model, dim_feedforward)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Dense(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model, epsilon=layer_norm_eps)

        self.activation = ACT2FN[activation]
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, output):
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = output + self.dropout(output2)
        output = self.norm(output)
        return output

    def forward_pre(self, output):
        output2 = self.norm(output)
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output2))))
        output = output + self.dropout(output2)
        return output

    def construct(self, output):
        if self.normalize_before:
            return self.forward_pre(output)
        return self.forward_post(output)


class OneFormerMLPPredictionHead(nn.Cell):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        """
        A classic Multi Layer Perceptron (MLP).

        Args:
            input_dim (`int`):
                The input dimensions.
            hidden_dim (`int`):
                The hidden dimensions.
            output_dim (`int`):
                The output dimensions.
            num_layers (int, *optional*, defaults to 3):
                The number of layers.
        """
        super().__init__()
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            layers.append(
                PredictionBlock(in_dim, out_dim, activation=nn.ReLU() if i < num_layers - 1 else nn.Identity())
            )

        self.layers = nn.SequentialCell(*layers)

    def construct(self, input: Tensor) -> Tensor:
        return self.layers(input)


# refactored from original implementation
class OneFormerTransformerDecoderLayer(nn.Cell):
    def __init__(self, config: OneFormerConfig):
        super().__init__()
        self.embed_dim = config.hidden_dim
        self.num_feature_levels = 3

        self.cross_attn = OneFormerTransformerDecoderCrossAttentionLayer(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=0.0,
            normalize_before=config.pre_norm,
            layer_norm_eps=config.layer_norm_eps,
        )

        self.self_attn = OneFormerTransformerDecoderSelfAttentionLayer(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=0.0,
            normalize_before=config.pre_norm,
            layer_norm_eps=config.layer_norm_eps,
        )

        self.ffn = OneFormerTransformerDecoderFFNLayer(
            d_model=self.embed_dim,
            dim_feedforward=config.dim_feedforward,
            dropout=0.0,
            normalize_before=config.pre_norm,
            layer_norm_eps=config.layer_norm_eps,
        )

    def construct(
        self,
        index: int,
        output: mindspore.Tensor,
        multi_stage_features: List[mindspore.Tensor],
        multi_stage_positional_embeddings: List[mindspore.Tensor],
        attention_mask: Optional[mindspore.Tensor] = None,
        query_embeddings: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            index (`int`): index of the layer in the Transformer decoder.
            output (`mindspore.Tensor`): the object queries of shape `(N, batch, hidden_dim)`
            multi_stage_features (`List[mindspore.Tensor]`): the multi-scale features from the pixel decoder.
            multi_stage_positional_embeddings (`List[mindspore.Tensor]`):
                positional embeddings for the multi_stage_features
            attention_mask (`mindspore.Tensor`): attention mask for the masked cross attention layer
            query_embeddings (`mindspore.Tensor`, *optional*):
                position embeddings that are added to the queries and keys in the self-attention layer.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        level_index = index % self.num_feature_levels
        attention_mask[attention_mask.sum(-1) == attention_mask.shape[-1]] = False

        # Masked Cross Attention
        output, cross_attn_weights = self.cross_attn(
            output,
            multi_stage_features[level_index],
            memory_mask=attention_mask,
            memory_key_padding_mask=None,  # here we do not apply masking on padded region
            pos=multi_stage_positional_embeddings[level_index],
            query_pos=query_embeddings,
        )

        # Self Attention
        output, self_attn_weights = self.self_attn(
            output,
            output_mask=None,
            output_key_padding_mask=None,
            query_pos=query_embeddings,
        )

        # Fully Connected
        output = self.ffn(output)

        outputs = (output,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


class OneFormerTransformerDecoderQueryTransformerDecoder(nn.Cell):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def construct(
        self,
        output,
        memory,
        output_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                output_mask=output_mask,
                memory_mask=memory_mask,
                output_key_padding_mask=output_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return ops.stack(intermediate)

        return output.unsqueeze(0)


class OneFormerTransformerDecoderQueryTransformerDecoderLayer(nn.Cell):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        layer_norm_eps=1e-05,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Dense(d_model, dim_feedforward)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Dense(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, epsilon=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, epsilon=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, epsilon=layer_norm_eps)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

        self.activation = ACT2FN[activation]
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        output,
        memory,
        output_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(output, query_pos)
        output2 = self.self_attn(q, k, output, attn_mask=output_mask, key_padding_mask=output_key_padding_mask)
        output2 = output2[0]
        output = output + self.dropout1(output2)
        output = self.norm1(output)
        output2 = self.multihead_attn(
            self.with_pos_embed(output, query_pos),
            self.with_pos_embed(memory, pos),
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        output2 = output2[0]
        output = output + self.dropout2(output2)
        output = self.norm2(output)
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = output + self.dropout3(output2)
        output = self.norm3(output)
        return output

    def forward_pre(
        self,
        output,
        memory,
        output_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output2 = self.norm1(output)
        q = k = self.with_pos_embed(output2, query_pos)
        output2 = self.self_attn(q, k, value=output2, attn_mask=output_mask, key_padding_mask=output_key_padding_mask)
        output2 = output2[0]
        output = output + self.dropout1(output2)
        output2 = self.norm2(output)
        output2 = self.multihead_attn(
            self.with_pos_embed(output2, query_pos),
            self.with_pos_embed(memory, pos),
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        output2 = output2[0]
        output = output + self.dropout2(output2)
        output2 = self.norm3(output)
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output2))))
        output = output + self.dropout3(output2)
        return output

    def construct(
        self,
        output,
        memory,
        output_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                output,
                memory,
                output_mask,
                memory_mask,
                output_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            output,
            memory,
            output_mask,
            memory_mask,
            output_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


class OneFormerTransformerDecoderQueryTransformer(nn.Cell):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        layer_norm_eps=1e-05,
    ):
        super().__init__()

        decoder_layer = OneFormerTransformerDecoderQueryTransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before, layer_norm_eps
        )
        decoder_norm = nn.LayerNorm(d_model, epsilon=layer_norm_eps)
        self.decoder = OneFormerTransformerDecoderQueryTransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self.d_model = d_model
        self.nhead = nhead

    def construct(self, src, mask, query_embed, pos_embed, task_token=None):
        batch_size = src.shape[0]
        src = src.flatten(start_dim=2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(start_dim=2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        if mask is not None:
            mask = mask.flatten(start_dim=1)

        if task_token is None:
            queries = ops.zeros_like(query_embed)
        else:
            queries = task_token.repeat(query_embed.shape[0], 1, 1)

        queries = self.decoder(queries, src, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return queries.swapaxes(1, 2)


class OneFormerTransformerDecoder(nn.Cell):
    """
    Transformer decoder
    """

    def __init__(self, in_channels: int, config: OneFormerConfig):
        super().__init__()
        self.config = config

        self.dropout = config.dropout
        self.num_heads = config.num_attention_heads
        self.is_training = config.is_training
        self.use_task_norm = config.use_task_norm
        self.use_auxiliary_loss = config.use_auxiliary_loss

        self.query_transformer = OneFormerTransformerDecoderQueryTransformer(
            d_model=config.hidden_dim,
            dropout=config.dropout,
            nhead=config.num_attention_heads,
            dim_feedforward=config.dim_feedforward,
            num_decoder_layers=config.query_dec_layers,
            normalize_before=config.pre_norm,
            return_intermediate_dec=False,
            layer_norm_eps=config.layer_norm_eps,
        )

        self.decoder_norm = nn.LayerNorm(config.hidden_dim, epsilon=config.layer_norm_eps)

        self.num_feature_levels = 3

        self.layers = nn.CellList(
            [OneFormerTransformerDecoderLayer(config) for _ in range(config.decoder_layers - 1)]
        )

        self.query_input_projection = nn.Conv2d(in_channels, config.hidden_dim, kernel_size=1, has_bias=True)

        self.class_embed = nn.Dense(config.hidden_dim, config.num_labels + 1)
        self.mask_embed = OneFormerMLPPredictionHead(
            config.hidden_dim,
            config.hidden_dim,
            config.mask_dim,
            3,
        )

    def construct(
        self,
        task_token=None,
        multi_stage_features=None,
        multi_stage_positional_embeddings=None,
        mask_features=None,
        query_features=None,
        query_embeddings=None,
        query_embedder=None,
        size_list=None,
        output_attentions=None,
    ):
        if self.use_task_norm:
            task_token = self.decoder_norm(task_token)

        object_queries = self.query_transformer(
            query_features,
            None,
            query_embedder.weight[:-1],
            self.query_input_projection(mask_features),
            task_token if self.use_task_norm else None,
        )

        object_queries = object_queries[0].permute(1, 0, 2)

        queries = ops.cat([object_queries, task_token], axis=0)

        output = Tensor.copy(queries)

        intermediate_class_predictions = []
        intermediate_mask_predictions = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attention_mask = self.forward_prediction_heads(
            output, mask_features, attention_mask_target_size=size_list[0]
        )
        intermediate_class_predictions.append(outputs_class)
        intermediate_mask_predictions.append(outputs_mask)

        attentions = ()

        for index, layer in enumerate(self.layers):
            layer_outputs = layer(
                index=index,
                output=output,
                multi_stage_features=multi_stage_features,
                multi_stage_positional_embeddings=multi_stage_positional_embeddings,
                attention_mask=attention_mask,
                query_embeddings=query_embeddings,
                output_attentions=output_attentions,
            )

            output = layer_outputs[0]
            attentions += (layer_outputs[1:],)

            outputs_class, outputs_mask, attention_mask = self.forward_prediction_heads(
                output, mask_features, attention_mask_target_size=size_list[(index + 1) % self.num_feature_levels]
            )
            intermediate_class_predictions.append(outputs_class)
            intermediate_mask_predictions.append(outputs_mask)

        if not len(intermediate_mask_predictions) == len(self.layers) + 1:
            raise ValueError(
                "Intermediate predictions in the transformer decoder must have the same number of elements as number"
                " of layers"
            )

        object_queries = layer_outputs[0].permute(1, 0, 2)

        contrastive_logits = queries.permute(1, 0, 2)

        return OneFormerTransformerDecoderOutput(
            object_queries=object_queries,
            contrastive_logits=contrastive_logits,
            prediction_masks=intermediate_mask_predictions[-1],
            prediction_class=intermediate_class_predictions[-1],
            auxiliary_predictions=self._get_aux_predictions(
                intermediate_class_predictions, intermediate_mask_predictions
            )
            if self.use_auxiliary_loss
            else None,
            attentions=attentions,
        )

    def forward_prediction_heads(self, output, mask_features, attention_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.swapaxes(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = ops.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        attention_mask = ops.interpolate(
            outputs_mask, size=attention_mask_target_size, mode="bilinear", align_corners=False
        )

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        def flatten_01(x:Tensor) -> Tensor: # impl. flatten(start_dim=0, stop_dim=1)
            B, C, *others = x.shape
            return x.reshape([B * C, *others])

        attention_mask = (
            flatten_01(attention_mask.sigmoid().flatten(start_dim=2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)) < 0.5
        ).bool()

        return outputs_class, outputs_mask, attention_mask

    # @ops.jit.unused
    def _get_aux_predictions(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        aux_list = [
            {"class_queries_logits": a, "masks_queries_logits": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
        return tuple(aux_list)


class OneFormerTransformerModule(nn.Cell):
    """
    The OneFormer's transformer module.
    """

    def __init__(self, in_features: int, config: OneFormerConfig):
        super().__init__()
        hidden_dim = config.hidden_dim
        self.num_feature_levels = 3
        self.position_embedder = OneFormerSinePositionEmbedding(num_pos_feats=hidden_dim // 2, normalize=True)
        self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)
        self.input_projections = []

        for _ in range(self.num_feature_levels):
            if in_features != hidden_dim or config.enforce_input_proj:
                self.input_projections.append(nn.Conv2d(in_features, hidden_dim, kernel_size=1, has_bias=True))
            else:
                self.input_projections.append(nn.SequentialCell())

        self.decoder = OneFormerTransformerDecoder(in_channels=in_features, config=config)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

    def construct(
        self,
        multi_scale_features: List[Tensor],
        mask_features: Tensor,
        task_token: Tensor,
        output_attentions: bool = False,
    ) -> OneFormerTransformerDecoderOutput:
        if not len(multi_scale_features) == self.num_feature_levels:
            raise ValueError(
                f"Number of elements in multi_scale_features ({len(multi_scale_features)}) and num_feature_levels"
                f" ({self.num_feature_levels}) do not match!"
            )
        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            multi_stage_positional_embeddings.append(self.position_embedder(multi_scale_features[i], None).flatten(start_dim=2))
            multi_stage_features.append(
                self.input_projections[i](multi_scale_features[i]).flatten(start_dim=2)
                + self.level_embed.weight[i][None, :, None]
            )

            # flatten NxCxHxW to HWxNxC
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        _, batch_size, _ = multi_stage_features[0].shape

        # QxNxC
        query_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        task_token = task_token.unsqueeze(0)

        query_features = self.position_embedder(mask_features, None)

        return self.decoder(
            task_token=task_token,
            multi_stage_features=multi_stage_features,
            multi_stage_positional_embeddings=multi_stage_positional_embeddings,
            mask_features=mask_features,
            query_features=query_features,
            query_embeddings=query_embeddings,
            query_embedder=self.queries_embedder,
            size_list=size_list,
            output_attentions=output_attentions,
        )


# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerSinePositionEmbedding with Mask->One
class OneFormerSinePositionEmbedding(nn.Cell):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def construct(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            mask = ops.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=mindspore.bool_)
        not_mask = (~mask).to(x.dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = ops.arange(self.num_pos_feats, dtype=mindspore.int64).type_as(x)
        dim_t = self.temperature ** (2 * ops.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = ops.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), axis=4).flatten(start_dim=3)
        pos_y = ops.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), axis=4).flatten(start_dim=3)
        pos = ops.cat((pos_y, pos_x), axis=3).permute(0, 3, 1, 2)
        return pos


# Copied from transformers.models.maskformer.modeling_maskformer.PredictionBlock
class PredictionBlock(nn.Cell):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Cell) -> None:
        super().__init__()
        self.layers = [nn.Dense(in_dim, out_dim), activation]
        # Maintain submodule indexing as if part of a Sequential block
        for i, layer in enumerate(self.layers):
            self.insert_child_to_cell(str(i), layer)

    def construct(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class OneFormerTextMapperAttention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.k_proj = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.v_proj = nn.Dense(dim, dim, has_bias=qkv_bias)

        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, q, k, v):
        batch_size, q_sequence_length, num_channels = q.shape
        if not k.shape == v.shape:
            raise ValueError(f"keys ({list(k.shape)}) and values ({list(v.shape)}) have different shapes!")
        batch_size, k_sequence_length, num_channels = k.shape
        q = self.q_proj(q).reshape(batch_size, q_sequence_length, self.num_heads, num_channels // self.num_heads)
        k = self.k_proj(k).reshape(batch_size, k_sequence_length, self.num_heads, num_channels // self.num_heads)
        v = self.v_proj(v).reshape(batch_size, k_sequence_length, self.num_heads, num_channels // self.num_heads)

        attn = ops.einsum("bnkc,bmkc->bknm", q, k) * self.scale

        attn = ops.softmax(attn, axis=-1)

        output = ops.einsum("bknm,bmkc->bnkc", attn, v).reshape(batch_size, q_sequence_length, num_channels)

        output = self.proj(output)
        output = self.proj_drop(output)
        return output


class OneFormerTextTransformerDecoderLayer(nn.Cell):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
        layer_norm_eps=1e-05,
    ):
        super().__init__()
        self.self_attn = OneFormerTextMapperAttention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = OneFormerTextMapperAttention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model, epsilon=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, epsilon=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(p=dropout)

        self.mlp = nn.SequentialCell(
            nn.Dense(d_model, d_model * 4), nn.GELU(), nn.Dropout(p=dropout), nn.Dense(d_model * 4, d_model)
        )

    def construct(self, hidden_state, mem):
        q = k = v = self.norm1(hidden_state)
        hidden_state = hidden_state + self.self_attn(q, k, v)
        q = self.norm2(hidden_state)
        hidden_state = hidden_state + self.cross_attn(q, mem, mem)
        hidden_state = hidden_state + self.dropout(self.mlp(self.norm3(hidden_state)))
        return hidden_state


class OneFormerTextContextDecoder(nn.Cell):
    def __init__(
        self,
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=6,
        visual_dim=1024,
        dropout=0.1,
        layer_norm_eps=1e-05,
        **kwargs,
    ):
        super().__init__()

        self.memory_proj = nn.SequentialCell(
            nn.LayerNorm(visual_dim, epsilon=layer_norm_eps),
            nn.Dense(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width, epsilon=layer_norm_eps),
        )

        self.text_proj = nn.SequentialCell(
            nn.LayerNorm(visual_dim, epsilon=layer_norm_eps),
            nn.Dense(visual_dim, transformer_width),
        )

        self.decoder = nn.CellList(
            [
                OneFormerTextTransformerDecoderLayer(transformer_width, transformer_heads, dropout, layer_norm_eps)
                for _ in range(transformer_layers)
            ]
        )

        self.out_proj = nn.SequentialCell(
            nn.LayerNorm(transformer_width, epsilon=layer_norm_eps), nn.Dense(transformer_width, visual_dim)
        )

    def construct(self, text, visual):
        visual = self.memory_proj(visual)
        hidden_state = self.text_proj(text)

        for layer in self.decoder:
            hidden_state = layer(hidden_state, visual)

        return self.out_proj(hidden_state)


class OneFormerTextMLP(nn.Cell):
    def __init__(
        self,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        output_size: Optional[int] = None,
    ):
        super().__init__()
        self.activation_fn = ACT2FN["quick_gelu"]
        self.fc1 = nn.Dense(hidden_size, intermediate_size)
        self.fc2 = nn.Dense(intermediate_size, output_size)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class OneFormerTextTransformerLayer(nn.Cell):
    def __init__(self, width: int, heads: int, attn_mask: mindspore.Tensor, layer_norm_eps=1e-05):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, heads)
        self.layer_norm1 = nn.LayerNorm(width, epsilon=layer_norm_eps)
        self.mlp = OneFormerTextMLP(width, width * 4, width)
        self.layer_norm2 = nn.LayerNorm(width, epsilon=layer_norm_eps)
        self.attn_mask = attn_mask

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        key_padding_mask: Optional[mindspore.Tensor] = None,
    ) -> mindspore.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            need_weights=False,
            key_padding_mask=key_padding_mask,
        )[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class OneFormerTextTransformer(nn.Cell):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: mindspore.Tensor = None,
        use_checkpoint=False,
        layer_norm_eps=1e-05,
    ):
        super().__init__()
        self.width = width
        self.num_layers = layers
        self.layers = nn.SequentialCell(
            *[OneFormerTextTransformerLayer(width, heads, attn_mask, layer_norm_eps) for _ in range(layers)]
        )
        self.use_checkpoint = use_checkpoint

    def construct(self, hidden_states: mindspore.Tensor):
        for layer in self.layers:
            if self.use_checkpoint:
                hidden_states = self._gradient_checkpointing_func(layer, hidden_states)
            else:
                hidden_states = layer(hidden_states)
        return hidden_states


class OneFormerTextEncoder(nn.Cell):
    def __init__(
        self,
        context_length: int,
        width: int,
        layers: int,
        vocab_size,
        use_checkpoint=False,
        layer_norm_eps=1e-05,
    ):
        super().__init__()
        heads = width // 64
        self.context_length = context_length
        self.width = width
        self.transformer = OneFormerTextTransformer(
            width=width,
            layers=layers,
            heads=heads,
            attn_mask=self.build_attention_mask(),
            use_checkpoint=use_checkpoint,
            layer_norm_eps=layer_norm_eps,
        )

        self.positional_embedding = Parameter(ops.zeros(self.context_length, width))
        self.ln_final = nn.LayerNorm(width, epsilon=layer_norm_eps)
        self.token_embedding = nn.Embedding(vocab_size, width)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = ops.fill(mindspore.float32, (self.context_length, self.context_length), value=float("-inf"))
        ops.triu(mask, 1)  # zero out the lower diagonal
        return mask

    def construct(self, text):
        hidden_state = self.token_embedding(text)
        hidden_state = hidden_state + self.positional_embedding
        hidden_state = hidden_state.permute(1, 0, 2)
        hidden_state = self.transformer(hidden_state)
        hidden_state = hidden_state.permute(1, 0, 2)
        hidden_state = self.ln_final(hidden_state)
        hidden_state = hidden_state[ops.arange(hidden_state.shape[0]), text.argmax(axis=-1)]

        return hidden_state


class OneFormerTextMapper(nn.Cell):
    def __init__(self, config: OneFormerConfig):
        super().__init__()
        self.text_encoder = OneFormerTextEncoder(
            context_length=config.text_encoder_context_length,
            width=config.text_encoder_width,
            layers=config.text_encoder_num_layers,
            vocab_size=config.text_encoder_vocab_size,
            layer_norm_eps=config.layer_norm_eps,
        )

        self.text_projector = OneFormerMLPPredictionHead(
            config.text_encoder_width,
            config.hidden_dim,
            config.hidden_dim,
            config.text_encoder_proj_layers,
        )
        if config.text_encoder_n_ctx > 0:
            self.prompt_ctx = nn.Embedding(
                config.text_encoder_n_ctx,
                config.text_encoder_width,
            )
        else:
            self.prompt_ctx = None

    def construct(
        self,
        inputs: Tensor,
    ) -> Tensor:
        text_queries = self.encode_text(inputs)

        return text_queries

    def encode_text(self, text):
        if text.ndim is None:
            raise ValueError("text must not be NoneType")
        if text.ndim not in [2, 3]:
            raise ValueError("Number of dimensions in text must be 2 or 3")
        squeeze_dim = False
        num_text = 1
        if text.ndim == 3:
            num_text = text.shape[1]
            batch_size, num_text, hidden_dim = text.shape
            text = text.reshape(batch_size * num_text, hidden_dim)
            squeeze_dim = True

        # [batch_size, num_channels]
        encoded_text = self.text_encoder(text)

        text_queries = self.text_projector(encoded_text)

        if squeeze_dim:
            _, hidden_dim = text_queries.shape
            text_queries = text_queries.reshape(batch_size, num_text, hidden_dim)
            if self.prompt_ctx is not None:
                text_queries_ctx = self.prompt_ctx.weight.unsqueeze(0).repeat(text_queries.shape[0], 1, 1)
                text_queries = ops.cat([text_queries, text_queries_ctx], axis=1)

        return text_queries


class OneFormerTaskModel(nn.Cell):
    def __init__(self, config: OneFormerConfig):
        super().__init__()
        self.task_mlp = OneFormerMLPPredictionHead(
            config.task_seq_len,
            config.hidden_dim,
            config.hidden_dim,
            2,
        )

    def construct(self, inputs: Tensor) -> Tensor:
        task_tokens = self.task_mlp(inputs)
        return task_tokens


class OneFormerPreTrainedModel(PreTrainedModel):
    config_class = OneFormerConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def _init_weights(self, cell: nn.Cell):
        xavier_std = self.config.init_xavier_std
        std = self.config.init_std
        if isinstance(cell, OneFormerTransformerModule):
            if cell.input_projections is not None:
                for input_projection in cell.input_projections:
                    if not isinstance(input_projection, nn.SequentialCell):
                        input_projection.weight.set_data(initializer(XavierUniform(xavier_std), input_projection.weight.shape, input_projection.weight.dtype))
                        input_projection.bias.set_data(initializer('zeros', input_projection.bias.shape, input_projection.bias.dtype))
        elif isinstance(cell, OneFormerTransformerDecoder):
            cell.query_input_projection.weight.set_data(initializer(XavierUniform(xavier_std), cell.query_input_projection.weight.shape, cell.query_input_projection.weight.dtype))
            cell.query_input_projection.bias.set_data(initializer('zeros', cell.query_input_projection.bias.shape, cell.query_input_projection.bias.dtype))
            cell.query_input_projection._is_hf_initialized = True
        elif isinstance(cell, OneFormerPixelDecoderEncoderMultiscaleDeformableAttention):
            cell.sampling_offsets.weight.set_data(initializer('zeros', cell.sampling_offsets.weight.shape, cell.sampling_offsets.weight.dtype))
            thetas = ops.arange(cell.n_heads, dtype=mindspore.int64).float() * (2.0 * math.pi / cell.n_heads)
            grid_init = ops.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdims=True)[0])
                .view(cell.n_heads, 1, 1, 2)
                .repeat(1, cell.n_levels, cell.n_points, 1)
            )
            for i in range(cell.n_points):
                grid_init[:, :, i, :] *= i + 1
            cell.sampling_offsets.bias = Parameter(grid_init.view(-1))

            cell.attention_weights.weight.set_data(initializer('zeros', cell.attention_weights.weight.shape, cell.attention_weights.weight.dtype))
            cell.attention_weights.bias.set_data(initializer('zeros', cell.attention_weights.bias.shape, cell.attention_weights.bias.dtype))
            cell.value_proj.weight.set_data(initializer(XavierUniform(), cell.value_proj.weight.shape, cell.value_proj.weight.dtype))
            cell.value_proj.bias.set_data(initializer('zeros', cell.value_proj.bias.shape, cell.value_proj.bias.dtype))
            cell.output_proj.weight.set_data(initializer(XavierUniform(), cell.output_proj.weight.shape, cell.output_proj.weight.dtype))
            cell.output_proj.bias.set_data(initializer('zeros', cell.output_proj.bias.shape, cell.output_proj.bias.dtype))
        elif isinstance(cell, OneFormerPixelDecoderEncoderOnly):
            for _, p in cell.parameters_and_names():
                if p.dim() > 1:
                    p.set_data(initializer(XavierUniform(), p.shape, p.dtype))
        elif isinstance(cell, OneFormerPixelDecoder):
            for _, p in cell.parameters_and_names():
                if p.dim() > 1:
                    p.set_data(initializer(XavierUniform(xavier_std), p.shape, p.dtype))
            cell.level_embed.set_data(initializer(Normal(0), cell.level_embed.shape, cell.level_embed.dtype))
        elif isinstance(cell, OneFormerTransformerDecoderSelfAttentionLayer):
            for _, p in cell.parameters_and_names():
                if p.dim() > 1:
                    p.set_data(initializer(XavierUniform(xavier_std), p.shape, p.dtype))
        elif isinstance(cell, OneFormerTransformerDecoderCrossAttentionLayer):
            for _, p in cell.parameters_and_names():
                if p.dim() > 1:
                    p.set_data(initializer(XavierUniform(xavier_std), p.shape, p.dtype))
        elif isinstance(cell, OneFormerTransformerDecoderFFNLayer):
            for _, p in cell.parameters_and_names():
                if p.dim() > 1:
                    p.set_data(initializer(XavierUniform(xavier_std), p.shape, p.dtype))
        elif isinstance(cell, OneFormerTransformerDecoderQueryTransformer):
            for _, p in cell.parameters_and_names():
                if p.dim() > 1:
                    p.set_data(initializer(XavierUniform(xavier_std), p.shape, p.dtype))
        elif isinstance(cell, OneFormerPixelLevelModule):
            for subcell in cell.cells():
                if isinstance(subcell, (nn.Conv2d, nn.Dense)):
                    subcell.weight.set_data(initializer(Normal(sigma=std), subcell.weight.shape, subcell.weight.dtype))
                    if subcell.bias is not None:
                        subcell.bias.set_data(initializer('zeros', subcell.bias.shape, subcell.bias.dtype))
        elif isinstance(cell, OneFormerTextContextDecoder):
            for subcell in cell.cells():
                if isinstance(subcell, nn.Dense):
                    subcell.weight.set_data(initializer(TruncatedNormal(sigma=0.02), subcell.weight.shape, subcell.weight.dtype))
                    if isinstance(subcell, nn.Dense) and subcell.bias is not None:
                        subcell.bias.set_data(initializer('zeros', subcell.bias.shape, subcell.bias.dtype))
                elif isinstance(subcell, nn.LayerNorm):
                    subcell.bias.set_data(initializer('zeros', subcell.bias.shape, subcell.bias.dtype))
                    subcell.weight.set_data(initializer('ones', subcell.weight.shape, subcell.weight.dtype))
        elif isinstance(cell, OneFormerTextTransformer):
            proj_std = (cell.width**-0.5) * ((2 * cell.num_layers) ** -0.5)
            attn_std = cell.width**-0.5
            fc_std = (2 * cell.width) ** -0.5
            for layer in cell.layers:
                layer.self_attn.in_proj_weight.set_data(initializer(Normal(sigma=attn_std, mean=0), layer.self_attn.in_proj_weight.shape, layer.self_attn.in_proj_weight.dtype))
                layer.self_attn.out_proj.weight.set_data(initializer(Normal(sigma=proj_std, mean=0), layer.self_attn.out_proj.weight.shape, layer.self_attn.out_proj.weight.dtype))
                layer.mlp.fc1.weight.set_data(initializer(Normal(sigma=fc_std, mean=0), layer.mlp.fc1.weight.shape, layer.mlp.fc1.weight.dtype))
                layer.mlp.fc2.weight.set_data(initializer(Normal(sigma=proj_std, mean=0), layer.mlp.fc2.weight.shape, layer.mlp.fc2.weight.dtype))
        elif isinstance(cell, OneFormerTextEncoder):
            cell.token_embedding.weight.set_data(initializer(Normal(sigma=0.02, mean=0), cell.token_embedding.weight.shape, cell.token_embedding.weight.dtype))
            cell.positional_embedding.set_data(initializer(Normal(sigma=0.01, mean=0), cell.positional_embedding.shape, cell.positional_embedding.dtype))
        if hasattr(cell, "reference_points"):
            cell.reference_points.weight.set_data(initializer(XavierUniform(1.0), cell.reference_points.weight.shape, cell.reference_points.weight.dtype))
            cell.reference_points.set_data(initializer('zeros', cell.reference_points.shape, cell.reference_points.dtype))
        elif isinstance(cell, OneFormerTaskModel):
            for subcell in cell.cells():
                if isinstance(cell, OneFormerMLPPredictionHead):
                    for subcell in cell.cells():
                        if isinstance(subcell, nn.Dense):
                            subcell.weight.set_data(initializer(XavierUniform(xavier_std), subcell.weight.shape, subcell.weight.dtype))
                            subcell.bias.set_data(initializer('zeros', subcell.bias.shape, subcell.bias.dtype))
                        elif isinstance(cell, nn.LayerNorm):
                            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
                            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
        elif isinstance(cell, nn.MultiheadAttention):
            cell.in_proj_weight.set_data(initializer(Normal(sigma=std, mean=0), cell.in_proj_weight.shape, cell.in_proj_weight.dtype))
            cell.in_proj_bias.set_data(initializer('zeros', cell.in_proj_bias.shape, cell.in_proj_bias.dtype))
        elif isinstance(cell, (nn.Dense, nn.Conv2d, nn.BatchNorm2d)):
            cell.weight.set_data(initializer(Normal(std), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            cell.weight.set_data(initializer(Normal(std), cell.weight.shape, cell.weight.dtype))
            if cell.padding_idx is not None:
                cell.weight.data[cell.padding_idx] = 0


class OneFormerModel(OneFormerPreTrainedModel):
    main_input_name = ["pixel_values", "task_inputs"]

    def __init__(self, config: OneFormerConfig):
        super().__init__(config)
        self.pixel_level_module = OneFormerPixelLevelModule(config)
        self.transformer_module = OneFormerTransformerModule(in_features=config.conv_dim, config=config)
        self.task_encoder = OneFormerTaskModel(config)
        self.is_training = config.is_training

        if self.is_training:
            self.text_mapper = OneFormerTextMapper(config)
        else:
            self.text_mapper = None

        self.post_init()

    def construct(
        self,
        pixel_values: Tensor,
        task_inputs: Tensor,
        text_inputs: Optional[Tensor] = None,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> OneFormerModelOutput:
        r"""
        Returns:
            `OneFormerModelOutput`
        Example:

        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import OneFormerProcessor, OneFormerModel

        >>> # download texting image
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # load processor for preprocessing the inputs
        >>> processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        >>> model = OneFormerModel.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        >>> inputs = processor(image, ["semantic"], return_tensors="ms")

        >>> # TODO: remove line
        ...     outputs = model(**inputs)

        >>> mask_predictions = outputs.transformer_decoder_mask_predictions
        >>> class_predictions = outputs.transformer_decoder_class_predictions

        >>> f"👉 Mask Predictions Shape: {list(mask_predictions.shape)}, Class Predictions Shape: {list(class_predictions.shape)}"
        '👉 Mask Predictions Shape: [1, 150, 128, 171], Class Predictions Shape: [1, 150, 151]'
        ```"""

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, _, height, width = pixel_values.shape

        if pixel_mask is None:
            pixel_mask = ops.ones((batch_size, height, width))

        pixel_level_module_output = self.pixel_level_module(pixel_values, output_hidden_states)

        multi_scale_features = pixel_level_module_output.decoder_features
        mask_features = pixel_level_module_output.decoder_last_feature

        task_token = self.task_encoder(task_inputs.to(self.dtype))

        if self.is_training:
            text_queries = self.text_mapper(text_inputs)
        else:
            text_queries = None

        transformer_module_output = self.transformer_module(
            multi_scale_features=multi_scale_features,
            mask_features=mask_features,
            task_token=task_token,
            output_attentions=output_attentions,
        )

        queries = transformer_module_output.object_queries

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None

        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output.encoder_features
            pixel_decoder_hidden_states = (pixel_level_module_output.decoder_last_feature,)
            for f in pixel_level_module_output.decoder_features:
                pixel_decoder_hidden_states += (f,)
            transformer_decoder_hidden_states = transformer_module_output.auxiliary_predictions

        output = OneFormerModelOutput(
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            transformer_decoder_object_queries=queries,
            transformer_decoder_contrastive_queries=transformer_module_output.contrastive_logits,
            transformer_decoder_mask_predictions=transformer_module_output.prediction_masks,
            transformer_decoder_class_predictions=transformer_module_output.prediction_class,
            transformer_decoder_auxiliary_predictions=transformer_module_output.auxiliary_predictions,
            text_queries=text_queries,
            task_token=task_token,
            attentions=transformer_module_output.attentions,
        )

        if not return_dict:
            output = tuple(v for v in output.values())

        return output


class OneFormerForUniversalSegmentation(OneFormerPreTrainedModel):
    main_input_name = ["pixel_values", "task_inputs"]

    def __init__(self, config: OneFormerConfig):
        super().__init__(config)
        self.model = OneFormerModel(config)

        self.matcher = OneFormerHungarianMatcher(
            cost_class=config.class_weight,
            cost_dice=config.dice_weight,
            cost_mask=config.mask_weight,
            num_points=config.train_num_points,
        )

        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
            "loss_contrastive": config.contrastive_weight,
        }

        self.criterion = OneFormerLoss(
            num_classes=config.num_labels,
            matcher=self.matcher,
            weight_dict=self.weight_dict,
            eos_coef=config.no_object_weight,
            num_points=config.train_num_points,
            oversample_ratio=config.oversample_ratio,
            importance_sample_ratio=config.importance_sample_ratio,
            contrastive_temperature=config.contrastive_temperature,
        )

        self.post_init()

    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        contrastive_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        text_queries: Tensor,
        auxiliary_predictions: Dict[str, Tensor],
        calculate_contrastive_loss: bool,
    ) -> Dict[str, Tensor]:
        loss_dict: Dict[str, Tensor] = self.criterion(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            contrastive_queries_logits=contrastive_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            text_queries=text_queries,
            auxiliary_predictions=auxiliary_predictions,
            calculate_contrastive_loss=calculate_contrastive_loss,
        )

        # weight each loss by `self.weight_dict[<LOSS_NAME>]` including auxiliary losses
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict

    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    def construct(
        self,
        pixel_values: Tensor,
        task_inputs: Tensor,
        text_inputs: Optional[Tensor] = None,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        pixel_mask: Optional[Tensor] = None,
        output_auxiliary_logits: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> OneFormerForUniversalSegmentationOutput:
        r"""
        text_inputs (`List[mindspore.Tensor]`, *optional*):
            Tensor fof shape `(num_queries, sequence_length)` to be fed to a model
        mask_labels (`List[mindspore.Tensor]`, *optional*):
            List of mask labels of shape `(num_labels, height, width)` to be fed to a model
        class_labels (`List[mindspore.Tensor]`, *optional*):
            list of target class labels of shape `(num_labels, height, width)` to be fed to a model. They identify the
            labels of `mask_labels`, e.g. the label of `mask_labels[i][j]` if `class_labels[i][j]`.

        Returns:
            `OneFormerUniversalSegmentationOutput`
        Example:

        Universal segmentation example:

        ```python
        >>> from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> # load OneFormer fine-tuned on ADE20k for universal segmentation
        >>> processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        >>> model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")

        >>> url = (
        ...     "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
        ... )
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # Semantic Segmentation
        >>> inputs = processor(image, ["semantic"], return_tensors="ms")

        >>> # TODO: remove line
        ...     outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to processor for semantic postprocessing
        >>> predicted_semantic_map = processor.post_process_semantic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]
        >>> f"👉 Semantic Predictions Shape: {list(predicted_semantic_map.shape)}"
        '👉 Semantic Predictions Shape: [512, 683]'

        >>> # Instance Segmentation
        >>> inputs = processor(image, ["instance"], return_tensors="ms")

        >>> # TODO: remove line
        ...     outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to processor for instance postprocessing
        >>> predicted_instance_map = processor.post_process_instance_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]["segmentation"]
        >>> f"👉 Instance Predictions Shape: {list(predicted_instance_map.shape)}"
        '👉 Instance Predictions Shape: [512, 683]'

        >>> # Panoptic Segmentation
        >>> inputs = processor(image, ["panoptic"], return_tensors="ms")

        >>> # TODO: remove line
        ...     outputs = model(**inputs)
        >>> # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        >>> # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        >>> class_queries_logits = outputs.class_queries_logits
        >>> masks_queries_logits = outputs.masks_queries_logits

        >>> # you can pass them to processor for panoptic postprocessing
        >>> predicted_panoptic_map = processor.post_process_panoptic_segmentation(
        ...     outputs, target_sizes=[image.size[::-1]]
        ... )[0]["segmentation"]
        >>> f"👉 Panoptic Predictions Shape: {list(predicted_panoptic_map.shape)}"
        '👉 Panoptic Predictions Shape: [512, 683]'
        ```
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            pixel_values=pixel_values,
            task_inputs=task_inputs,
            text_inputs=text_inputs,
            pixel_mask=pixel_mask,
            output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss,
            output_attentions=output_attentions,
            return_dict=True,
        )

        loss, loss_dict, auxiliary_predictions = None, None, None

        class_queries_logits = outputs.transformer_decoder_class_predictions
        masks_queries_logits = outputs.transformer_decoder_mask_predictions
        contrastive_queries_logits = outputs.transformer_decoder_contrastive_queries
        auxiliary_predictions = outputs.transformer_decoder_auxiliary_predictions
        text_queries = outputs.text_queries

        if mask_labels is not None and class_labels is not None:
            loss_dict: Dict[str, Tensor] = self.get_loss_dict(
                masks_queries_logits=masks_queries_logits,
                class_queries_logits=class_queries_logits,
                contrastive_queries_logits=contrastive_queries_logits,
                mask_labels=mask_labels,
                class_labels=class_labels,
                text_queries=text_queries,
                auxiliary_predictions=auxiliary_predictions,
                calculate_contrastive_loss=self.config.contrastive_temperature is not None,
            )
            loss = self.get_loss(loss_dict)

        output_auxiliary_logits = (
            self.config.output_auxiliary_logits if output_auxiliary_logits is None else output_auxiliary_logits
        )
        if not output_auxiliary_logits:
            auxiliary_predictions = None

        output = OneFormerForUniversalSegmentationOutput(
            class_queries_logits=class_queries_logits,
            masks_queries_logits=masks_queries_logits,
            auxiliary_predictions=auxiliary_predictions,
            loss=loss,
            **outputs,
        )

        if not return_dict:
            output = tuple(v for v in output.values())
            if loss is not None:
                output = (loss) + output
        return output


__all__ = [
    "OneFormerModel",
    "OneFormerPreTrainedModel",
    "OneFormerForUniversalSegmentation",
]
