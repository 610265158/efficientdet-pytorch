import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List

from effdet.anchors import decode_box_outputs


def focal_loss(logits, targets, alpha: float, gamma: float, normalizer):
    """Compute the focal loss between `logits` and the golden `target` values.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Args:
        logits: A float32 tensor of size [batch, height_in, width_in, num_predictions].

        targets: A float32 tensor of size [batch, height_in, width_in, num_predictions].

        alpha: A float32 scalar multiplying alpha to the loss from positive examples
            and (1-alpha) to the loss from negative examples.

        gamma: A float32 scalar modulating loss from hard and easy examples.

         normalizer: A float32 scalar normalizes the total loss from all examples.

    Returns:
        loss: A float32 scalar representing normalized total loss.
    """

    positive_label_mask = targets == 1.0
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction='none')
    # Below are comments/derivations for computing modulator.
    # For brevity, let x = logits,  z = targets, r = gamma, and p_t = sigmod(x)
    # for positive samples and 1 - sigmoid(x) for negative examples.
    #
    # The modulator, defined as (1 - P_t)^r, is a critical part in focal loss
    # computation. For r > 0, it puts more weights on hard examples, and less
    # weights on easier ones. However if it is directly computed as (1 - P_t)^r,
    # its back-propagation is not stable when r < 1. The implementation here
    # resolves the issue.
    #
    # For positive samples (labels being 1),
    #    (1 - p_t)^r
    #  = (1 - sigmoid(x))^r
    #  = (1 - (1 / (1 + exp(-x))))^r
    #  = (exp(-x) / (1 + exp(-x)))^r
    #  = exp(log((exp(-x) / (1 + exp(-x)))^r))
    #  = exp(r * log(exp(-x)) - r * log(1 + exp(-x)))
    #  = exp(- r * x - r * log(1 + exp(-x)))
    #
    # For negative samples (labels being 0),
    #    (1 - p_t)^r
    #  = (sigmoid(x))^r
    #  = (1 / (1 + exp(-x)))^r
    #  = exp(log((1 / (1 + exp(-x)))^r))
    #  = exp(-r * log(1 + exp(-x)))
    #
    # Therefore one unified form for positive (z = 1) and negative (z = 0)
    # samples is:
    #      (1 - p_t)^r = exp(-r * z * x - r * log(1 + exp(-x))).
    neg_logits = -1.0 * logits
    modulator = torch.exp(gamma * targets * neg_logits - gamma * torch.log1p(torch.exp(neg_logits)))
    loss = modulator * cross_entropy
    weighted_loss = torch.where(positive_label_mask, alpha * loss, (1.0 - alpha) * loss)
    weighted_loss /= normalizer
    return weighted_loss


def huber_loss(
        input, target, delta: float = 1., weights: Optional[torch.Tensor] = None, size_average: bool = True):
    """
    """
    err = input - target
    abs_err = err.abs()
    quadratic = torch.clamp(abs_err, max=delta)
    linear = abs_err - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    if weights is not None:
        loss *= weights
    return loss.mean() if size_average else loss.sum()


def smooth_l1_loss(
        input, target, beta: float = 1. / 9, weights: Optional[torch.Tensor] = None, size_average: bool = True):
    """
    very similar to the smooth_l1_loss from pytorch, but with the extra beta parameter
    """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        err = torch.abs(input - target)
        loss = torch.where(err < beta, 0.5 * err.pow(2) / beta, err - 0.5 * beta)
    if weights is not None:
        loss *= weights
    return loss.mean() if size_average else loss.sum()


def _classification_loss(cls_outputs, cls_targets, num_positives, alpha: float = 0.25, gamma: float = 2.0):
    """Computes classification loss."""
    normalizer = num_positives
    classification_loss = focal_loss(cls_outputs, cls_targets, alpha, gamma, normalizer)
    return classification_loss


# def _box_loss(anchor_boxes,box_outputs, box_targets, num_positives, delta: float = 0.1):
#     """Computes box regression loss."""
#     # delta is typically around the mean value of regression target.
#     # for instances, the regression targets of 512x512 input with 6 anchors on
#     # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
#     normalizer = num_positives * 4.0
#     mask = box_targets != 0.0
#     box_loss = huber_loss(box_targets, box_outputs, weights=mask, delta=delta, size_average=False)
#
#     box_loss /= normalizer
#
#     return box_loss

def _box_loss(anchor_boxes,box_outputs, box_targets, num_positives,cls_targets_non_neg, delta: float = 0.1):
    """Computes box regression loss."""
    # delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
    # normalizer = num_positives * 4.0
    # mask = box_targets != 0.0
    # box_loss = huber_loss(box_targets, box_outputs, weights=mask, delta=delta, size_average=False)
    # box_loss /= normalizer
    box_loss = ciou_loss(anchor_boxes,box_outputs,box_targets,cls_targets_non_neg)


    return box_loss

def ciou_loss(anchor_boxes,box_outputs, box_targets,weights,avg_factor=None,eps=1.e-6):



    bs=box_outputs.shape[0]

    for i in range(bs):
        box_outputs[i] = decode_box_outputs(box_outputs[i], anchor_boxes, output_xyxy=True)
        box_targets[i] = decode_box_outputs(box_targets[i], anchor_boxes, output_xyxy=True)

    pos_mask=weights>0

    if avg_factor is None:
        avg_factor = torch.sum(pos_mask) + eps


    bboxes1 =box_outputs[pos_mask,...].view(-1, 4).float()
    bboxes2 = box_targets[pos_mask,...].view(-1, 4).float()


    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
    # enclose_x1y1 = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    # enclose_x2y2 = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    # enclose_wh =  torch.max((enclose_x2y2 - enclose_x1y1 + 1),0)

    overlap = wh[:, 0] * wh[:, 1]
    ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (ap + ag - overlap)

    # cal outer boxes
    outer_left_up = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    outer_right_down = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    outer = (outer_right_down - outer_left_up).clamp(min=0)
    outer_diagonal_line = (outer[:, 0])**2 + (outer[:, 1])**2


    boxes1_center = (bboxes1[:, :2] + bboxes1[:, 2:]+ 1) * 0.5
    boxes2_center = (bboxes2[:, :2] + bboxes2[:, 2:]+ 1) * 0.5
    center_dis = (boxes1_center[:, 0] - boxes2_center[:, 0])**2 + \
                 (boxes1_center[:, 1] - boxes2_center[:, 1])**2

    boxes1_size =(bboxes1[:,2:]-bboxes1[:,:2]).clamp(min=0)
    boxes2_size = (bboxes2[:, 2:] - bboxes2[:, :2]).clamp(min=0)

    v = (4.0 / (np.pi**2)) * \
        ((torch.atan(boxes2_size[:, 0] / (boxes2_size[:, 1]+eps)) -
                    torch.atan(boxes1_size[:, 0] / (boxes1_size[:, 1]+eps)))**2)

    S = (ious >0.5).float()
    alpha = S * v / (1 - ious + v+eps)

    cious = ious - (center_dis / outer_diagonal_line+eps)-alpha * v

    cious = 1-cious

    return torch.sum(cious ) / avg_factor



class DetectionLoss(nn.Module):
    def __init__(self, config):
        super(DetectionLoss, self).__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.delta = config.delta
        self.box_loss_weight = config.box_loss_weight

    def forward(self,
                anchor_boxes,
                cls_outputs: List[torch.Tensor],
                box_outputs: List[torch.Tensor],
                cls_targets: List[torch.Tensor],
                box_targets: List[torch.Tensor],
                num_positives: torch.Tensor):
        """Computes total detection loss.
        Computes total detection loss including box and class loss from all levels.
        Args:
            cls_outputs: a List with values representing logits in [batch_size, height, width, num_anchors].
                at each feature level (index)

            box_outputs: a List with values representing box regression targets in
                [batch_size, height, width, num_anchors * 4] at each feature level (index)

            cls_targets: groundtruth class targets.

            box_targets: groundtrusth box targets.

            num_positives: num positive grountruth anchors

        Returns:
            total_loss: an integer tensor representing total loss reducing from class and box losses from all levels.

            cls_loss: an integer tensor representing total class loss.

            box_loss: an integer tensor representing total box regression loss.
        """
        # Sum all positives in a batch for normalization and avoid zero
        # num_positives_sum, which would lead to inf loss during training
        num_positives_sum = num_positives.sum() + 1.0
        levels = len(cls_outputs)

        cls_losses = []
        box_losses = []

        ####

        cls_outputs_all=[]
        box_outputs_all=[]
        cls_targets_all=[]
        box_targetsall=[]


        bs=cls_outputs[0].shape[0]

        for l in range(levels):
            cls_outputs[l]=cls_outputs[l].permute(0, 2, 3, 1)
            cls_output_at_level=cls_outputs[l].reshape(bs,-1)
            box_outputs[l]=box_outputs[l].permute(0, 2, 3, 1)
            box_output_at_level=box_outputs[l].reshape(bs,-1, 4)

            cls_outputs_all.append(cls_output_at_level)
            box_outputs_all.append(box_output_at_level)

            cls_targets_at_level = cls_targets[l].reshape(bs,-1)
            box_targets_at_level= box_targets[l].reshape(bs,-1, 4)

            cls_targets_all.append(cls_targets_at_level)
            box_targetsall.append(box_targets_at_level)


        cls_outputs_all=torch.cat(cls_outputs_all, dim=1)
        box_outputs_all = torch.cat(box_outputs_all, dim=1)
        cls_targets_all = torch.cat(cls_targets_all, dim=1)
        box_targetsall = torch.cat(box_targetsall, dim=1)


        # Onehot encoding for classification labels.
        # NOTE: PyTorch one-hot does not handle -ve entries (no hot) like Tensorflow, so mask them out
        cls_targets_non_neg = cls_targets_all >= 0
        cls_targets_all_oh = F.one_hot(cls_targets_all * cls_targets_non_neg, self.num_classes)
        cls_targets_all_oh = torch.where(
           cls_targets_non_neg.unsqueeze(-1), cls_targets_all_oh, torch.zeros_like(cls_targets_all_oh))

        bs,_, _ = cls_targets_all_oh.shape
        cls_targets_all_oh = cls_targets_all_oh.view(bs,  -1)
        cls_loss = _classification_loss(
            cls_outputs_all,
            cls_targets_all_oh,
            num_positives_sum,
            alpha=self.alpha, gamma=self.gamma)
        cls_loss = cls_loss.view(bs, -1, self.num_classes)
        cls_loss *= (cls_targets_all != -2).unsqueeze(-1).float()
        cls_losses.append(cls_loss.sum())

        box_losses.append(_box_loss(
            anchor_boxes,
            box_outputs_all,
            box_targetsall,
            num_positives_sum,
            cls_targets_non_neg,
            delta=self.delta))

        # Sum per level losses to total loss.
        cls_loss = torch.sum(torch.stack(cls_losses, dim=-1), dim=-1)
        box_loss = torch.sum(torch.stack(box_losses, dim=-1), dim=-1)
        total_loss = cls_loss + self.box_loss_weight * box_loss
        return total_loss, cls_loss, box_loss

