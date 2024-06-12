# import torch.nn.functional as F
# from torch import nn
# import torch

import mindspore
from mindspore import nn, ops


def hcl(fstudent, fteacher):
    loss_all = 0.0
    # for the last feature of the fstudent and fteacher,
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = ops.mse_loss(fs, ft, reduction='mean')   # calculate MSE of the original feature
        cnt = 1.0   # weight = 1.0
        tot = 1.0   # sum of weight = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = ops.adaptive_avg_pool2d(fs, (l, l))   # adaptive average pooling
            tmpft = ops.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0   # halve weight
            loss += ops.mse_loss(tmpfs, tmpft, reduction='mean') * cnt   # sum of loss * weight
            tot += cnt
        loss = loss / tot   # weighted mean
        loss_all = loss_all + loss

    return loss_all


class DistillKL(nn.Cell):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    # KL divergence
    def construct(self, y_s, y_t):
        p_s = ops.log_softmax(y_s / self.T, axis=1)
        p_t = ops.softmax(y_t / self.T, axis=1)
        loss = ops.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss


class DKDLoss(nn.Cell):
    def __init__(self, T, alpha=1.0, beta=8.0):
        super(DKDLoss, self).__init__()
        self.T = T
        self.alpha = alpha
        self.beta = beta

    def construct(self, logits_student, logits_teacher, target):
        gt_mask = _get_gt_mask(logits_student, target)   # get mask of target class
        other_mask = _get_other_mask(logits_student, target)   # get mask of non target class

        pred_student = ops.softmax(logits_student / self.T, axis=1)
        pred_teacher = ops.softmax(logits_teacher / self.T, axis=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)   # two columns
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)   # probability of correct class and incorrect class
        log_pred_student = ops.log(pred_student)
        tckd_loss = (
            ops.kl_div(log_pred_student, pred_teacher)
            * (self.T**2)
            / target.shape[0]
        )

        pred_teacher_part2 = ops.softmax(
            logits_teacher / self.T - 1000.0 * gt_mask, axis=1
        )
        log_pred_student_part2 = ops.log_softmax(
            logits_student / self.T - 1000.0 * gt_mask, axis=1
        )
        nckd_loss = (
            ops.kl_div(log_pred_student_part2, pred_teacher_part2)
            * (self.T**2)
            / target.shape[0]
        )

        return self.alpha * tckd_loss + self.beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)   # Convert to one row
    mask = ops.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()     # one-hot encoding
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = ops.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()   # negate one-hot encoding
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = ops.cat([t1, t2], axis=1)
    return rt


class ReLoss(nn.Cell):
    """
    1. use PKDLoss on the logit
    2. use PKDLoss on the feature
    3. use PKDLoss both on the logit and feature
    """

    def __init__(self, n_cls, k=3):
        super(ReLoss, self).__init__()
        self.n_cls = n_cls
        self.k = k
        self.matmul = nn.MatMul()

    def construct(self, target, logit_s, logit_t):
        """construct the prior knowledge"""
        l2_normalize = ops.L2Normalize(axis=1)
        # target = target.astype(mindspore.int64)
        one_hot = ops.one_hot(target, depth=self.n_cls).float()
        # prior = ops.einsum("aj, bj->ab", one_hot, one_hot)   # get label correlations matrix
        prior = self.matmul(one_hot, ops.t(one_hot))

        """Diagonal position elements are not considered"""
        cc_mask = (prior - ops.eye(prior.shape[0])) * self.k   # (C-E)*ρ
        ncc_mask = ops.ones(shape=prior.shape) - prior  # keep the inter-class similarity
        prior = cc_mask + ncc_mask

        bsz = logit_s.shape[0]
        logit_s = logit_s.view(bsz, -1)
        logit_t = logit_t.view(bsz, -1)

        # emd_s = ops.einsum("aj, bj->ab", logit_s, logit_s)   # similarity matrix of student
        emd_s = self.matmul(logit_s, ops.t(logit_s))
        s_mask = 1 - ops.eye(bsz)  # remove the diagonal elements of the student
        emd_s = ops.mul(emd_s, s_mask)
        emd_s = l2_normalize(emd_s)   # L-2 normalization

        # emd_t = ops.einsum("aj, bj->ab", logit_t, logit_t)    # similarity matrix of teacher
        emd_t = self.matmul(logit_t,ops.t(logit_t))
        """prior knowledge enhance"""
        emd_t = ops.mul(emd_t, prior)  # prior similarity enhancement
        # emd_t = torch.mul(emd_t, s_mask)  # remove the diagonal elements of the teacher
        emd_t = l2_normalize(emd_t)

        diff = emd_s - emd_t
        loss = (diff * diff).view(bsz, -1).sum() / bsz   # MSE loss

        return loss


if __name__ == '__main__':
    kd = ReLoss(5)
    f_t = ops.randn(5, 64, 8, 8)
    f_s = ops.randn(5, 64, 8, 8)
    x1 = ops.randn(5, 5)
    x2 = ops.randn(5, 5)

    target = ops.randint(0,5, (5,))
    print(target)
    kd_loss = kd(target, x1, x2)
    print(kd_loss)
