"""SimCC training losses.

kl_discret_loss: student logits vs Gaussian GT target (RTMPose's GT loss),
                 masked by per-keypoint visibility.
kd_simcc_loss:   student logits vs teacher logits (knowledge distillation),
                 temperature-softened KL on both axes.
"""
import torch
import torch.nn.functional as F


def _axis_kl(pred_logits, target_prob, vis):
    logp = F.log_softmax(pred_logits, dim=-1)          # (B, K, bins)
    kl = (target_prob * (torch.log(target_prob + 1e-9) - logp)).sum(-1)  # (B, K)
    denom = vis.sum().clamp(min=1.0)
    return (kl * vis).sum() / denom


def kl_discret_loss(pred_logits, target_prob, vis):
    """pred_logits/target on one axis: (B, K, bins). target is a (sums-to-1) prob.
    Here target may be an un-normalized Gaussian; normalize it first."""
    target_prob = target_prob / (target_prob.sum(-1, keepdim=True) + 1e-9)
    return _axis_kl(pred_logits, target_prob, vis)


def kd_simcc_loss(student_logits, teacher_logits, T: float = 1.0):
    """Temperature-softened KL(teacher || student) on one axis. Mean over all kpts."""
    t = F.softmax(teacher_logits / T, dim=-1)
    logs = F.log_softmax(student_logits / T, dim=-1)
    kl = (t * (torch.log(t + 1e-9) - logs)).sum(-1)    # (B, K)
    return (T * T) * kl.mean()
