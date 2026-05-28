import torch
from compression.losses import kl_discret_loss, kd_simcc_loss


def test_kl_discret_zero_when_pred_matches_target():
    target = torch.softmax(torch.randn(2, 17, 384), dim=-1)
    pred_logits = torch.log(target + 1e-9)  # softmax(log p) == p
    vis = torch.ones(2, 17)
    loss = kl_discret_loss(pred_logits, target, vis)
    assert loss.item() < 1e-4


def test_kl_discret_ignores_invisible_keypoints():
    target = torch.softmax(torch.randn(2, 17, 384), dim=-1)
    pred_logits = torch.randn(2, 17, 384)
    vis = torch.zeros(2, 17)  # nothing visible -> zero loss
    loss = kl_discret_loss(pred_logits, target, vis)
    assert loss.item() == 0.0


def test_kd_loss_decreases_as_student_approaches_teacher():
    t_logits = torch.randn(2, 17, 384)
    far = kd_simcc_loss(torch.zeros_like(t_logits), t_logits, T=1.0)
    near = kd_simcc_loss(t_logits.clone(), t_logits, T=1.0)
    assert near.item() < far.item()
