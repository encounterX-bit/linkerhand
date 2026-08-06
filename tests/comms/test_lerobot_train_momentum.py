import torch

from scripts.lerobot_train_momentum import demonstrated_direction_loss


def _pad() -> torch.Tensor:
    return torch.zeros((1, 4), dtype=torch.bool)


def test_direction_loss_is_zero_when_prediction_follows_demo_direction():
    target = torch.tensor([[[0.0], [0.1], [0.2], [0.3]]])
    predicted = torch.tensor([[[0.0], [0.1], [0.2], [0.3]]], requires_grad=True)

    loss = demonstrated_direction_loss(
        predicted, target, _pad(), joints=(0,), deadband=0.01, margin=0.005
    )

    assert loss.item() == 0.0


def test_direction_loss_penalizes_backtracking_but_allows_demo_reversal():
    target = torch.tensor([[[0.0], [0.1], [0.2], [0.1]]])
    follows = torch.tensor([[[0.0], [0.1], [0.2], [0.1]]])
    backtracks_early = torch.tensor([[[0.0], [-0.1], [-0.2], [-0.1]]])

    good = demonstrated_direction_loss(
        follows, target, _pad(), joints=(0,), deadband=0.01, margin=0.005
    )
    bad = demonstrated_direction_loss(
        backtracks_early, target, _pad(), joints=(0,), deadband=0.01, margin=0.005
    )

    assert good.item() == 0.0
    assert bad.item() > 0.0


def test_direction_loss_ignores_stationary_labels_and_padding():
    target = torch.zeros((1, 4, 1))
    predicted = torch.tensor([[[0.0], [-1.0], [1.0], [-1.0]]])
    pad = torch.tensor([[False, False, True, True]])

    loss = demonstrated_direction_loss(
        predicted, target, pad, joints=(0,), deadband=0.01, margin=0.005
    )

    assert loss.item() == 0.0
