#!/usr/bin/env python3
"""Train LeRobot ACT with an additional demonstrated-direction loss.

The normal ACT L1 objective constrains positions but does not explicitly make
the predicted chunk preserve its demonstrated velocity direction.  This
wrapper adds a small hinge loss on consecutive predicted actions.  It is only
active where the demonstration is moving beyond a deadband, and it uses the
demonstration's sign at every step, so intentional reversals remain legal.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

import torch

try:
    from scripts.lerobot_train_frozen import output_dir_from_args
except ModuleNotFoundError:  # Direct execution puts scripts/ on sys.path.
    from lerobot_train_frozen import output_dir_from_args


G20_ACTIVE_JOINTS = tuple(i for i in range(20) if i not in (11, 12, 13, 14))


def parse_joint_indices(value: str) -> tuple[int, ...]:
    try:
        joints = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("joint indices must be comma-separated integers") from exc
    if not joints or joints[0] < 0 or joints[-1] >= 20:
        raise argparse.ArgumentTypeError("joint indices must be in [0, 19]")
    return joints


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--momentum-weight", type=float, required=True)
    parser.add_argument("--momentum-deadband", type=float, default=0.01)
    parser.add_argument("--momentum-margin", type=float, default=0.005)
    parser.add_argument(
        "--momentum-joints",
        type=parse_joint_indices,
        default=G20_ACTIVE_JOINTS,
    )
    args, remaining = parser.parse_known_args(argv)
    if args.momentum_weight <= 0:
        parser.error("--momentum-weight must be positive")
    if args.momentum_deadband < 0 or args.momentum_margin < 0:
        parser.error("momentum deadband and margin must be non-negative")
    return args, remaining


def demonstrated_direction_loss(
    predicted_actions: torch.Tensor,
    target_actions: torch.Tensor,
    action_is_pad: torch.Tensor,
    *,
    joints: tuple[int, ...] = G20_ACTIVE_JOINTS,
    deadband: float = 0.01,
    margin: float = 0.005,
) -> torch.Tensor:
    """Penalize motion opposite to each demonstrated per-joint direction.

    Actions are already normalized by LeRobot.  For each valid adjacent pair,
    the required signed velocity is capped by ``margin``.  A stationary target
    (within ``deadband``) contributes no loss, while a genuine target reversal
    simply changes the allowed direction at that step.
    """
    if predicted_actions.ndim != 3 or target_actions.shape != predicted_actions.shape:
        raise ValueError("predicted and target actions must have equal (B,T,D) shape")
    if action_is_pad.shape != predicted_actions.shape[:2]:
        raise ValueError("action_is_pad must have shape (B,T)")
    if predicted_actions.shape[1] < 2:
        return predicted_actions.sum() * 0.0

    selected = torch.as_tensor(joints, device=predicted_actions.device)
    predicted_velocity = (
        predicted_actions[:, 1:, selected] - predicted_actions[:, :-1, selected]
    )
    target_velocity = (
        target_actions[:, 1:, selected] - target_actions[:, :-1, selected]
    ).to(predicted_velocity.dtype)
    valid_pairs = ~(action_is_pad[:, 1:] | action_is_pad[:, :-1])
    moving = target_velocity.abs() > deadband
    mask = valid_pairs.unsqueeze(-1) & moving
    if not bool(mask.any()):
        return predicted_actions.sum() * 0.0

    demonstrated_sign = target_velocity.sign()
    required_velocity = torch.clamp(target_velocity.abs(), max=margin)
    aligned_velocity = predicted_velocity * demonstrated_sign
    violations = torch.relu(required_velocity - aligned_velocity)
    return violations.masked_select(mask).mean()


def write_or_validate_metadata(output_dir: Path | None, args: argparse.Namespace) -> None:
    if output_dir is None:
        print("[momentum] warning: output_dir unavailable; metadata not saved", flush=True)
        return
    path = output_dir / "momentum_loss.json"
    metadata = {
        "schema": "linkerhand_act_direction_momentum_v1",
        "weight": args.momentum_weight,
        "deadband_normalized": args.momentum_deadband,
        "margin_normalized": args.momentum_margin,
        "joint_indices": list(args.momentum_joints),
        "loss": "hinge on predicted adjacent-action velocity aligned with demonstrated direction",
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise RuntimeError(f"momentum metadata mismatch in {path}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def install_momentum_forward(policy: Any, args: argparse.Namespace) -> None:
    if getattr(getattr(policy, "config", None), "type", None) != "act":
        raise RuntimeError("momentum loss is defined only for ACT policies")
    from lerobot.utils.constants import ACTION

    original_forward = policy.forward
    captured_actions: list[torch.Tensor] = []

    def capture_actions(_module: Any, _inputs: Any, output: Any) -> None:
        captured_actions.append(output[0])

    # The ACTPolicy forward calls its ACT model once.  Capturing that tensor
    # avoids a second expensive visual/model forward and keeps the exact VAE
    # sample used by the normal behavior-cloning loss.
    policy.model.register_forward_hook(capture_actions)
    forward_calls = 0

    def momentum_forward(
        _policy: Any, batch: dict[str, Any], *_args: Any, **_kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, float]]:
        nonlocal forward_calls
        captured_actions.clear()
        behavior_loss, output = original_forward(batch, *_args, **_kwargs)
        if len(captured_actions) != 1:
            raise RuntimeError(
                f"expected one ACT model forward, captured {len(captured_actions)}"
            )
        if ACTION not in batch or "action_is_pad" not in batch:
            raise RuntimeError("momentum training batch is missing action labels or padding mask")
        momentum_loss = demonstrated_direction_loss(
            captured_actions[0],
            batch[ACTION],
            batch["action_is_pad"],
            joints=args.momentum_joints,
            deadband=args.momentum_deadband,
            margin=args.momentum_margin,
        )
        loss = behavior_loss + args.momentum_weight * momentum_loss
        output["momentum_direction_loss"] = float(momentum_loss.detach())
        output["momentum_weighted_loss"] = float(
            (args.momentum_weight * momentum_loss).detach()
        )
        forward_calls += 1
        if forward_calls == 1 or forward_calls % 100 == 0:
            print(
                f"[momentum] step={forward_calls} bc_l1={output['l1_loss']:.5f} "
                f"direction={output['momentum_direction_loss']:.6f} "
                f"weighted={output['momentum_weighted_loss']:.6f}",
                flush=True,
            )
        return loss, output

    policy.forward = types.MethodType(momentum_forward, policy)


def main() -> None:
    wrapper_args, lerobot_args = parse_wrapper_args(sys.argv[1:])

    import lerobot.scripts.lerobot_train as lerobot_train

    original_make_policy = lerobot_train.make_policy
    output_dir = output_dir_from_args(lerobot_args)

    def make_momentum_policy(*args: Any, **kwargs: Any) -> Any:
        policy = original_make_policy(*args, **kwargs)
        install_momentum_forward(policy, wrapper_args)
        write_or_validate_metadata(output_dir, wrapper_args)
        print(
            f"[momentum] weight={wrapper_args.momentum_weight:g} "
            f"deadband={wrapper_args.momentum_deadband:g} "
            f"margin={wrapper_args.momentum_margin:g} "
            f"joints={list(wrapper_args.momentum_joints)}",
            flush=True,
        )
        return policy

    lerobot_train.make_policy = make_momentum_policy
    sys.argv = [sys.argv[0], *lerobot_args]
    lerobot_train.main()


if __name__ == "__main__":
    main()
