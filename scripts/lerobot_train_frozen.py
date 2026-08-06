#!/usr/bin/env python3
"""Run LeRobot training with an explicit parameter-freezing profile.

The upstream trainer creates the policy before it creates the optimizer.  This
wrapper freezes the selected parameters at that boundary, so LeRobot's normal
``get_optim_params`` method includes only the remaining trainable parameters.
All dataset, policy, checkpoint, and resume arguments are still parsed by the
installed LeRobot trainer.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


FREEZE_PROFILES: dict[str, tuple[str, ...]] = {
    # Most conservative edge adaptation: change only visual cross-attention,
    # its normalization, and the final action readout.  Keep the decoder's
    # self-attention/FFN/position embedding as the learned motion prior.
    "edge-head": (
        "model.decoder.layers.0.multihead_attn.",
        "model.decoder.layers.0.norm2.",
        "model.action_head.",
    ),
    # Preserve the visual representation, state encoder, and VAE.  Let the
    # decoder attend differently to existing features and adjust final actions.
    "decoder-head": (
        "model.decoder.",
        "model.decoder_pos_embed.",
        "model.action_head.",
    ),
}


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--freeze-profile",
        choices=sorted(FREEZE_PROFILES),
        default=None,
    )
    parser.add_argument("--preserve-pretrained-stats", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    if args.freeze_profile is None and not any(
        value in ("-h", "--help") for value in remaining
    ):
        parser.error("--freeze-profile is required")
    return args, remaining


def cli_value(args: list[str], name: str) -> str | None:
    """Read either ``--name value`` or ``--name=value`` without consuming it."""
    flag = f"--{name}"
    for index, value in enumerate(args):
        if value == flag and index + 1 < len(args):
            return args[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    return None


def output_dir_from_args(args: list[str]) -> Path | None:
    value = cli_value(args, "output_dir")
    if value:
        return Path(value).expanduser().resolve()
    config_path = cli_value(args, "config_path")
    if not config_path:
        return None
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return Path(config["output_dir"]).expanduser().resolve()
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None


def preserve_pretrained_processor_stats(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop only dataset-stat overrides while retaining device/key overrides."""
    updated = copy.deepcopy(kwargs)
    preprocessor_overrides = updated.get("preprocessor_overrides") or {}
    postprocessor_overrides = updated.get("postprocessor_overrides") or {}
    preprocessor_overrides.pop("normalizer_processor", None)
    postprocessor_overrides.pop("unnormalizer_processor", None)
    updated["preprocessor_overrides"] = preprocessor_overrides
    updated["postprocessor_overrides"] = postprocessor_overrides
    updated.pop("dataset_stats", None)
    return updated


def write_or_validate_freeze_metadata(
    output_dir: Path | None,
    summary: dict[str, Any],
    preserve_stats: bool,
) -> None:
    if output_dir is None:
        print("[freeze] warning: output_dir unavailable; metadata not saved", flush=True)
        return
    path = output_dir / "freeze_profile.json"
    metadata = {
        "schema": "linkerhand_lerobot_freeze_profile_v1",
        **summary,
        "preserve_pretrained_stats": preserve_stats,
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        keys = (
            "profile",
            "trainable_prefixes",
            "trainable_names",
            "preserve_pretrained_stats",
        )
        mismatched = [key for key in keys if existing.get(key) != metadata.get(key)]
        if mismatched:
            raise RuntimeError(
                f"freeze metadata mismatch in {path}: {', '.join(mismatched)}"
            )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def apply_freeze_profile(policy: Any, profile: str) -> dict[str, Any]:
    """Freeze every parameter except prefixes allowed by ``profile``."""
    if getattr(getattr(policy, "config", None), "type", None) != "act":
        raise RuntimeError("freeze profiles are defined only for ACT policies")
    trainable_prefixes = FREEZE_PROFILES[profile]
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    trainable_count = 0
    total_count = 0

    for name, parameter in policy.named_parameters():
        is_trainable = name.startswith(trainable_prefixes)
        parameter.requires_grad_(is_trainable)
        total_count += parameter.numel()
        if is_trainable:
            trainable_names.append(name)
            trainable_count += parameter.numel()
        else:
            frozen_names.append(name)

    if not trainable_names:
        raise RuntimeError(f"freeze profile {profile!r} selected no parameters")
    missing_prefixes = [
        prefix
        for prefix in trainable_prefixes
        if not any(name.startswith(prefix) for name in trainable_names)
    ]
    if missing_prefixes:
        raise RuntimeError(
            f"freeze profile prefixes matched no parameters: {missing_prefixes}"
        )

    return {
        "profile": profile,
        "trainable_prefixes": list(trainable_prefixes),
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_count": total_count - trainable_count,
        "total_parameter_count": total_count,
        "trainable_tensor_count": len(trainable_names),
        "frozen_tensor_count": len(frozen_names),
        "trainable_names": trainable_names,
    }


def format_summary(summary: dict[str, Any]) -> Iterable[str]:
    yield (
        "[freeze] profile={profile} trainable={trainable_parameter_count}/"
        "{total_parameter_count} ({percent:.2f}%) tensors={trainable_tensor_count}"
    ).format(
        **summary,
        percent=(
            100.0
            * summary["trainable_parameter_count"]
            / summary["total_parameter_count"]
        ),
    )
    for prefix in summary["trainable_prefixes"]:
        yield f"[freeze] trainable prefix: {prefix}"


def main() -> None:
    wrapper_args, lerobot_args = parse_wrapper_args(sys.argv[1:])

    import lerobot.scripts.lerobot_train as lerobot_train

    original_make_policy = lerobot_train.make_policy
    original_make_processors = lerobot_train.make_pre_post_processors
    original_make_optimizer = lerobot_train.make_optimizer_and_scheduler
    output_dir = output_dir_from_args(lerobot_args)

    def make_frozen_policy(*args: Any, **kwargs: Any) -> Any:
        policy = original_make_policy(*args, **kwargs)
        summary = apply_freeze_profile(policy, wrapper_args.freeze_profile)
        write_or_validate_freeze_metadata(
            output_dir,
            summary,
            wrapper_args.preserve_pretrained_stats,
        )
        for line in format_summary(summary):
            print(line, flush=True)
        return policy

    def make_processors(*args: Any, **kwargs: Any) -> Any:
        if wrapper_args.preserve_pretrained_stats:
            kwargs = preserve_pretrained_processor_stats(kwargs)
            print("[freeze] processor stats: preserved from pretrained checkpoint", flush=True)
        return original_make_processors(*args, **kwargs)

    def make_checked_optimizer(*args: Any, **kwargs: Any) -> Any:
        optimizer, scheduler = original_make_optimizer(*args, **kwargs)
        optimized = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        policy = args[1] if len(args) > 1 else kwargs["policy"]
        expected = {
            id(parameter) for parameter in policy.parameters() if parameter.requires_grad
        }
        if optimized != expected:
            raise RuntimeError(
                "optimizer parameters do not exactly match the freeze profile"
            )
        return optimizer, scheduler

    lerobot_train.make_policy = make_frozen_policy
    lerobot_train.make_pre_post_processors = make_processors
    lerobot_train.make_optimizer_and_scheduler = make_checked_optimizer
    sys.argv = [sys.argv[0], *lerobot_args]
    lerobot_train.main()


if __name__ == "__main__":
    main()
