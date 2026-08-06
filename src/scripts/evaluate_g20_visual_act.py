#!/usr/bin/env python3
"""Offline validation for the camera-conditioned G20 ACT policy.

This script reads only recorded validation episodes.  It never imports ROS,
opens a live camera, or publishes commands to the hand.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "g20_visual_act"
DEFAULT_TRAIN_PYTHON = Path(
    "/home/zhaoyan-qian/Desktop/Jacky/ros2_pairlab3-main/"
    ".venv_ros2_pairlab3/bin/python"
)
RESERVED_JOINTS = {11, 12, 13, 14}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="pretrained_model directory; default is the newest numbered checkpoint",
    )
    ap.add_argument("--samples-per-episode", type=int, default=20)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    if args.samples_per_episode <= 0:
        ap.error("--samples-per-episode must be positive")
    return args


def ensure_lerobot_python() -> None:
    try:
        import lerobot  # noqa: F401
        import torch  # noqa: F401
        return
    except ImportError as exc:
        if not DEFAULT_TRAIN_PYTHON.is_file():
            raise RuntimeError("LeRobot training environment was not found") from exc
        # Do not use resolve(): both a venv interpreter and /usr/bin/python3
        # ultimately resolve to the same binary while loading different envs.
        if Path(sys.executable).absolute() == DEFAULT_TRAIN_PYTHON.absolute():
            raise RuntimeError("Training Python cannot import LeRobot") from exc
        print(f"[env] re-executing with {DEFAULT_TRAIN_PYTHON}", flush=True)
        os.execv(
            str(DEFAULT_TRAIN_PYTHON),
            [str(DEFAULT_TRAIN_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        )


def newest_checkpoint(artifact_root: Path) -> Path:
    checkpoint_root = artifact_root / "training" / "checkpoints"
    candidates = sorted(
        (p for p in checkpoint_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    if not candidates:
        raise RuntimeError(f"No numbered checkpoints found in {checkpoint_root}")
    return candidates[-1] / "pretrained_model"


def load_manifest(artifact_root: Path) -> dict[str, Any]:
    path = artifact_root / "dataset" / "g20_source_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"Missing dataset manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evenly_spaced_indices(dataset: Any, episodes: list[int], per_episode: int) -> list[int]:
    episode_column = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    frame_column = np.asarray(dataset.hf_dataset["frame_index"], dtype=np.int64)
    selected: list[int] = []
    for episode in episodes:
        candidates = np.flatnonzero(episode_column == episode)
        if candidates.size == 0:
            continue
        # Skip only the first/last frame when the episode is long enough.  Those
        # frames often contain the keyboard transition rather than the grasp.
        if candidates.size > 4:
            candidates = candidates[1:-1]
        count = min(per_episode, candidates.size)
        positions = np.linspace(0, candidates.size - 1, count, dtype=np.int64)
        selected.extend(int(candidates[pos]) for pos in positions)
    selected.sort(key=lambda i: (int(episode_column[i]), int(frame_column[i])))
    return selected


def load_policy(checkpoint: Path, device: str) -> tuple[Any, Any, Any]:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(
        str(checkpoint),
        local_files_only=True,
        cli_overrides=["--device", device, "--n_action_steps", "1"],
    )
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(
        str(checkpoint), config=cfg, local_files_only=True
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    policy.eval()
    return policy, preprocessor, postprocessor


def make_plots(
    output_dir: Path,
    records: list[dict[str, Any]],
    images: list[np.ndarray],
    joint_mae: np.ndarray,
    active_mae: float,
    clipped_active_mae: float,
    persistence_mae: float,
) -> None:
    import matplotlib.pyplot as plt

    preview_count = min(9, len(records))
    preview_indices = np.linspace(0, len(records) - 1, preview_count, dtype=int)
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx in zip(axes.flat, preview_indices):
        ax.imshow(images[idx])
        item = records[idx]
        ax.set_title(
            f"ep {item['episode_index']} frame {item['frame_index']}\n"
            f"active-joint MAE {item['active_joint_mae']:.1f} ticks",
            fontsize=10,
        )
    fig.suptitle("Held-out camera frames (never used for training)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_preview.png", dpi=150)
    plt.close(fig)

    active = np.asarray([i not in RESERVED_JOINTS for i in range(len(joint_mae))])
    colors = ["#4472c4" if value else "#b8b8b8" for value in active]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 9))
    ax0.bar(np.arange(len(joint_mae)), joint_mae, color=colors)
    ax0.set_xticks(np.arange(len(joint_mae)))
    ax0.set_xlabel("G20 command index (grey = reserved)")
    ax0.set_ylabel("MAE (SDK ticks, 0–255)")
    ax0.set_title("ACT error by output joint")
    ax0.grid(axis="y", alpha=0.25)

    ax1.bar(
        ["ACT raw", "ACT clamped\nto 0–255", "repeat current\nstate"],
        [active_mae, clipped_active_mae, persistence_mae],
        color=["#4472c4", "#70ad47", "#ed7d31"],
    )
    ax1.set_ylabel("Active-joint MAE (SDK ticks, lower is better)")
    ax1.set_title("Held-out validation comparison")
    ax1.grid(axis="y", alpha=0.25)
    for i, value in enumerate((active_mae, clipped_active_mae, persistence_mae)):
        ax1.text(i, value, f" {value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_metrics.png", dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    ensure_lerobot_python()

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = load_manifest(args.artifact_root)
    validation_episodes = [int(v) for v in manifest["validation_episode_indices"]]
    if not validation_episodes:
        raise RuntimeError("The manifest has no held-out validation episodes")
    checkpoint = args.checkpoint or newest_checkpoint(args.artifact_root)
    if not (checkpoint / "model.safetensors").is_file():
        raise RuntimeError(f"Invalid checkpoint directory: {checkpoint}")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir or args.artifact_root / "evaluation" / checkpoint.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(
        repo_id=str(manifest["repo_id"]),
        root=args.artifact_root / "dataset",
        episodes=validation_episodes,
    )
    indices = evenly_spaced_indices(dataset, validation_episodes, args.samples_per_episode)
    if not indices:
        raise RuntimeError("No validation frames were selected")
    print(
        f"[eval] checkpoint={checkpoint.parent.name} device={device} "
        f"episodes={validation_episodes} samples={len(indices)}",
        flush=True,
    )
    policy, preprocessor, postprocessor = load_policy(checkpoint, device)
    input_keys = list(policy.config.input_features.keys())
    active_mask = np.asarray([i not in RESERVED_JOINTS for i in range(20)])

    records: list[dict[str, Any]] = []
    images: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    states: list[np.ndarray] = []
    for n, index in enumerate(indices, start=1):
        sample = dataset[index]
        obs = {key: sample[key].unsqueeze(0) for key in input_keys}
        processed = preprocessor(obs)
        model_inputs = {key: processed[key] for key in input_keys}
        if hasattr(policy, "reset"):
            policy.reset()
        with torch.inference_mode():
            predicted = policy.select_action(model_inputs)
        predicted = postprocessor(predicted).detach().cpu().numpy().reshape(-1)[:20]
        target = sample["action"].detach().cpu().numpy().reshape(-1)[:20]
        # State-history datasets concatenate oldest -> current; persistence
        # baseline must compare the current (last) 20-D state to the action.
        state = sample["observation.state"].detach().cpu().numpy().reshape(-1)[-20:]
        error = np.abs(predicted - target)
        records.append({
            "dataset_index": int(index),
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["frame_index"]),
            "active_joint_mae": float(error[active_mask].mean()),
            "prediction": predicted.astype(float).tolist(),
            "target": target.astype(float).tolist(),
        })
        image = sample["observation.images.scene"].detach().cpu().numpy()
        images.append(np.clip(np.moveaxis(image, 0, -1), 0.0, 1.0))
        predictions.append(predicted)
        targets.append(target)
        states.append(state)
        if n % 20 == 0 or n == len(indices):
            print(f"[eval] {n}/{len(indices)}", flush=True)

    pred = np.stack(predictions)
    target = np.stack(targets)
    state = np.stack(states)
    joint_mae = np.abs(pred - target).mean(axis=0)
    active_mae = float(np.abs(pred[:, active_mask] - target[:, active_mask]).mean())
    clipped_pred = np.clip(pred, 0.0, 255.0)
    clipped_active_mae = float(
        np.abs(clipped_pred[:, active_mask] - target[:, active_mask]).mean()
    )
    persistence_mae = float(np.abs(state[:, active_mask] - target[:, active_mask]).mean())
    active_out_of_range = (pred[:, active_mask] < 0.0) | (pred[:, active_mask] > 255.0)
    result = {
        "schema": "linkerhand_g20_visual_act_offline_eval_v1",
        "checkpoint": str(checkpoint),
        "validation_episodes": validation_episodes,
        "sample_count": len(records),
        "act_active_joint_mae_ticks": active_mae,
        "act_clipped_active_joint_mae_ticks": clipped_active_mae,
        "repeat_state_active_joint_mae_ticks": persistence_mae,
        "active_predictions_out_of_range": int(active_out_of_range.sum()),
        "active_prediction_count": int(active_out_of_range.size),
        "samples_with_active_prediction_out_of_range": int(
            active_out_of_range.any(axis=1).sum()
        ),
        "act_all_joint_mae_ticks": float(np.abs(pred - target).mean()),
        "per_joint_mae_ticks": joint_mae.astype(float).tolist(),
        "note": "Offline one-step imitation error; this is not a hardware success rate.",
        "samples": records,
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    make_plots(
        output_dir,
        records,
        images,
        joint_mae,
        active_mae,
        clipped_active_mae,
        persistence_mae,
    )
    print(f"[result] ACT active-joint MAE: {active_mae:.2f} SDK ticks")
    print(f"[result] ACT clamped MAE: {clipped_active_mae:.2f} SDK ticks")
    print(f"[result] repeat-state baseline: {persistence_mae:.2f} SDK ticks")
    print(
        "[result] out-of-range active outputs: "
        f"{int(active_out_of_range.sum())}/{int(active_out_of_range.size)} values, "
        f"in {int(active_out_of_range.any(axis=1).sum())}/{len(records)} samples"
    )
    print(f"[result] preview: {output_dir / 'validation_preview.png'}")
    print(f"[result] metrics: {output_dir / 'validation_metrics.png'}")
    print(f"[result] report: {output_dir / 'validation_report.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
