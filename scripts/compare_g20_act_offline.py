#!/usr/bin/env python3
"""Compare G20 ACT checkpoints on the same recorded, held-out trajectories.

This is a strictly offline evaluator: it reads JPEGs and joint/action records,
loads local checkpoints, and never imports ROS or publishes a hand command.

The metrics are imitation diagnostics rather than real-robot success rates:

* chunk MAE: clipped 0..255 active-joint error over the predicted action chunk;
* direction disagreement: predicted and demonstrated adjacent action deltas
  have opposite signs when the demonstration moves by at least the deadband;
* boundary jump: discontinuity between the end of one predicted execution
  horizon and the first action after replanning from a recorded observation;
* chunk-end MAE: error at the final predicted action in the chunk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_PYTHON = Path(
    "/home/zhaoyan-qian/Desktop/Jacky/ros2_pairlab3-main/"
    ".venv_ros2_pairlab3/bin/python"
)
JOINT_DIM = 20
RESERVED_JOINTS = {11, 12, 13, 14}
ACTIVE_MASK = np.asarray([i not in RESERVED_JOINTS for i in range(JOINT_DIM)])


@dataclass(frozen=True)
class ModelSpec:
    label: str
    artifact: Path
    checkpoint: Path
    image_offsets: tuple[int, ...]
    state_offsets: tuple[int, ...]
    chunk_size: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=ARTIFACT:STEP",
        help="repeat for each model, for example history=artifacts/model:020000",
    )
    ap.add_argument(
        "--episode",
        action="append",
        required=True,
        type=Path,
        help="raw/clean episode directory containing samples.jsonl; repeatable",
    )
    ap.add_argument("--samples-per-episode", type=int, default=12)
    ap.add_argument("--execution-horizon", type=int, default=30)
    ap.add_argument("--direction-deadband", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "offline_ablation_20260727",
    )
    args = ap.parse_args()
    if args.samples_per_episode <= 0:
        ap.error("--samples-per-episode must be positive")
    if args.execution_horizon <= 0:
        ap.error("--execution-horizon must be positive")
    if args.direction_deadband < 0:
        ap.error("--direction-deadband must be non-negative")
    if args.batch_size <= 0:
        ap.error("--batch-size must be positive")
    args.episode = [p.expanduser().resolve() for p in args.episode]
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def ensure_lerobot_python() -> None:
    try:
        import lerobot  # noqa: F401
        import torch  # noqa: F401
        return
    except ImportError as exc:
        if not LEROBOT_PYTHON.is_file():
            raise RuntimeError("LeRobot environment was not found") from exc
        if Path(sys.executable).absolute() == LEROBOT_PYTHON.absolute():
            raise RuntimeError("LeRobot Python cannot import its dependencies") from exc
        os.execv(
            str(LEROBOT_PYTHON),
            [str(LEROBOT_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        )


def parse_model(value: str) -> ModelSpec:
    if "=" not in value or ":" not in value.rsplit("=", 1)[-1]:
        raise RuntimeError(f"Invalid --model value: {value!r}")
    label, target = value.split("=", 1)
    artifact_text, step = target.rsplit(":", 1)
    artifact = Path(artifact_text).expanduser()
    if not artifact.is_absolute():
        artifact = REPO_ROOT / artifact
    artifact = artifact.resolve()
    checkpoint = artifact / "training" / "checkpoints" / step / "pretrained_model"
    manifest_path = artifact / "dataset" / "g20_source_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing manifest for {label}: {manifest_path}")
    if not (checkpoint / "model.safetensors").is_file():
        raise RuntimeError(f"Missing checkpoint for {label}: {checkpoint}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_offsets = tuple(int(v) for v in manifest.get("history_frame_offsets") or [0])
    state_offsets = tuple(int(v) for v in manifest.get("state_history_offsets") or [0])
    return ModelSpec(
        label=label.strip(),
        artifact=artifact,
        checkpoint=checkpoint,
        image_offsets=image_offsets,
        state_offsets=state_offsets,
        chunk_size=int(manifest.get("chunk_size", 30)),
    )


def read_episode(path: Path) -> list[dict[str, Any]]:
    samples = path / "samples.jsonl"
    if not samples.is_file():
        raise RuntimeError(f"Missing samples.jsonl: {samples}")
    rows = [
        json.loads(line)
        for line in samples.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    valid: list[dict[str, Any]] = []
    for row in rows:
        image_path = Path(str(row.get("image_path", ""))).expanduser()
        if not image_path.is_absolute():
            image_path = path / image_path
        if (
            image_path.is_file()
            and len(row.get("joint_pos", [])) >= JOINT_DIM
            and len(row.get("last_action", [])) >= JOINT_DIM
        ):
            item = dict(row)
            item["_resolved_image_path"] = str(image_path.resolve())
            valid.append(item)
    if not valid:
        raise RuntimeError(f"No valid camera/state/action rows in {samples}")
    return valid


def history_mosaic(
    rows: list[dict[str, Any]], index: int, offsets: tuple[int, ...]
) -> np.ndarray:
    width, height = 320, 240
    images: list[np.ndarray] = []
    for offset in offsets:
        source = rows[max(0, index - offset)]
        bgr = cv2.imread(source["_resolved_image_path"])
        if bgr is None:
            raise RuntimeError(f"Could not read {source['_resolved_image_path']}")
        images.append(bgr)
    if len(images) == 1:
        bgr = cv2.resize(images[0], (width, height), interpolation=cv2.INTER_AREA)
    else:
        if len(images) not in (4, 6):
            raise RuntimeError(f"Only 1, 4, or 6 image offsets are supported: {offsets}")
        grid_rows, columns = (2, 2) if len(images) == 4 else (2, 3)
        strips: list[np.ndarray] = []
        for grid_row in range(grid_rows):
            tiles: list[np.ndarray] = []
            y0 = round(grid_row * height / grid_rows)
            y1 = round((grid_row + 1) * height / grid_rows)
            for column in range(columns):
                x0 = round(column * width / columns)
                x1 = round((column + 1) * width / columns)
                tiles.append(
                    cv2.resize(
                        images[grid_row * columns + column],
                        (x1 - x0, y1 - y0),
                        interpolation=cv2.INTER_AREA,
                    )
                )
            strips.append(np.concatenate(tiles, axis=1))
        bgr = np.concatenate(strips, axis=0)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def state_history(
    rows: list[dict[str, Any]], index: int, offsets: tuple[int, ...]
) -> np.ndarray:
    values: list[float] = []
    for offset in offsets:
        values.extend(rows[max(0, index - offset)]["joint_pos"][:JOINT_DIM])
    return np.asarray(values, dtype=np.float32)


def selected_indices(
    rows: list[dict[str, Any]], samples: int, history: int, future: int
) -> list[int]:
    first = min(max(1, history), max(1, len(rows) - future - 1))
    last = len(rows) - future - 1
    if last < first:
        return []
    count = min(samples, last - first + 1)
    return np.linspace(first, last, count, dtype=np.int64).astype(int).tolist()


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


def predict(
    spec: ModelSpec,
    episodes: list[list[dict[str, Any]]],
    requests: list[tuple[int, int]],
    batch_size: int,
    device: str,
) -> dict[tuple[int, int], np.ndarray]:
    import torch

    policy, preprocessor, postprocessor = load_policy(spec.checkpoint, device)
    output: dict[tuple[int, int], np.ndarray] = {}
    for start in range(0, len(requests), batch_size):
        keys = requests[start : start + batch_size]
        images: list[np.ndarray] = []
        states: list[np.ndarray] = []
        for episode_index, frame_index in keys:
            rows = episodes[episode_index]
            images.append(history_mosaic(rows, frame_index, spec.image_offsets))
            states.append(state_history(rows, frame_index, spec.state_offsets))
        batch = {
            "observation.images.scene": torch.from_numpy(
                np.stack(images).transpose(0, 3, 1, 2)
            ).float()
            / 255.0,
            "observation.state": torch.from_numpy(np.stack(states)).float(),
        }
        processed = preprocessor(batch)
        model_inputs = {
            key: processed[key] for key in policy.config.input_features.keys()
        }
        with torch.inference_mode():
            raw = policy.predict_action_chunk(model_inputs)
            commands = postprocessor(raw).detach().cpu().numpy()
        for key, command in zip(keys, commands):
            output[key] = command[:, :JOINT_DIM].astype(np.float32)
        done = min(start + batch_size, len(requests))
        print(f"[{spec.label}] {done}/{len(requests)} predictions", flush=True)
    return output


def evaluate_model(
    spec: ModelSpec,
    episodes: list[list[dict[str, Any]]],
    samples_per_episode: int,
    execution_horizon: int,
    direction_deadband: float,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    horizon = min(execution_horizon, spec.chunk_size)
    anchors: list[tuple[int, int]] = []
    for episode_index, rows in enumerate(episodes):
        indices = selected_indices(
            rows,
            samples_per_episode,
            max(spec.image_offsets + spec.state_offsets),
            spec.chunk_size + horizon,
        )
        anchors.extend((episode_index, frame_index) for frame_index in indices)
    if not anchors:
        raise RuntimeError(f"No samples selected for {spec.label}")
    requests = sorted(set(anchors + [(ep, frame + horizon) for ep, frame in anchors]))
    predictions = predict(spec, episodes, requests, batch_size, device)

    chunk_errors: list[np.ndarray] = []
    endpoint_errors: list[np.ndarray] = []
    boundary_jumps: list[np.ndarray] = []
    direction_wrong = 0
    direction_total = 0
    out_of_range = 0
    prediction_total = 0
    per_sample: list[dict[str, Any]] = []
    for episode_index, frame_index in anchors:
        rows = episodes[episode_index]
        raw = predictions[(episode_index, frame_index)][: spec.chunk_size]
        out_of_range += int(((raw[:, ACTIVE_MASK] < 0) | (raw[:, ACTIVE_MASK] > 255)).sum())
        prediction_total += int(raw[:, ACTIVE_MASK].size)
        predicted = np.clip(raw, 0.0, 255.0)
        target = np.asarray(
            [
                rows[min(frame_index + step, len(rows) - 1)]["last_action"][:JOINT_DIM]
                for step in range(spec.chunk_size)
            ],
            dtype=np.float32,
        )
        error = np.abs(predicted[:, ACTIVE_MASK] - target[:, ACTIVE_MASK])
        chunk_errors.append(error.reshape(-1))
        endpoint_errors.append(error[-1])

        predicted_velocity = np.diff(predicted[:, ACTIVE_MASK], axis=0)
        target_velocity = np.diff(target[:, ACTIVE_MASK], axis=0)
        moving = np.abs(target_velocity) >= direction_deadband
        wrong = moving & (predicted_velocity * target_velocity <= 0)
        direction_wrong += int(wrong.sum())
        direction_total += int(moving.sum())

        next_prediction = np.clip(
            predictions[(episode_index, frame_index + horizon)][0], 0.0, 255.0
        )
        previous_last = predicted[horizon - 1]
        boundary = np.abs(
            next_prediction[ACTIVE_MASK] - previous_last[ACTIVE_MASK]
        )
        boundary_jumps.append(boundary)
        per_sample.append(
            {
                "episode": episode_index,
                "frame": frame_index,
                "chunk_mae_ticks": float(error.mean()),
                "chunk_end_mae_ticks": float(error[-1].mean()),
                "boundary_jump_ticks": float(boundary.mean()),
            }
        )

    return {
        "label": spec.label,
        "artifact": str(spec.artifact),
        "checkpoint": str(spec.checkpoint),
        "image_history_offsets": list(spec.image_offsets),
        "state_history_offsets": list(spec.state_offsets),
        "chunk_size": spec.chunk_size,
        "execution_horizon": horizon,
        "sample_count": len(anchors),
        "chunk_active_joint_mae_ticks": float(np.concatenate(chunk_errors).mean()),
        "direction_disagreement_percent": (
            100.0 * direction_wrong / direction_total if direction_total else None
        ),
        "direction_comparison_count": direction_total,
        "chunk_boundary_jump_ticks": float(np.stack(boundary_jumps).mean()),
        "chunk_end_active_joint_mae_ticks": float(
            np.concatenate(endpoint_errors).mean()
        ),
        "active_outputs_out_of_range_percent": (
            100.0 * out_of_range / prediction_total if prediction_total else 0.0
        ),
        "samples": per_sample,
    }


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# G20 ACT offline ablation diagnostics",
        "",
        "These values use the same recorded held-out trajectories and do not move "
        "the real hand. Lower is better for every numeric metric.",
        "",
        "| Variant | Chunk MAE (ticks) | Direction disagreement | Boundary jump (ticks) | Chunk-end MAE (ticks) |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in result["models"]:
        lines.append(
            f"| {item['label']} | {item['chunk_active_joint_mae_ticks']:.2f} | "
            f"{item['direction_disagreement_percent']:.1f}% | "
            f"{item['chunk_boundary_jump_ticks']:.2f} | "
            f"{item['chunk_end_active_joint_mae_ticks']:.2f} |"
        )
    for missing in result.get("missing_rows", []):
        lines.append(f"| {missing['label']} | N/A | N/A | N/A | N/A |")
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- The original real-robot columns (success, stay in hand, and success "
            "time) cannot be measured from offline recordings.",
            "- Reserved command channels q11-q14 are excluded.",
            "- Predictions are clipped to the deployed 0-255 command range before "
            "the four metrics are calculated.",
            "- This is a descriptive comparison, not a perfectly controlled "
            "ablation: the available checkpoints were trained on different data.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    ensure_lerobot_python()
    import torch

    specs = [parse_model(value) for value in args.model]
    episodes = [read_episode(path) for path in args.episode]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[offline] device={device} episodes={len(episodes)} "
        f"models={len(specs)}",
        flush=True,
    )
    models = [
        evaluate_model(
            spec,
            episodes,
            args.samples_per_episode,
            args.execution_horizon,
            args.direction_deadband,
            args.batch_size,
            device,
        )
        for spec in specs
    ]
    result = {
        "schema": "linkerhand_g20_act_common_offline_ablation_v1",
        "episodes": [str(path) for path in args.episode],
        "samples_per_episode": args.samples_per_episode,
        "direction_deadband_ticks": args.direction_deadband,
        "models": models,
        "missing_rows": [
            {
                "label": "+ MP + action library",
                "reason": "No separate trained checkpoint exists for this row.",
            },
            {
                "label": "+ posttraining",
                "reason": "The posttraining dataset exists, but no trained checkpoint exists.",
            },
        ],
        "warning": (
            "Offline imitation diagnostics are not real-robot success, retention, "
            "or completion-time measurements."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "offline_ablation.json"
    md_path = args.output_dir / "offline_ablation.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(result), encoding="utf-8")
    print(markdown_report(result))
    print(f"[result] JSON: {json_path}")
    print(f"[result] Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
