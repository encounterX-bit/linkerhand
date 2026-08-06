#!/usr/bin/env python3
"""Convert G20 camera demonstrations and train a visual ACT policy.

The source format is written by ``linkerhand_g20_touch_recorder.py``::

    data/<session>/episode_NNN/samples.jsonl
    data/<session>/episode_NNN/images/000000.jpg

Only episodes with real image files, 20-D ``joint_pos``, and 20-D
``last_action`` are included. Camera-free datasets such as ``data/grasp_cube``
are therefore skipped automatically.

The produced LeRobot policy observes the scene camera plus the current 20-D SDK-range
joint state and predicts a chunk of future 20-D absolute SDK-range commands.
Optional visual/state history resolves trajectories that revisit the same
single-frame pose.  ``--success-reference`` keeps only demonstrations whose
final ArUco face and oriented marker layout match a known successful result.
This script trains/checkpoints only; it never imports ROS and never publishes a
hardware command. Human-rated self-imitation episodes are filtered by
``episode.json`` quality score, and ``--finetune-from`` starts a new low-rate
training run from an existing ACT checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "g20_visual_act"
DEFAULT_TRAIN_PYTHON = Path(
    "/home/zhaoyan-qian/Desktop/Jacky/ros2_pairlab3-main/"
    ".venv_ros2_pairlab3/bin/python"
)
JOINT_DIM = 20
FLEXION_JOINTS = (1, 2, 3, 4, 16, 17, 18, 19)
TACTILE_DIMS = {"none": 0, "mass-contact": 12}


@dataclass(frozen=True)
class EpisodeSource:
    path: Path
    rows: tuple[dict[str, Any], ...]
    task: str
    quality_score: float | None = None
    action_reversals: int = 0
    thumb_route_reversals: int = 0
    success_marker_layout_error: float | None = None
    original_frames: int | None = None
    observed_fps: float | None = None


@dataclass(frozen=True)
class MarkerSuccessReference:
    path: Path
    dictionary_name: str
    marker_centers: dict[int, np.ndarray]

    @property
    def marker_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.marker_centers))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("all", "convert", "train"), default="all")
    ap.add_argument(
        "--data-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "source session or parent directory; repeat this option to combine "
            "multiple selected data roots"
        ),
    )
    ap.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    ap.add_argument(
        "--reuse-dataset-from",
        type=Path,
        default=None,
        help=(
            "artifact root whose existing dataset/ and manifest should be reused; "
            "valid only with --stage train"
        ),
    )
    ap.add_argument("--repo-id", default="linkerhand_g20_orientation_grasp_visual")
    ap.add_argument("--task", default="grasp the tagged cube from its observed orientation")
    ap.add_argument(
        "--include-task-id",
        action="append",
        default=[],
        help=(
            "only include source sessions whose session.json task_id matches; "
            "repeat the option to allow multiple task IDs"
        ),
    )
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument(
        "--resample-rated-to-fps",
        action="store_true",
        help=(
            "timestamp-resample human-rated episodes to --fps before building "
            "history and future chunks; useful when online inference/recording "
            "runs slower than the expert recorder"
        ),
    )
    ap.add_argument("--image-height", type=int, default=240)
    ap.add_argument("--image-width", type=int, default=320)
    ap.add_argument("--image-key", default="scene")
    ap.add_argument(
        "--history-frame-offsets",
        default="0",
        help=(
            "comma-separated frame offsets used to build the visual input; "
            "use 15,10,5,0 for a 2x2 half-second mosaic or "
            "90,72,54,36,18,0 for a 2x3 three-second mosaic"
        ),
    )
    ap.add_argument(
        "--state-history-offsets",
        default="0",
        help=(
            "comma-separated frame offsets concatenated into observation.state; "
            "for example 90,60,30,0 gives ACT exact recent joint direction "
            "instead of only the current 20-D state"
        ),
    )
    ap.add_argument(
        "--tactile-mode",
        choices=tuple(TACTILE_DIMS),
        default="none",
        help=(
            "append current fingertip/palm feedback to observation.state; "
            "mass-contact adds mass_values[6] and contact_6[6]. This is the "
            "coarse tactile level used by the local Being-H0.8-style baseline"
        ),
    )
    ap.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav",
                    help="LeRobot video decoder; pyav avoids local torchcodec/PyTorch ABI issues")
    ap.add_argument("--min-frames", type=int, default=30)
    ap.add_argument(
        "--success-reference",
        type=Path,
        default=None,
        help=(
            "reference success image; when set, an episode is kept only if its "
            "final frame contains the same ArUco IDs in a similar layout"
        ),
    )
    ap.add_argument(
        "--success-aruco-dictionary",
        default="DICT_4X4_50",
        help="OpenCV aruco dictionary constant used by the success reference",
    )
    ap.add_argument(
        "--success-max-layout-error",
        type=float,
        default=0.15,
        help=(
            "maximum translation/scale-normalized marker-center RMS error for "
            "a final frame to count as successful"
        ),
    )
    ap.add_argument(
        "--max-action-reversals",
        type=int,
        default=None,
        help=(
            "keep only episodes with at most this many smoothed close/open "
            "direction changes; unset keeps every episode"
        ),
    )
    ap.add_argument(
        "--min-thumb-route-reversals",
        type=int,
        default=None,
        help=(
            "keep only episodes whose smoothed q5+q10 route has at least this "
            "many direction changes; combine with the maximum to select one route"
        ),
    )
    ap.add_argument(
        "--max-thumb-route-reversals",
        type=int,
        default=None,
        help=(
            "keep only episodes whose smoothed q5+q10 route has at most this "
            "many direction changes; useful for removing inconsistent thumb paths"
        ),
    )
    ap.add_argument("--max-episodes", type=int, default=0,
                    help="0 keeps every valid episode; useful for smoke tests")
    ap.add_argument("--val-episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=30)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--save-freq", type=int, default=1_000)
    ap.add_argument("--log-freq", type=int, default=20)
    ap.add_argument(
        "--min-rated-score",
        type=float,
        default=1.0,
        help="minimum human score for rated self-imitation episodes; unscored expert data is kept",
    )
    ap.add_argument(
        "--finetune-from",
        type=Path,
        default=None,
        help="pretrained_model checkpoint directory used to initialize a new fine-tuning run",
    )
    ap.add_argument("--finetune-learning-rate", type=float, default=3e-6)
    ap.add_argument(
        "--freeze-profile",
        choices=("none", "edge-head", "decoder-head"),
        default="none",
        help=(
            "parameter-freezing profile for fine-tuning; edge-head is the most "
            "conservative and decoder-head allows the full decoder to adapt"
        ),
    )
    ap.add_argument(
        "--distill-teacher",
        type=Path,
        default=None,
        help=(
            "known-good ACT pretrained_model used to constrain deterministic "
            "full-chunk outputs during fine-tuning"
        ),
    )
    ap.add_argument("--distill-base-weight", type=float, default=2.0)
    ap.add_argument("--distill-edge-weight", type=float, default=0.1)
    ap.add_argument(
        "--distill-edge-source-prefix",
        action="append",
        default=[],
        help=(
            "manifest source prefix treated as edge data; repeat for multiple "
            "sessions"
        ),
    )
    ap.add_argument(
        "--momentum-weight",
        type=float,
        default=0.0,
        help=(
            "weight for an ACT adjacent-action direction loss; 0 disables it. "
            "A useful conservative starting value is 0.2"
        ),
    )
    ap.add_argument(
        "--momentum-deadband",
        type=float,
        default=0.01,
        help=(
            "ignore demonstrated per-frame velocities below this value in "
            "LeRobot-normalized action units"
        ),
    )
    ap.add_argument(
        "--momentum-margin",
        type=float,
        default=0.005,
        help=(
            "minimum correctly signed predicted velocity requested by the "
            "momentum hinge, in normalized action units"
        ),
    )
    ap.add_argument(
        "--exclude-source-episode",
        action="append",
        default=[],
        help="exact manifest source path excluded from training; repeat as needed",
    )
    ap.add_argument(
        "--train-episode-indices",
        default=None,
        help=(
            "optional comma-separated dataset episode indices used for training; "
            "indices must already belong to the manifest training split"
        ),
    )
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--overwrite-dataset", action="store_true")
    ap.add_argument("--overwrite-output", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-auto-python", action="store_true",
                    help="fail instead of re-executing with the known LeRobot environment")
    args = ap.parse_args()
    args.data_roots = [
        path.expanduser().resolve()
        for path in (args.data_root or [DEFAULT_DATA_ROOT])
    ]
    missing_data_roots = [path for path in args.data_roots if not path.is_dir()]
    if missing_data_roots:
        ap.error(
            "--data-root directory does not exist: "
            + ", ".join(str(path) for path in missing_data_roots)
        )
    try:
        args.history_frame_offsets = tuple(
            sorted(
                {int(value.strip()) for value in args.history_frame_offsets.split(",")},
                reverse=True,
            )
        )
    except ValueError:
        ap.error("--history-frame-offsets must be comma-separated integers")
    try:
        args.state_history_offsets = tuple(
            sorted(
                {int(value.strip()) for value in args.state_history_offsets.split(",")},
                reverse=True,
            )
        )
    except ValueError:
        ap.error("--state-history-offsets must be comma-separated integers")
    if args.train_episode_indices is not None:
        try:
            args.train_episode_indices = tuple(
                sorted(
                    {
                        int(value.strip())
                        for value in args.train_episode_indices.split(",")
                        if value.strip()
                    }
                )
            )
        except ValueError:
            ap.error("--train-episode-indices must be comma-separated integers")
        if (
            not args.train_episode_indices
            or args.train_episode_indices[0] < 0
        ):
            ap.error(
                "--train-episode-indices must contain at least one "
                "non-negative integer"
            )
    if (
        not args.history_frame_offsets
        or args.history_frame_offsets[-1] != 0
        or any(value < 0 for value in args.history_frame_offsets)
        or len(args.history_frame_offsets) not in (1, 4, 6)
    ):
        ap.error(
            "--history-frame-offsets must contain 0 and have 1, 4, or 6 "
            "non-negative offsets"
        )
    if (
        not args.state_history_offsets
        or args.state_history_offsets[-1] != 0
        or any(value < 0 for value in args.state_history_offsets)
        or len(args.state_history_offsets) > 8
    ):
        ap.error(
            "--state-history-offsets must contain 0 and at most 8 "
            "non-negative offsets"
        )
    args.artifact_root = args.artifact_root.expanduser().resolve()
    if args.success_reference is not None:
        args.success_reference = args.success_reference.expanduser().resolve()
        if not args.success_reference.is_file():
            ap.error(
                f"--success-reference image does not exist: "
                f"{args.success_reference}"
            )
    if args.success_max_layout_error < 0.0:
        ap.error("--success-max-layout-error must be nonnegative")
    if args.reuse_dataset_from is not None:
        args.reuse_dataset_from = args.reuse_dataset_from.expanduser().resolve()
        if args.stage != "train":
            ap.error("--reuse-dataset-from requires --stage train")
        reused_manifest = (
            args.reuse_dataset_from / "dataset" / "g20_source_manifest.json"
        )
        if not reused_manifest.is_file():
            ap.error(
                "--reuse-dataset-from does not contain a completed dataset manifest: "
                f"{reused_manifest}"
            )
    if args.finetune_from is not None:
        args.finetune_from = args.finetune_from.expanduser().resolve()
    if args.distill_teacher is not None:
        args.distill_teacher = args.distill_teacher.expanduser().resolve()
    if args.fps <= 0 or args.min_frames <= 0:
        ap.error("--fps and --min-frames must be positive")
    if args.max_action_reversals is not None and args.max_action_reversals < 0:
        ap.error("--max-action-reversals must be non-negative")
    if (
        args.min_thumb_route_reversals is not None
        and args.min_thumb_route_reversals < 0
    ):
        ap.error("--min-thumb-route-reversals must be non-negative")
    if (
        args.max_thumb_route_reversals is not None
        and args.max_thumb_route_reversals < 0
    ):
        ap.error("--max-thumb-route-reversals must be non-negative")
    if (
        args.min_thumb_route_reversals is not None
        and args.max_thumb_route_reversals is not None
        and args.min_thumb_route_reversals > args.max_thumb_route_reversals
    ):
        ap.error("--min-thumb-route-reversals cannot exceed the maximum")
    if args.chunk_size <= 0:
        ap.error("--chunk-size must be positive")
    if not 1 <= args.n_action_steps <= args.chunk_size:
        ap.error("--n-action-steps must be in [1, chunk-size]")
    if not 0.0 <= args.min_rated_score <= 1.0:
        ap.error("--min-rated-score must be in [0, 1]")
    if args.finetune_learning_rate <= 0:
        ap.error("--finetune-learning-rate must be positive")
    if args.finetune_from is not None and not (args.finetune_from / "model.safetensors").is_file():
        ap.error(f"--finetune-from is not a pretrained_model directory: {args.finetune_from}")
    if args.freeze_profile != "none" and args.finetune_from is None:
        ap.error("--freeze-profile requires --finetune-from")
    if args.distill_teacher is not None:
        if not (args.distill_teacher / "model.safetensors").is_file():
            ap.error(
                f"--distill-teacher is not a pretrained_model directory: "
                f"{args.distill_teacher}"
            )
        if args.finetune_from is None:
            ap.error("--distill-teacher requires --finetune-from")
        if args.freeze_profile == "none":
            ap.error("--distill-teacher requires a non-none --freeze-profile")
        if not args.distill_edge_source_prefix:
            ap.error(
                "--distill-teacher requires at least one "
                "--distill-edge-source-prefix"
            )
    if args.distill_base_weight < 0 or args.distill_edge_weight < 0:
        ap.error("distillation weights must be non-negative")
    if args.momentum_weight < 0:
        ap.error("--momentum-weight must be non-negative")
    if args.momentum_deadband < 0 or args.momentum_margin < 0:
        ap.error("momentum deadband and margin must be non-negative")
    if args.momentum_weight > 0 and args.distill_teacher is not None:
        ap.error("momentum and distillation wrappers cannot be combined")
    if args.momentum_weight > 0 and args.freeze_profile != "none":
        ap.error("momentum training currently requires --freeze-profile none")
    return args


def ensure_training_python(args: argparse.Namespace) -> None:
    """Re-exec with the installed LeRobot environment when necessary."""
    try:
        import torch  # noqa: F401
        import lerobot  # noqa: F401
        return
    except ImportError as exc:
        if args.no_auto_python or not DEFAULT_TRAIN_PYTHON.is_file():
            raise RuntimeError(
                "PyTorch + LeRobot are required. Run with "
                f"{DEFAULT_TRAIN_PYTHON}"
            ) from exc
        target = DEFAULT_TRAIN_PYTHON.absolute()
        current = Path(sys.executable).absolute()
        if target == current:
            raise RuntimeError("The configured training Python cannot import torch/lerobot") from exc
        print(f"[env] re-executing with {DEFAULT_TRAIN_PYTHON}", flush=True)
        os.execv(str(DEFAULT_TRAIN_PYTHON), [str(DEFAULT_TRAIN_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def find_session(episode_dir: Path) -> dict[str, Any]:
    for parent in (episode_dir, episode_dir.parent):
        path = parent / "session.json"
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
    return {}


def find_quality_score(episode_dir: Path) -> tuple[bool, float | None]:
    path = episode_dir / "episode.json"
    if not path.is_file():
        return False, None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, AttributeError):
        return False, None
    if not isinstance(metadata, dict) or "quality_score" not in metadata:
        return False, None
    value = metadata.get("quality_score")
    if value is None:
        return True, None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return True, None
    return True, score if 0.0 <= score <= 1.0 else None


def _row_times(rows: tuple[dict[str, Any], ...]) -> np.ndarray | None:
    """Return a strictly increasing episode-relative time axis when available."""
    for key in ("elapsed", "timestamp"):
        try:
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            continue
        if (
            len(values) >= 2
            and np.all(np.isfinite(values))
            and np.all(np.diff(values) > 1e-6)
        ):
            return values - values[0]
    return None


def resample_rows_to_fps(
    rows: tuple[dict[str, Any], ...],
    target_fps: float,
) -> tuple[tuple[dict[str, Any], ...], float | None]:
    """Linearly resample state/action while using the nearest recorded image."""
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    times = _row_times(rows)
    if times is None:
        return rows, None
    duration = float(times[-1])
    if duration <= 0:
        return rows, None
    observed_fps = float((len(rows) - 1) / duration)
    if abs(observed_fps - target_fps) / target_fps <= 0.02:
        return rows, observed_fps

    target_times = np.arange(
        int(np.floor(duration * target_fps + 1e-6)) + 1,
        dtype=np.float64,
    ) / target_fps
    output: list[dict[str, Any]] = []
    for output_index, target_time in enumerate(target_times):
        upper = min(int(np.searchsorted(times, target_time, side="left")), len(rows) - 1)
        lower = max(0, upper - 1)
        if upper == lower:
            alpha = 0.0
        else:
            span = float(times[upper] - times[lower])
            alpha = float(np.clip((target_time - times[lower]) / span, 0.0, 1.0))
        nearest = lower if alpha <= 0.5 else upper
        row = dict(rows[nearest])
        for key in ("joint_pos", "last_action"):
            left = np.asarray(rows[lower][key], dtype=np.float32)
            right = np.asarray(rows[upper][key], dtype=np.float32)
            row[key] = ((1.0 - alpha) * left + alpha * right).tolist()
        row["index"] = output_index
        if "elapsed" in rows[0]:
            row["elapsed"] = float(rows[0]["elapsed"]) + float(target_time)
        if "timestamp" in rows[0]:
            row["timestamp"] = float(rows[0]["timestamp"]) + float(target_time)
        output.append(row)
    return tuple(output), observed_fps


def tactile_vector(
    row: dict[str, Any],
    mode: str,
) -> np.ndarray | None:
    """Return the current coarse tactile observation, or None if unavailable."""
    if mode == "none":
        return np.zeros(0, dtype=np.float32)
    if mode != "mass-contact":
        raise ValueError(f"unsupported tactile mode: {mode}")
    mass = row.get("mass_values")
    contact = row.get("contact_6")
    if (
        not isinstance(mass, list)
        or len(mass) < 6
        or not isinstance(contact, list)
        or len(contact) < 6
    ):
        return None
    values = np.asarray(mass[:6] + contact[:6], dtype=np.float32)
    if values.shape != (12,) or not np.all(np.isfinite(values)):
        return None
    values[:6] = np.maximum(values[:6], 0.0)
    values[6:] = np.clip(values[6:], 0.0, 1.0)
    return values


def valid_row(
    episode_dir: Path,
    row: dict[str, Any],
    tactile_mode: str = "none",
) -> bool:
    image_path = row.get("image_path")
    joint_pos = row.get("joint_pos")
    action = row.get("last_action")
    return bool(
        image_path
        and (episode_dir / str(image_path)).is_file()
        and isinstance(joint_pos, list)
        and len(joint_pos) >= JOINT_DIM
        and isinstance(action, list)
        and len(action) >= JOINT_DIM
        and tactile_vector(row, tactile_mode) is not None
    )


def detect_aruco_marker_centers(
    image_path: Path,
    dictionary_name: str,
) -> dict[int, np.ndarray]:
    """Detect marker centers without depending on marker return order."""
    import cv2

    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise RuntimeError(
            f"Unknown OpenCV ArUco dictionary {dictionary_name!r}"
        )
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot decode success image: {image_path}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    corners, ids, _rejected = cv2.aruco.detectMarkers(image, dictionary)
    if ids is None:
        return {}
    return {
        int(marker_id): np.asarray(marker_corners, dtype=np.float32)
        .reshape(4, 2)
        .mean(axis=0)
        for marker_corners, marker_id in zip(corners, ids.reshape(-1))
    }


def normalized_marker_layout(
    centers: dict[int, np.ndarray],
    marker_ids: tuple[int, ...],
) -> np.ndarray:
    """Normalize marker centers for translation and overall image scale."""
    if len(marker_ids) < 2 or any(marker_id not in centers for marker_id in marker_ids):
        raise ValueError("at least two requested marker IDs must be present")
    points = np.stack([centers[marker_id] for marker_id in marker_ids]).astype(
        np.float32
    )
    points -= points.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(np.square(points), axis=1))))
    if scale <= 1e-6:
        raise ValueError("marker layout is degenerate")
    return points / scale


def marker_layout_error(
    reference: MarkerSuccessReference,
    candidate_centers: dict[int, np.ndarray],
) -> tuple[float | None, tuple[int, ...]]:
    """Compare a candidate with the reference IDs and their oriented layout."""
    missing = tuple(
        marker_id
        for marker_id in reference.marker_ids
        if marker_id not in candidate_centers
    )
    if missing:
        return None, missing
    reference_layout = normalized_marker_layout(
        reference.marker_centers, reference.marker_ids
    )
    candidate_layout = normalized_marker_layout(
        candidate_centers, reference.marker_ids
    )
    error = float(
        np.sqrt(
            np.mean(
                np.sum(np.square(candidate_layout - reference_layout), axis=1)
            )
        )
    )
    return error, ()


def load_success_reference(args: argparse.Namespace) -> MarkerSuccessReference | None:
    if hasattr(args, "_loaded_success_reference"):
        return args._loaded_success_reference
    if args.success_reference is None:
        args._loaded_success_reference = None
        return None
    centers = detect_aruco_marker_centers(
        args.success_reference, args.success_aruco_dictionary
    )
    if len(centers) < 2:
        raise RuntimeError(
            "Success reference must contain at least two detectable ArUco markers; "
            f"found IDs {sorted(centers)} in {args.success_reference}"
        )
    reference = MarkerSuccessReference(
        args.success_reference,
        args.success_aruco_dictionary,
        centers,
    )
    print(
        f"[success-reference] ids={list(reference.marker_ids)} "
        f"dictionary={reference.dictionary_name} image={reference.path}",
        flush=True,
    )
    args._loaded_success_reference = reference
    return reference


def count_smoothed_reversals(values: np.ndarray) -> int:
    """Count robust direction changes in one scalar trajectory."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) < 2:
        return 0
    window = min(15, len(values))
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")
    delta = np.diff(smoothed)
    direction = np.where(delta > 0.35, 1, np.where(delta < -0.35, -1, 0))
    runs: list[int] = []
    for value in direction[direction != 0]:
        value = int(value)
        if not runs or value != runs[-1]:
            runs.append(value)
    return max(0, len(runs) - 1)


def count_action_reversals(rows: tuple[dict[str, Any], ...]) -> int:
    """Count robust close/open changes across the primary flexion joints."""
    if len(rows) < 2:
        return 0
    actions = np.asarray(
        [[float(row["last_action"][i]) for i in FLEXION_JOINTS] for row in rows],
        dtype=np.float32,
    )
    closure = 255.0 - actions.mean(axis=1)
    return count_smoothed_reversals(closure)


def count_thumb_route_reversals(rows: tuple[dict[str, Any], ...]) -> int:
    """Count q5 side-swing plus q10 roll direction changes.

    These are the two channels that define the repeated thumb-only cube-push
    route. Summing their reversal counts separates the dominant demonstrations
    from retries that add an extra backtrack, while state history still tells
    the policy which side of a necessary return stroke it is currently on.
    """
    if len(rows) < 2:
        return 0
    actions = np.asarray(
        [[float(row["last_action"][i]) for i in (5, 10)] for row in rows],
        dtype=np.float32,
    )
    return sum(count_smoothed_reversals(actions[:, column]) for column in range(2))


def history_mosaic(
    episode: EpisodeSource,
    frame_index: int,
    offsets: tuple[int, ...],
    width: int,
    height: int,
) -> np.ndarray | None:
    """Build a current-frame image or a chronological history mosaic."""
    import cv2

    images: list[np.ndarray] = []
    for offset in offsets:
        source_index = max(0, frame_index - offset)
        row = episode.rows[source_index]
        bgr = cv2.imread(str(episode.path / str(row["image_path"])))
        if bgr is None:
            return None
        images.append(bgr)
    if len(images) == 1:
        if images[0].shape[:2] != (height, width):
            images[0] = cv2.resize(
                images[0], (width, height), interpolation=cv2.INTER_AREA
            )
        return images[0]

    rows, columns = (2, 2) if len(images) == 4 else (2, 3)
    mosaic_rows: list[np.ndarray] = []
    for row_index in range(rows):
        tiles: list[np.ndarray] = []
        y0 = round(row_index * height / rows)
        y1 = round((row_index + 1) * height / rows)
        for column_index in range(columns):
            x0 = round(column_index * width / columns)
            x1 = round((column_index + 1) * width / columns)
            image = images[row_index * columns + column_index]
            tiles.append(
                cv2.resize(
                    image,
                    (x1 - x0, y1 - y0),
                    interpolation=cv2.INTER_AREA,
                )
            )
        mosaic_rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(mosaic_rows, axis=0)


def discover_episodes(args: argparse.Namespace) -> tuple[list[EpisodeSource], list[dict[str, Any]]]:
    kept: list[EpisodeSource] = []
    skipped: list[dict[str, Any]] = []
    seen_samples: set[Path] = set()
    success_reference = load_success_reference(args)
    for data_root in args.data_roots:
        for samples_path in sorted(data_root.rglob("samples.jsonl")):
            samples_path = samples_path.resolve()
            if samples_path in seen_samples:
                continue
            seen_samples.add(samples_path)
            episode_dir = samples_path.parent
            all_rows = tuple(read_jsonl(samples_path))
            rows = tuple(
                row
                for row in all_rows
                if valid_row(episode_dir, row, args.tactile_mode)
            )
            rel = f"{data_root.name}/{episode_dir.relative_to(data_root)}"
            session = find_session(episode_dir)
            source_task_id = str(
                session.get("task_id")
                or (rows[0].get("task_id") if rows else "")
                or ""
            )
            if args.include_task_id and source_task_id not in args.include_task_id:
                skipped.append({
                    "source": rel,
                    "reason": "task_id_filter",
                    "task_id": source_task_id or None,
                    "allowed_task_ids": list(args.include_task_id),
                    "total_frames": len(all_rows),
                    "usable_frames": len(rows),
                })
                continue
            is_rated_episode, quality_score = find_quality_score(episode_dir)
            if is_rated_episode and (
                quality_score is None or quality_score < args.min_rated_score
            ):
                skipped.append({
                    "source": rel,
                    "reason": "unrated_or_low_quality_self_imitation",
                    "quality_score": quality_score,
                    "minimum_score": args.min_rated_score,
                    "total_frames": len(all_rows),
                    "usable_frames": len(rows),
                })
                continue
            original_frames = len(rows)
            observed_fps: float | None = None
            if is_rated_episode and args.resample_rated_to_fps:
                rows, observed_fps = resample_rows_to_fps(rows, args.fps)
            action_reversals = count_action_reversals(rows)
            thumb_route_reversals = count_thumb_route_reversals(rows)
            if len(rows) < args.min_frames:
                skipped.append({
                    "source": rel,
                    "reason": "missing_camera_or_required_fields",
                    "total_frames": len(all_rows),
                    "usable_frames": len(rows),
                })
                continue
            success_marker_layout_error: float | None = None
            if success_reference is not None:
                final_image = episode_dir / str(rows[-1]["image_path"])
                candidate_centers = detect_aruco_marker_centers(
                    final_image, success_reference.dictionary_name
                )
                success_marker_layout_error, missing_ids = marker_layout_error(
                    success_reference, candidate_centers
                )
                if (
                    success_marker_layout_error is None
                    or success_marker_layout_error > args.success_max_layout_error
                ):
                    skipped.append({
                        "source": rel,
                        "reason": (
                            "final_frame_missing_success_markers"
                            if missing_ids
                            else "final_frame_marker_layout_mismatch"
                        ),
                        "reference_marker_ids": list(success_reference.marker_ids),
                        "observed_marker_ids": sorted(candidate_centers),
                        "missing_marker_ids": list(missing_ids),
                        "marker_layout_error": success_marker_layout_error,
                        "maximum_marker_layout_error": args.success_max_layout_error,
                        "final_image": str(final_image),
                        "total_frames": len(all_rows),
                        "usable_frames": len(rows),
                    })
                    continue
            if (
                args.max_action_reversals is not None
                and action_reversals > args.max_action_reversals
            ):
                skipped.append({
                    "source": rel,
                    "reason": "too_many_action_reversals",
                    "action_reversals": action_reversals,
                    "maximum_action_reversals": args.max_action_reversals,
                    "total_frames": len(all_rows),
                    "usable_frames": len(rows),
                })
                continue
            if (
                args.min_thumb_route_reversals is not None
                and thumb_route_reversals < args.min_thumb_route_reversals
            ):
                skipped.append({
                    "source": rel,
                    "reason": "too_few_thumb_route_reversals",
                    "thumb_route_reversals": thumb_route_reversals,
                    "minimum_thumb_route_reversals": args.min_thumb_route_reversals,
                    "total_frames": len(all_rows),
                    "usable_frames": len(rows),
                })
                continue
            if (
                args.max_thumb_route_reversals is not None
                and thumb_route_reversals > args.max_thumb_route_reversals
            ):
                skipped.append({
                    "source": rel,
                    "reason": "too_many_thumb_route_reversals",
                    "thumb_route_reversals": thumb_route_reversals,
                    "maximum_thumb_route_reversals": args.max_thumb_route_reversals,
                    "total_frames": len(all_rows),
                    "usable_frames": len(rows),
                })
                continue
            task = args.task or str(session.get("task_id", "g20_orientation_grasp"))
            kept.append(
                EpisodeSource(
                    episode_dir,
                    rows,
                    task,
                    quality_score,
                    action_reversals,
                    thumb_route_reversals,
                    success_marker_layout_error,
                    original_frames,
                    observed_fps,
                )
            )
    if args.max_episodes > 0:
        for ep in kept[args.max_episodes:]:
            skipped.append({
                "source": source_name(ep.path, args.data_roots),
                "reason": "max_episodes",
                "total_frames": len(ep.rows),
                "usable_frames": len(ep.rows),
            })
        kept = kept[: args.max_episodes]
    return kept, skipped


def source_name(path: Path, data_roots: list[Path]) -> str:
    for data_root in data_roots:
        try:
            return f"{data_root.name}/{path.relative_to(data_root)}"
        except ValueError:
            continue
    return str(path)


def dataset_root(args: argparse.Namespace) -> Path:
    artifact_root = args.reuse_dataset_from or args.artifact_root
    return artifact_root / "dataset"


def output_root(args: argparse.Namespace) -> Path:
    return args.artifact_root / "training"


def manifest_path(args: argparse.Namespace) -> Path:
    return dataset_root(args) / "g20_source_manifest.json"


def persist_reused_manifest_for_inference(args: argparse.Namespace) -> None:
    """Keep temporal-image metadata discoverable beside a new checkpoint.

    Training may read a dataset from another artifact root, while the hardware
    runner searches upward from the new checkpoint for ``dataset/`` metadata.
    Copying this small manifest prevents a history-trained policy from silently
    falling back to a single camera frame at inference time.
    """
    if args.reuse_dataset_from is None:
        return
    source = manifest_path(args)
    destination = args.artifact_root / "dataset" / source.name
    if source == destination:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"[train] copied history manifest for inference: {destination}", flush=True)


def split_episodes(
    episodes: list[EpisodeSource], val_count: int, seed: int
) -> tuple[list[int], list[int]]:
    indices = list(range(len(episodes)))
    if len(indices) <= 1:
        return indices, []
    # Human-rated self-imitation successes are newly collected training signal;
    # keep all of them in train and hold out only independent expert episodes.
    validation_candidates = [i for i, ep in enumerate(episodes) if ep.quality_score is None]
    random.Random(seed).shuffle(validation_candidates)
    n_val = min(max(0, val_count), len(validation_candidates), len(indices) - 1)
    val = sorted(validation_candidates[:n_val])
    train = sorted(i for i in indices if i not in set(val))
    return train, val


def state_history_vector(
    episode: EpisodeSource,
    frame_index: int,
    offsets: tuple[int, ...],
    tactile_mode: str = "none",
) -> np.ndarray:
    """Concatenate joint history and the latest tactile feedback."""
    values: list[float] = []
    for offset in offsets:
        source_index = max(0, frame_index - offset)
        values.extend(episode.rows[source_index]["joint_pos"][:JOINT_DIM])
    tactile = tactile_vector(episode.rows[frame_index], tactile_mode)
    if tactile is None:
        raise ValueError("episode row is missing required tactile feedback")
    values.extend(tactile.tolist())
    return np.asarray(values, dtype=np.float32)


def convert(args: argparse.Namespace) -> dict[str, Any]:
    episodes, skipped = discover_episodes(args)
    if not episodes:
        raise RuntimeError(f"No camera episodes found under {args.data_roots}")
    train_ids, val_ids = split_episodes(episodes, args.val_episodes, args.seed)
    root = dataset_root(args)
    summary = {
        "schema": "linkerhand_g20_visual_act_sources_v2",
        "data_root": str(args.data_roots[0]) if len(args.data_roots) == 1 else None,
        "data_roots": [str(path) for path in args.data_roots],
        "repo_id": args.repo_id,
        "fps": args.fps,
        "resample_rated_to_fps": args.resample_rated_to_fps,
        "image_hw": [args.image_height, args.image_width],
        "history_frame_offsets": list(args.history_frame_offsets),
        "history_layout": (
            "single"
            if len(args.history_frame_offsets) == 1
            else f"2x{len(args.history_frame_offsets) // 2}_oldest_to_newest"
        ),
        "state_history_offsets": list(args.state_history_offsets),
        "state_history_layout": "oldest_to_current",
        "tactile_mode": args.tactile_mode,
        "tactile_layout": (
            None
            if args.tactile_mode == "none"
            else "current_mass_6_then_contact_6"
        ),
        "state_dim": (
            JOINT_DIM * len(args.state_history_offsets)
            + TACTILE_DIMS[args.tactile_mode]
        ),
        "action_dim": JOINT_DIM,
        "chunk_size": args.chunk_size,
        "max_action_reversals": args.max_action_reversals,
        "min_thumb_route_reversals": args.min_thumb_route_reversals,
        "max_thumb_route_reversals": args.max_thumb_route_reversals,
        "momentum": {
            "weight": args.momentum_weight,
            "deadband_normalized": args.momentum_deadband,
            "margin_normalized": args.momentum_margin,
        },
        "freeze_profile": args.freeze_profile,
        "preserve_pretrained_stats": args.freeze_profile != "none",
        "include_task_ids": list(args.include_task_id),
        "success_filter": (
            None
            if args.success_reference is None
            else {
                "reference_image": str(args.success_reference),
                "aruco_dictionary": args.success_aruco_dictionary,
                "reference_marker_ids": list(load_success_reference(args).marker_ids),
                "maximum_marker_layout_error": args.success_max_layout_error,
                "comparison": "final_usable_frame",
            }
        ),
        "episodes": [
            {
                "dataset_episode_index": i,
                "source": source_name(ep.path, args.data_roots),
                "frames": len(ep.rows),
                "split": "validation" if i in val_ids else "train",
                "quality_score": ep.quality_score,
                "action_reversals": ep.action_reversals,
                "thumb_route_reversals": ep.thumb_route_reversals,
                "success_marker_layout_error": ep.success_marker_layout_error,
                "original_frames": ep.original_frames,
                "observed_fps": ep.observed_fps,
            }
            for i, ep in enumerate(episodes)
        ],
        "skipped": skipped,
        "train_episode_indices": train_ids,
        "validation_episode_indices": val_ids,
        "total_frames": sum(len(ep.rows) for ep in episodes),
    }
    print(
        f"[scan] camera episodes={len(episodes)} frames={summary['total_frames']} "
        f"train={len(train_ids)} val={len(val_ids)} skipped={len(skipped)} "
        f"history={list(args.history_frame_offsets)} "
        f"state_history={list(args.state_history_offsets)} "
        f"tactile={args.tactile_mode}"
    )
    if args.success_reference is not None:
        for episode in episodes:
            print(
                f"[success-keep] {source_name(episode.path, args.data_roots)} "
                f"layout_error={episode.success_marker_layout_error:.4f}"
            )
    if args.resample_rated_to_fps:
        for episode in episodes:
            if (
                episode.quality_score is not None
                and episode.original_frames is not None
                and episode.original_frames != len(episode.rows)
            ):
                print(
                    f"[rated-resample] {source_name(episode.path, args.data_roots)} "
                    f"{episode.original_frames}->{len(episode.rows)} frames "
                    f"observed={episode.observed_fps:.2f}Hz target={args.fps}Hz"
                )
    for item in skipped:
        if item["usable_frames"] == 0:
            print(f"[skip] {item['source']}: no usable camera frames")
        elif str(item["reason"]).startswith("final_frame_"):
            print(
                f"[success-skip] {item['source']}: {item['reason']} "
                f"ids={item.get('observed_marker_ids', [])} "
                f"layout_error={item.get('marker_layout_error')}"
            )
    if args.dry_run:
        return summary
    if root.exists():
        if not args.overwrite_dataset:
            raise RuntimeError(f"Dataset already exists: {root}; pass --overwrite-dataset")
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)

    import cv2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    image_feature = f"observation.images.{args.image_key}"
    state_names = [
        f"g20_joint_tminus_{offset:03d}_{joint:02d}"
        for offset in args.state_history_offsets
        for joint in range(JOINT_DIM)
    ]
    if args.tactile_mode == "mass-contact":
        state_names.extend(
            [f"g20_mass_{name}" for name in ("thumb", "index", "middle", "ring", "little", "palm")]
            + [
                f"g20_contact_{name}"
                for name in ("thumb", "index", "middle", "ring", "little", "palm")
            ]
        )
    state_dim = JOINT_DIM * len(args.state_history_offsets) + TACTILE_DIMS[
        args.tactile_mode
    ]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names,
        },
        image_feature: {
            # Keep frames as images. The installed torchcodec is ABI-incompatible
            # with the local PyTorch, and torchvision 0.26 removed VideoReader.
            "dtype": "image",
            "shape": (args.image_height, args.image_width, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (JOINT_DIM,),
            "names": [f"g20_command_{i:02d}" for i in range(JOINT_DIM)],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=str(root),
        robot_type="linkerhand_g20",
        use_videos=False,
    )
    for episode_idx, episode in enumerate(episodes):
        dropped = 0
        for frame_index, row in enumerate(episode.rows):
            bgr = history_mosaic(
                episode,
                frame_index,
                args.history_frame_offsets,
                args.image_width,
                args.image_height,
            )
            if bgr is None:
                dropped += 1
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dataset.add_frame({
                "observation.state": state_history_vector(
                    episode,
                    frame_index,
                    args.state_history_offsets,
                    args.tactile_mode,
                ),
                image_feature: np.ascontiguousarray(rgb, dtype=np.uint8),
                "action": np.asarray(row["last_action"][:JOINT_DIM], dtype=np.float32),
                "task": episode.task,
            })
        dataset.save_episode()
        print(
            f"[convert] {episode_idx + 1:02d}/{len(episodes):02d} "
            f"{episode.path.name}: {len(episode.rows) - dropped} frames"
        )
    manifest_path(args).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[convert] dataset: {root}")
    return summary


def load_manifest(args: argparse.Namespace) -> dict[str, Any]:
    path = manifest_path(args)
    if not path.is_file():
        raise RuntimeError(f"Missing dataset manifest: {path}; run --stage convert first")
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_training_manifest(
    args: argparse.Namespace, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Apply explicit source exclusions without changing the reused dataset."""
    prepared = copy.deepcopy(manifest)
    episodes = {
        str(episode["source"]): int(episode["dataset_episode_index"])
        for episode in prepared.get("episodes", [])
    }
    missing = [source for source in args.exclude_source_episode if source not in episodes]
    if missing:
        raise RuntimeError(
            "Excluded source episodes are absent from the manifest: "
            + ", ".join(missing)
        )
    excluded_indices = {episodes[source] for source in args.exclude_source_episode}
    if excluded_indices:
        before = len(prepared["train_episode_indices"])
        prepared["train_episode_indices"] = [
            int(index)
            for index in prepared["train_episode_indices"]
            if int(index) not in excluded_indices
        ]
        removed = before - len(prepared["train_episode_indices"])
        print(
            f"[train] excluded source episodes={sorted(excluded_indices)} "
            f"removed_from_train={removed}",
            flush=True,
        )
    if args.train_episode_indices is not None:
        available = {
            int(index) for index in prepared["train_episode_indices"]
        }
        requested = set(args.train_episode_indices)
        unavailable = sorted(requested - available)
        if unavailable:
            raise RuntimeError(
                "Requested --train-episode-indices are not in the manifest "
                f"training split: {unavailable}"
            )
        prepared["train_episode_indices"] = list(args.train_episode_indices)
        print(
            "[train] explicit episode subset="
            f"{prepared['train_episode_indices']}",
            flush=True,
        )
    return prepared


def distillation_edge_episode_indices(
    args: argparse.Namespace, manifest: dict[str, Any]
) -> list[int]:
    if args.distill_teacher is None:
        return []
    train_indices = {int(index) for index in manifest["train_episode_indices"]}
    matches = sorted(
        int(episode["dataset_episode_index"])
        for episode in manifest.get("episodes", [])
        if int(episode["dataset_episode_index"]) in train_indices
        and any(
            str(episode["source"]).startswith(prefix)
            for prefix in args.distill_edge_source_prefix
        )
    )
    if not matches:
        raise RuntimeError(
            "No retained training episodes match --distill-edge-source-prefix"
        )
    return matches


def resolve_device(args: argparse.Namespace) -> str:
    if args.device != "auto":
        return args.device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def training_command(args: argparse.Namespace, manifest: dict[str, Any]) -> list[str]:
    train_episodes = list(manifest["train_episode_indices"])
    if not train_episodes:
        raise RuntimeError("Training split is empty")
    edge_episode_indices = distillation_edge_episode_indices(args, manifest)
    if args.distill_teacher is not None:
        runner = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "lerobot_train_distilled.py"),
            "--freeze-profile",
            args.freeze_profile,
            "--preserve-pretrained-stats",
            "--teacher-checkpoint",
            str(args.distill_teacher),
            "--edge-episode-indices",
            ",".join(str(index) for index in edge_episode_indices),
            "--distill-base-weight",
            str(args.distill_base_weight),
            "--distill-edge-weight",
            str(args.distill_edge_weight),
        ]
    elif args.momentum_weight > 0:
        runner = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "lerobot_train_momentum.py"),
            "--momentum-weight",
            str(args.momentum_weight),
            "--momentum-deadband",
            str(args.momentum_deadband),
            "--momentum-margin",
            str(args.momentum_margin),
        ]
    elif args.freeze_profile == "none":
        runner = [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    else:
        runner = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "lerobot_train_frozen.py"),
            "--freeze-profile",
            args.freeze_profile,
            "--preserve-pretrained-stats",
        ]
    dataset_args = [
        "--dataset.repo_id", args.repo_id,
        "--dataset.root", str(dataset_root(args)),
        "--dataset.episodes", json.dumps(train_episodes, separators=(",", ":")),
        "--dataset.video_backend", args.video_backend,
        "--dataset.image_transforms.enable", "false",
    ]
    train_args = [
        "--output_dir", str(output_root(args)),
        "--job_name", "g20_visual_orientation_act",
        "--seed", str(args.seed),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--steps", str(args.steps),
        "--log_freq", str(args.log_freq),
        "--save_checkpoint", "true",
        "--save_freq", str(args.save_freq),
        "--wandb.enable", "false",
    ]
    policy_args: list[str]
    if args.finetune_from is not None:
        policy_args = [
            f"--policy.path={args.finetune_from}",
            f"--policy.device={resolve_device(args)}",
            f"--policy.use_amp={str(bool(args.amp)).lower()}",
            "--policy.push_to_hub=false",
            f"--policy.n_action_steps={args.n_action_steps}",
            f"--policy.optimizer_lr={args.finetune_learning_rate}",
            f"--policy.optimizer_lr_backbone={args.finetune_learning_rate}",
        ]
    else:
        policy_args = [
            "--policy.type", "act",
            "--policy.device", resolve_device(args),
            "--policy.use_amp", str(bool(args.amp)).lower(),
            "--policy.push_to_hub", "false",
            "--policy.chunk_size", str(args.chunk_size),
            "--policy.n_action_steps", str(args.n_action_steps),
            "--policy.dim_model", "256",
            "--policy.n_heads", "8",
            "--policy.dim_feedforward", "1024",
            "--policy.n_encoder_layers", "4",
            "--policy.n_decoder_layers", "1",
        ]
    # Keep path-style policy arguments together so draccus loads the checkpoint
    # configuration before applying its CLI overrides.
    return runner + dataset_args + policy_args + train_args


def train(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    out = output_root(args)
    if out.exists():
        if not args.overwrite_output:
            raise RuntimeError(f"Training output already exists: {out}; pass --overwrite-output")
        shutil.rmtree(out)
    cmd = training_command(args, manifest)
    print("[train] command:")
    print(" ".join(json.dumps(part) for part in cmd))
    print(
        f"[train] episodes={len(manifest['train_episode_indices'])} "
        f"held_out={manifest['validation_episode_indices']} device={resolve_device(args)} "
        f"finetune_from={args.finetune_from}"
    )
    if args.dry_run:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    persist_reused_manifest_for_inference(args)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> int:
    args = parse_args()
    ensure_training_python(args)
    manifest: dict[str, Any]
    if args.stage in ("all", "convert"):
        manifest = convert(args)
    else:
        manifest = load_manifest(args)
    if args.stage in ("all", "train"):
        manifest = prepare_training_manifest(args, manifest)
        train(args, manifest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
