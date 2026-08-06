#!/usr/bin/env python3
"""Train a frozen ACT student while anchoring deployed chunks to a teacher.

The normal ACT behavior-cloning loss uses the VAE training path.  This wrapper
adds a second, deterministic inference-path forward pass and constrains its
entire action chunk to remain near a known-good teacher.  Base/replay samples
receive a strong teacher constraint; selected edge samples retain a weak
constraint and a larger demonstration weight.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot_train_frozen import (
    FREEZE_PROFILES,
    apply_freeze_profile,
    format_summary,
    output_dir_from_args,
    preserve_pretrained_processor_stats,
    write_or_validate_freeze_metadata,
)

G20_DISTILL_JOINTS = (1, 2, 3, 4, 6, 7, 8, 9, 16, 17, 18, 19)
NORMALIZATION_STD_FLOOR = 1.0


def install_memory_safe_subset_loader() -> None:
    """Avoid fingerprinting a multi-gigabyte in-memory Arrow subset.

    LeRobot's filtered parquet path builds one in-memory Arrow table and then
    constructs ``datasets.Dataset(table)``.  Hugging Face generates a
    fingerprint by dill-serializing that entire table, temporarily requiring a
    second full-size copy.  Supplying a deterministic fingerprint keeps only
    the table that is actually needed for training.
    """
    import datasets
    import pyarrow.dataset as pa_ds
    import lerobot.datasets.lerobot_dataset as lerobot_dataset
    import lerobot.datasets.utils as dataset_utils

    original_loader = dataset_utils.load_nested_dataset

    def load_nested_dataset_without_copy(
        pq_dir: Path,
        features: datasets.Features | None = None,
        episodes: list[int] | None = None,
    ) -> datasets.Dataset:
        if episodes is None:
            return original_loader(pq_dir, features=features, episodes=episodes)

        pq_dir = Path(pq_dir)
        paths = sorted(pq_dir.glob("*/*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"Provided directory does not contain any parquet file: {pq_dir}"
            )

        selected = tuple(sorted({int(index) for index in episodes}))
        arrow_dataset = pa_ds.dataset(paths, format="parquet")
        table = arrow_dataset.to_table(
            filter=pa_ds.field("episode_index").isin(selected)
        )
        if features is not None:
            table = table.cast(features.arrow_schema)

        digest = hashlib.sha256()
        digest.update(str(pq_dir.resolve()).encode())
        digest.update(repr(selected).encode())
        for path in paths:
            stat = path.stat()
            digest.update(str(path.resolve()).encode())
            digest.update(f":{stat.st_size}:{stat.st_mtime_ns}".encode())
        fingerprint = digest.hexdigest()
        print(
            f"[dataset] memory-safe subset: episodes={len(selected)} "
            f"rows={table.num_rows} fingerprint={fingerprint[:12]}",
            flush=True,
        )
        return datasets.Dataset(table, fingerprint=fingerprint)

    # lerobot_dataset imports the function directly, so patch both references.
    dataset_utils.load_nested_dataset = load_nested_dataset_without_copy
    lerobot_dataset.load_nested_dataset = load_nested_dataset_without_copy


def stabilize_zero_variance_stats(*pipelines: Any) -> None:
    """Prevent float interpolation noise from exploding constant joint values."""
    adjusted: list[str] = []
    for pipeline_name, pipeline in zip(("pre", "post"), pipelines, strict=True):
        for step in pipeline.steps:
            features = getattr(step, "features", {})
            tensor_stats = getattr(step, "_tensor_stats", {})
            raw_stats = getattr(step, "stats", {})
            for feature_name in features:
                feature_stats = tensor_stats.get(feature_name, {})
                std = feature_stats.get("std")
                if std is None:
                    continue
                zero_variance = std.abs() < 1e-6
                count = int(zero_variance.sum().item())
                if not count:
                    continue
                safe_std = torch.where(
                    zero_variance,
                    torch.full_like(std, NORMALIZATION_STD_FLOOR),
                    std,
                )
                feature_stats["std"] = safe_std
                if feature_name in raw_stats and "std" in raw_stats[feature_name]:
                    raw_std = safe_std.detach().cpu().numpy()
                    raw_stats[feature_name]["std"] = raw_std
                adjusted.append(f"{pipeline_name}:{feature_name}={count}")
    print(
        "[distill] zero-variance std floor="
        f"{NORMALIZATION_STD_FLOOR:g} adjusted={','.join(adjusted) or 'none'}",
        flush=True,
    )


def parse_episode_indices(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("episode indices must be comma-separated integers") from exc
    if not result or result[0] < 0:
        raise argparse.ArgumentTypeError("at least one non-negative edge episode index is required")
    return result


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--freeze-profile", choices=sorted(FREEZE_PROFILES), required=True)
    parser.add_argument("--preserve-pretrained-stats", action="store_true")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--edge-episode-indices", type=parse_episode_indices, required=True)
    parser.add_argument("--distill-base-weight", type=float, default=2.0)
    parser.add_argument("--distill-edge-weight", type=float, default=0.1)
    args, remaining = parser.parse_known_args(argv)
    if any(value in ("-h", "--help") for value in remaining):
        return args, remaining
    args.teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
    if not (args.teacher_checkpoint / "model.safetensors").is_file():
        parser.error(f"invalid teacher checkpoint: {args.teacher_checkpoint}")
    if not args.preserve_pretrained_stats:
        parser.error("distillation requires --preserve-pretrained-stats")
    if args.distill_base_weight < 0 or args.distill_edge_weight < 0:
        parser.error("distillation weights must be non-negative")
    if args.distill_base_weight == 0 and args.distill_edge_weight == 0:
        parser.error("at least one distillation weight must be positive")
    return args, remaining


def write_or_validate_distillation_metadata(
    output_dir: Path | None, args: argparse.Namespace
) -> None:
    if output_dir is None:
        print("[distill] warning: output_dir unavailable; metadata not saved", flush=True)
        return
    path = output_dir / "distillation.json"
    metadata = {
        "schema": "linkerhand_act_distillation_v1",
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "edge_episode_indices": list(args.edge_episode_indices),
        "distill_base_weight": args.distill_base_weight,
        "distill_edge_weight": args.distill_edge_weight,
        "distill_joint_indices": list(G20_DISTILL_JOINTS),
        "distill_smooth_l1_beta": 0.05,
        "normalization_std_floor": NORMALIZATION_STD_FLOOR,
        "loss": "standard ACT BC+KL plus deterministic full-chunk teacher SmoothL1",
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise RuntimeError(f"distillation metadata mismatch in {path}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def make_model_batch(policy: Any, batch: dict[str, Any]) -> dict[str, Any]:
    from lerobot.utils.constants import OBS_IMAGES

    model_batch = dict(batch)
    if policy.config.image_features:
        model_batch[OBS_IMAGES] = [
            model_batch[key] for key in policy.config.image_features
        ]
    return model_batch


def load_teacher(checkpoint: Path, student: Any) -> Any:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    config = PreTrainedConfig.from_pretrained(
        str(checkpoint),
        local_files_only=True,
        cli_overrides=[
            "--device",
            str(student.config.device),
            "--n_action_steps",
            str(student.config.n_action_steps),
        ],
    )
    if config.type != "act":
        raise RuntimeError(f"teacher policy must be ACT, got {config.type!r}")
    if config.input_features != student.config.input_features:
        raise RuntimeError("teacher and student input features differ")
    if config.output_features != student.config.output_features:
        raise RuntimeError("teacher and student output features differ")
    if config.chunk_size != student.config.chunk_size:
        raise RuntimeError("teacher and student chunk sizes differ")
    teacher_class = get_policy_class(config.type)
    teacher = teacher_class.from_pretrained(
        str(checkpoint), config=config, local_files_only=True
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def install_distilled_forward(
    student: Any, teacher: Any, args: argparse.Namespace
) -> None:
    edge_indices = tuple(args.edge_episode_indices)
    forward_calls = 0
    original_forward = student.forward

    def distilled_forward(
        policy: Any, batch: dict[str, Any], *_args: Any, **_kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, float]]:
        nonlocal forward_calls
        forward_calls += 1
        if "episode_index" not in batch:
            raise RuntimeError("distillation batch is missing episode_index")
        behavior_loss, output = original_forward(batch, *_args, **_kwargs)
        model_batch = make_model_batch(policy, batch)
        edge_ids = torch.as_tensor(
            edge_indices, device=batch["episode_index"].device, dtype=batch["episode_index"].dtype
        )
        edge_mask = torch.isin(batch["episode_index"], edge_ids)

        student_was_training = policy.model.training
        policy.model.eval()
        try:
            deployed_student_actions = policy.model(model_batch)[0]
        finally:
            policy.model.train(student_was_training)
        with torch.no_grad():
            deployed_teacher_actions = teacher.model(make_model_batch(teacher, batch))[0]

        per_sample_distill = F.smooth_l1_loss(
            deployed_student_actions[..., G20_DISTILL_JOINTS],
            deployed_teacher_actions[..., G20_DISTILL_JOINTS].to(
                deployed_student_actions.dtype
            ),
            reduction="none",
            beta=0.05,
        ).mean(dim=(1, 2))
        distill_weights = torch.where(
            edge_mask,
            torch.as_tensor(
                args.distill_edge_weight,
                device=deployed_student_actions.device,
                dtype=deployed_student_actions.dtype,
            ),
            torch.as_tensor(
                args.distill_base_weight,
                device=deployed_student_actions.device,
                dtype=deployed_student_actions.dtype,
            ),
        )
        distill_loss = (per_sample_distill * distill_weights).mean()
        loss = behavior_loss + distill_loss
        output["distill_loss"] = float(distill_loss.detach())
        output["teacher_chunk_smooth_l1"] = float(
            per_sample_distill.detach().mean()
        )
        output["edge_batch_fraction"] = float(edge_mask.float().mean().detach())
        if forward_calls == 1 or forward_calls % 100 == 0:
            print(
                f"[distill] step={forward_calls} "
                f"edge_batch={output['edge_batch_fraction']:.2f} "
                f"bc_l1={output['l1_loss']:.4f} "
                f"kd={output['distill_loss']:.6f} "
                f"teacher_chunk={output['teacher_chunk_smooth_l1']:.6f}",
                flush=True,
            )
        return loss, output

    student.forward = types.MethodType(distilled_forward, student)


def main() -> None:
    wrapper_args, lerobot_args = parse_wrapper_args(sys.argv[1:])

    import lerobot.scripts.lerobot_train as lerobot_train

    install_memory_safe_subset_loader()
    original_make_policy = lerobot_train.make_policy
    original_make_processors = lerobot_train.make_pre_post_processors
    original_make_optimizer = lerobot_train.make_optimizer_and_scheduler
    output_dir = output_dir_from_args(lerobot_args)

    def make_distilled_policy(*args: Any, **kwargs: Any) -> Any:
        student = original_make_policy(*args, **kwargs)
        initialized_from = getattr(student.config, "pretrained_path", None)
        if initialized_from is None or Path(initialized_from).expanduser().resolve() != wrapper_args.teacher_checkpoint:
            raise RuntimeError(
                "distilled student must initialize from the exact teacher checkpoint"
            )
        summary = apply_freeze_profile(student, wrapper_args.freeze_profile)
        teacher = load_teacher(wrapper_args.teacher_checkpoint, student)
        install_distilled_forward(student, teacher, wrapper_args)
        write_or_validate_freeze_metadata(
            output_dir, summary, wrapper_args.preserve_pretrained_stats
        )
        write_or_validate_distillation_metadata(output_dir, wrapper_args)
        for line in format_summary(summary):
            print(line, flush=True)
        print(
            f"[distill] teacher={wrapper_args.teacher_checkpoint} "
            f"base_weight={wrapper_args.distill_base_weight:g} "
            f"edge_weight={wrapper_args.distill_edge_weight:g} "
            f"edge_episodes={list(wrapper_args.edge_episode_indices)}",
            flush=True,
        )
        return student

    def make_processors(*args: Any, **kwargs: Any) -> Any:
        kwargs = preserve_pretrained_processor_stats(copy.deepcopy(kwargs))
        print("[distill] processor stats: preserved from teacher checkpoint", flush=True)
        processors = original_make_processors(*args, **kwargs)
        stabilize_zero_variance_stats(*processors)
        return processors

    def make_checked_optimizer(*args: Any, **kwargs: Any) -> Any:
        optimizer, scheduler = original_make_optimizer(*args, **kwargs)
        optimized = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        policy = args[1] if len(args) > 1 else kwargs["policy"]
        expected = {
            id(parameter)
            for parameter in policy.parameters()
            if parameter.requires_grad
        }
        if optimized != expected:
            raise RuntimeError(
                "optimizer parameters do not exactly match the freeze profile"
            )
        return optimizer, scheduler

    lerobot_train.make_policy = make_distilled_policy
    lerobot_train.make_pre_post_processors = make_processors
    lerobot_train.make_optimizer_and_scheduler = make_checked_optimizer
    sys.argv = [sys.argv[0], *lerobot_args]
    lerobot_train.main()


if __name__ == "__main__":
    main()
