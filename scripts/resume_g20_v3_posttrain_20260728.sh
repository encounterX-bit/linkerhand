#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
train_python="/home/zhaoyan-qian/Desktop/Jacky/ros2_pairlab3-main/.venv_ros2_pairlab3/bin/python"
artifact_root="artifacts/g20_flipping_act_v3_posttrain_20260728_v1"
v3_checkpoint="artifacts/g20_flipping_act_full_context_momentum_v3/training/checkpoints/030000/pretrained_model"
posttrain_steps="${POSTTRAIN_STEPS:-3000}"

cd "${repo_root}"

# Keep the desktop responsive while the student and distillation teacher share
# one GPU.  Unbuffered output also makes the last completed stage visible if a
# run is interrupted.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

"${train_python}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print(
        "CUDA is unavailable. Reboot first, then verify nvidia-smi before retrying.",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"[resume] CUDA ready: {torch.cuda.get_device_name(0)}", flush=True)
PY

test -f "${artifact_root}/dataset/g20_source_manifest.json"
test ! -e "${artifact_root}/training"

exec ionice -c 2 -n 7 nice -n 10 \
  "${train_python}" scripts/train_g20_visual_act.py \
  --stage train \
  --artifact-root "${artifact_root}" \
  --repo-id linkerhand_g20_flipping_act_v3_posttrain_20260728_v1 \
  --task "continuously flip the tagged cube according to visual and motion history" \
  --history-frame-offsets "120,96,72,48,24,0" \
  --state-history-offsets "120,90,60,30,0" \
  --chunk-size 30 \
  --n-action-steps 30 \
  --finetune-from "${v3_checkpoint}" \
  --finetune-learning-rate 1e-6 \
  --freeze-profile edge-head \
  --distill-teacher "${v3_checkpoint}" \
  --distill-base-weight 2.0 \
  --distill-edge-weight 0.1 \
  --distill-edge-source-prefix "20260728_144717_act_self_imitation/" \
  --distill-edge-source-prefix "20260728_151216_act_self_imitation/" \
  --distill-edge-source-prefix "20260728_154309_act_self_imitation/" \
  --distill-edge-source-prefix "20260728_154948_act_self_imitation/" \
  --train-episode-indices "0,18,21,34,35,36,37,38,39,40,41,42,43,44,45,46" \
  --steps "${posttrain_steps}" \
  --batch-size 4 \
  --num-workers 0 \
  --save-freq 1000 \
  --device cuda \
  --amp
