#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
train_python="/home/zhaoyan-qian/Desktop/Jacky/ros2_pairlab3-main/.venv_ros2_pairlab3/bin/python"

source_artifact="artifacts/g20_flipping_act_v3_posttrain_20260728_v1"
source_checkpoint="${source_artifact}/training/checkpoints/003000/pretrained_model"
artifact_root="artifacts/g20_flipping_act_v3_posttrain_20260729_more_v2"
additional_steps="${POSTTRAIN_MORE_STEPS:-20000}"

cd "${repo_root}"

# Keep the desktop responsive and avoid the high-memory worker configuration
# that previously froze the machine.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

"${train_python}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print(
        "CUDA is unavailable. Verify nvidia-smi before retrying.",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"[continue] CUDA ready: {torch.cuda.get_device_name(0)}", flush=True)
PY

test -f "${source_artifact}/dataset/g20_source_manifest.json"
test -f "${source_checkpoint}/model.safetensors"

if [[ -e "${artifact_root}/training" ]]; then
    echo "Training output already exists: ${artifact_root}/training" >&2
    echo "Use its saved checkpoints or choose a new artifact_root; nothing was overwritten." >&2
    exit 2
fi

exec ionice -c 2 -n 7 nice -n 10 \
  "${train_python}" scripts/train_g20_visual_act.py \
  --stage train \
  --reuse-dataset-from "${source_artifact}" \
  --artifact-root "${artifact_root}" \
  --repo-id linkerhand_g20_flipping_act_v3_posttrain_20260729_more_v2 \
  --task "continuously flip the tagged cube according to visual and motion history" \
  --history-frame-offsets "120,96,72,48,24,0" \
  --state-history-offsets "120,90,60,30,0" \
  --chunk-size 30 \
  --n-action-steps 30 \
  --finetune-from "${source_checkpoint}" \
  --finetune-learning-rate 1e-6 \
  --freeze-profile edge-head \
  --distill-teacher "${source_checkpoint}" \
  --distill-base-weight 2.0 \
  --distill-edge-weight 0.1 \
  --distill-edge-source-prefix "20260728_144717_act_self_imitation/" \
  --distill-edge-source-prefix "20260728_151216_act_self_imitation/" \
  --distill-edge-source-prefix "20260728_154309_act_self_imitation/" \
  --distill-edge-source-prefix "20260728_154948_act_self_imitation/" \
  --train-episode-indices "0,18,21,34,35,36,37,38,39,40,41,42,43,44,45,46" \
  --steps "${additional_steps}" \
  --batch-size 4 \
  --num-workers 0 \
  --save-freq 1000 \
  --device cuda \
  --amp
