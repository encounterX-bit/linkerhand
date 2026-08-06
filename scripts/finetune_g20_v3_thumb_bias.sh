#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
train_python="/home/zhaoyan-qian/Desktop/Jacky/ros2_pairlab3-main/.venv_ros2_pairlab3/bin/python"

cd "${repo_root}"

exec "${train_python}" scripts/train_g20_visual_act.py \
  --stage all \
  --overwrite-dataset \
  --data-root data/act_demos_clean/flipping_full_20260722_170628_anyface_v1 \
  --data-root data/self_imitation/flipping_v3_human_rated \
  --include-task-id action_library_hybrid_demo_v1_full_anyface_clean_v1 \
  --include-task-id orientation_grasp_self_imitation \
  --min-rated-score 1.0 \
  --artifact-root artifacts/g20_flipping_act_v3_thumb_bias_ft_v1 \
  --repo-id linkerhand_g20_flipping_act_v3_thumb_bias_ft_v1 \
  --task "continuously flip the tagged cube according to visual and motion history" \
  --history-frame-offsets "120,96,72,48,24,0" \
  --state-history-offsets "120,90,60,30,0" \
  --val-episodes 3 \
  --chunk-size 30 \
  --n-action-steps 30 \
  --momentum-weight 0.2 \
  --momentum-deadband 0.01 \
  --momentum-margin 0.005 \
  --finetune-from artifacts/g20_flipping_act_full_context_momentum_v3/training/checkpoints/030000/pretrained_model \
  --finetune-learning-rate 3e-6 \
  --steps 5000 \
  --batch-size 8 \
  --num-workers 4 \
  --save-freq 1000 \
  --device cuda \
  --amp
