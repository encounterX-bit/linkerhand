#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
train_python="/home/zhaoyan-qian/Desktop/Jacky/ros2_pairlab3-main/.venv_ros2_pairlab3/bin/python"
v3_checkpoint="artifacts/g20_flipping_act_full_context_momentum_v3/training/checkpoints/030000/pretrained_model"

cd "${repo_root}"

exec "${train_python}" scripts/train_g20_visual_act.py \
  --stage all \
  --overwrite-dataset \
  --data-root data/act_demos_clean/flipping_full_20260722_170628_anyface_v1 \
  --data-root data/act_demos_clean/flipping_canonical_v1 \
  --data-root data/self_imitation/flipping_v3_human_rated/20260728_144717_act_self_imitation \
  --data-root data/self_imitation/flipping_v3_human_rated/20260728_151216_act_self_imitation \
  --data-root data/self_imitation/flipping_v3_human_rated/20260728_154309_act_self_imitation \
  --data-root data/self_imitation/flipping_v3_human_rated/20260728_154948_act_self_imitation \
  --include-task-id action_library_hybrid_demo_v1_full_anyface_clean_v1 \
  --include-task-id action_library_hybrid_demo_v1_canonical_clean_v1 \
  --include-task-id orientation_grasp_self_imitation \
  --min-rated-score 1.0 \
  --resample-rated-to-fps \
  --artifact-root artifacts/g20_flipping_act_v3_posttrain_20260728_v1 \
  --repo-id linkerhand_g20_flipping_act_v3_posttrain_20260728_v1 \
  --task "continuously flip the tagged cube according to visual and motion history" \
  --history-frame-offsets "120,96,72,48,24,0" \
  --state-history-offsets "120,90,60,30,0" \
  --val-episodes 3 \
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
  --steps 3000 \
  --batch-size 16 \
  --num-workers 4 \
  --save-freq 1000 \
  --device cuda \
  --amp
