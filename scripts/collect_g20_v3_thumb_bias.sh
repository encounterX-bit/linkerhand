#!/usr/bin/env bash
set -eo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
run_python="/home/zhaoyan-qian/Desktop/Jacky/ros2_pairlab3-main/.venv_ros2_pairlab3/bin/python"

source /opt/ros/jazzy/setup.bash
source /home/zhaoyan-qian/Desktop/Jacky/linker_hand_ros2_sdk/install/setup.bash
set -u
export HW_ENABLE_TOKEN=1

cd "${repo_root}"

exec "${run_python}" \
  -m src.comms.visual_act_to_linkerhand \
  --checkpoint-dir artifacts/g20_flipping_act_full_context_momentum_v3/training/checkpoints/030000/pretrained_model \
  --camera-index 0 \
  --rate 30 \
  --n-action-steps 30 \
  --max-range-step 20 \
  --ema-alpha 0.10 \
  --keyboard-thumb-bias \
  --keyboard-thumb-bias-joint 15 \
  --thumb-bias-step 5 \
  --thumb-bias-limit 80 \
  --keyboard-thumb-side-bias \
  --thumb-side-bias-step 5 \
  --thumb-side-bias-limit 80 \
  --keyboard-action-library data/action_library/g20_right/core_actions_v1 \
  --action-intervention-blend-frames 8 \
  --thumb-final-push-offset 10 \
  --current-limit 180 \
  --speed-limit 180 \
  --max-target-delta 210 \
  --max-raw-overshoot 50 \
  --state-stale-seconds 2.0 \
  --reset-tolerance 12 \
  --max-active-seconds 0 \
  --auto-stop-endpoint-profile artifacts/g20_flipping_act_full_context_momentum_v3/dataset/g20_endpoint_profiles.json \
  --auto-stop-endpoint-tolerance 12 \
  --auto-stop-endpoint-confirm-frames 10 \
  --auto-stop-min-active-seconds 8 \
  --auto-stop-departure-delta 20 \
  --hold-on-disarm \
  --record-rated-attempts \
  --rated-output-dir data/self_imitation/flipping_v3_human_rated \
  --record-rate 30 \
  --jpeg-quality 92 \
  --ignore-touch \
  --no-require-touch-for-score-one \
  --no-stop-on-contact-success \
  --reset-after-rating \
  --reset-range-step 25 \
  --enable-motion
