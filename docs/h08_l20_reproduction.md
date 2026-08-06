# Being-H0.8-style LinkerHand L20/G20 reproduction

## Scope

Being-H0.8 was announced on 2026-07-28, but its training code, TopoHand
adapters, TactoHand model, and checkpoints are not public as of 2026-07-31.
The official demonstration names a LinkerHand L25 with 16 actuated and 21 total
joints. This repository's L20/G20 has the same counts and records five 12x6
fingertip pressure arrays, so it is suitable for a local tactile-feedback
baseline, but the result must not be presented as the official H0.8 checkpoint.

The implemented first stage is deliberately smaller:

- visual context: existing current/history camera mosaic;
- proprioception: existing 20-value SDK state/history;
- tactile feedback: current six regional masses plus six contact bits;
- action: existing 20-value absolute SDK target chunk;
- execution: existing re-plan/step-limit/contact-staleness safety path.

This captures H0.8's practical slow/fast idea: visual and state history provide
the slower plan context, while the newest tactile sample is refreshed on every
policy call. It does not yet implement future-aware posterior distillation,
TopoHand's morphology VAE, or a full spatial tactile Perceiver.

## 1. Audit existing data

```bash
cd /home/zhaoyan-qian/Desktop/Jacky/sims/linker-hand-teleopt
.venv/bin/python scripts/audit_h08_l20_data.py --data-root data
```

Only episodes with image, 20-D state, 20-D action, `mass_values[6]`, and
`contact_6[6]` enter the tactile baseline.

## 2. Build a small dry-run dataset

This command does not train and never imports ROS:

```bash
python scripts/train_g20_visual_act.py \
  --stage convert \
  --data-root data \
  --artifact-root artifacts/h08_lite_g20_smoke \
  --repo-id h08_lite_g20_smoke \
  --task "rotate the cube clockwise using tactile feedback" \
  --tactile-mode mass-contact \
  --history-frame-offsets 15,10,5,0 \
  --state-history-offsets 15,5,0 \
  --max-episodes 2
```

For the real dataset, select one consistent task ID and successful
demonstrations rather than mixing every folder. Train with the existing
`--stage train` path after inspecting the generated source manifest.

## 3. Offline and live evaluation

Compare a vision/state checkpoint against an identically trained
`--tactile-mode mass-contact` checkpoint. Report:

- held-out action MAE;
- contact-onset action MAE;
- peak pressure and pressure variance;
- object-rotation success rate;
- brittle-object or perturbation recovery rate.

The live runner reads `tactile_mode` from the checkpoint's nearby dataset
manifest. A tactile checkpoint refuses `--ignore-touch`, waits for a tactile
sample at startup, and pauses policy calls when touch becomes stale. It remains
dry-run unless the existing independent hardware gates are explicitly enabled
by the human operator.

## 4. Path toward a closer H0.8 reproduction

1. Replace the 12 coarse features with a spatial encoder over the five 12x6
   matrices and validity masks.
2. Add a future vision+tactile posterior during training and align it with a
   current-observation prior.
3. Cache visual-language context and refresh only state/tactile tokens at the
   fast control rate.
4. Implement an L20-to-TopoHand adapter from the URDF's 16 active joints to 20
   canonical articulation slots.
5. Fine-tune per task, then evaluate touch ablations and cross-object transfer.
