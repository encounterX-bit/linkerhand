# Hand-clip provenance & licensing

Test footage for the `--source video` path (video → MediaPipe → retarget →
safety.filter → sim). Fetched 2026-06-09 from Wikimedia Commons (free-to-use,
direct upload.wikimedia.org URLs). **No scraped / YouTube content.**

Only the tiny CC0 clip below is committed (reproducibility fixture); the larger
clips are gitignored — re-fetch them with the URLs here. All are monocular RGB,
so this validates **plumbing**, not depth/retarget accuracy (that is RealSense).

| file | source (Wikimedia Commons) | license | dur | size | committed |
|------|----------------------------|---------|-----|------|-----------|
| `hand_gesture_67.webm` | [File:6-7 hand gesture 2025.webm](https://commons.wikimedia.org/wiki/File:6-7_hand_gesture_2025.webm) — User:Buster-Nutt-67 | **CC0** | 3.1s | 0.57 MB | **yes** |
| `woman_counting_on_fingers.webm` | [File:Woman counting on fingers.webm](https://commons.wikimedia.org/wiki/File:Woman_counting_on_fingers.webm) | **CC0** | 12.8s | 5.2 MB | no (gitignored) |
| `finger_counting_dutch.webm` | [File:Finger-counting in Dutch.webm](https://commons.wikimedia.org/wiki/File:Finger-counting_in_Dutch.webm) | **CC BY-SA 4.0** | 13.3s | 5.5 MB | no (gitignored) |
| `hand_wave_example.webm` | [File:HandWaveExample.webm](https://commons.wikimedia.org/wiki/File:HandWaveExample.webm) — User:NMu11er | **CC BY-SA 4.0** | 5.2s | 0.62 MB | no (gitignored) |

Direct download URLs:
- https://upload.wikimedia.org/wikipedia/commons/6/6e/6-7_hand_gesture_2025.webm
- https://upload.wikimedia.org/wikipedia/commons/e/e5/Woman_counting_on_fingers.webm
- https://upload.wikimedia.org/wikipedia/commons/9/97/Finger-counting_in_Dutch.webm
- https://upload.wikimedia.org/wikipedia/commons/5/58/HandWaveExample.webm

CC BY-SA 4.0 attribution (if these clips are redistributed): credit the linked
Commons author pages above; share-alike applies.

## How they were used
- `scripts/validate_hand_videos.py assets/video/*.webm --out tests/viz/out/detection_report.json`
  — MediaPipe per-frame detection rate + handedness (the "bad clip vs broken
  pipeline" gate; discard < 70%).
- `scripts/stream_hand_videos.py assets/video/<clip>.webm --out-dir tests/viz/out`
  — streams the survivor through the real viz loop and dumps a sim GIF.
