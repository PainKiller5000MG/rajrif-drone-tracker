# 8. Running detect_drone.py

Standalone, **read-only** drone detection. Camera in, detections out, to
screen and a log file. No tracking across frames, no mount/serial control,
no pan/tilt output, no fire trigger of any kind — deliberately kept
separate from every tracker/mount file in this repo.

Two identical copies exist: `drone_detection_module/detect_drone.py` (the
original) and `drone_tracking_module/detect_drone.py` (cloned alongside
the newer module, left untouched). Either works the same way.

## Launch

```
cd drone_detection_module
python detect_drone.py --weights weights/drone_yolov11x.pt
```

Prompts for camera device index if `--source` isn't given (see
`04-camera-setup.md`).

## Behavior worth knowing

- **As of 2026-08-15, `--conf-threshold` gates everything**: what YOLO
  returns at all, what gets drawn on screen (green box + `class conf`
  label), what gets logged, and the console alert. Default is now `0.6`.
  (Previously this script passed no `conf=` to `model.predict()`, so
  Ultralytics' internal default of 0.25 silently decided what was drawn/
  logged, while `--conf-threshold` only gated the console print — fixed.)
- Logs every detection that clears `--conf-threshold` to a CSV file in the
  same folder, timestamped.
- The weights-size warning (heavy-model-on-small-GPU check) requires
  typing `y` to continue if it triggers — this is intentional, not a bug
  to script around.

## Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--conf-threshold` | `0.6` | confidence floor for a detection to be returned/drawn/logged/alerted at all |
| `--max-box-area-frac` | `0.5` | rejects frame-spanning false positives (see `11-troubleshooting.md`'s webcam-face incident) |
| `--half` / `--no-half` | half on | FP16 inference, requires CUDA |
| `--source` | prompts | camera index or video file path |
| `--no-display` | off | headless, for automated testing |
| `--log-format` | `csv` | or `json` |

## Testing on a video instead of a live camera

```
python detect_drone.py --weights weights/drone_yolov11x.pt \
  --source ../20RajRif/test_out/drone_flight.mp4
```
