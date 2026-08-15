# 7. Running detect_and_track.py

The newest module — loads the YOLO model directly, no manual click/drag
acquisition at all, with two switchable follow modes and full fire
control. `tracker_core.py`, `mount.py`, and the PD control law
(`_axis_cmd`) are imported live from `../20RajRif`, never copied.

## Folder layout required

`drone_tracking_module/` must sit **next to** `20RajRif/` (same parent
folder) — the import depends on that relative path.

```
army/
    20RajRif/
    drone_tracking_module/
```

## Basic launch (no mount, prompts for camera)

```
cd drone_tracking_module
python detect_and_track.py --weights weights/drone_yolov11x.pt
```

## Full mount control

```
python detect_and_track.py \
    --weights weights/drone_yolov11x.pt \
    --port COM3 \
    --mount-type "ESP32 DC (BTS7960)" \
    --mode lock
```

## Test against a recorded clip, no hardware

```
python detect_and_track.py \
    --weights weights/drone_yolov11x.pt \
    --source ../20RajRif/test_out/drone_flight.mp4
```

## Controls while running

| Key | Action |
|---|---|
| `q` | quit |
| `m` | switch RAW ↔ LOCK, live |
| `c` | clear a LOCK |
| `w` / `s` | manual tilt up / down |
| `a` / `d` | manual pan left / right |
| `space` | **panic stop** — kills movement AND fire, disarms AUTO FIRE, unconditionally |
| `f` | toggle manual FIRE (no gating at all) |
| `r` | arm/disarm AUTO FIRE |

The full control list is drawn on the video window itself, bottom-left.
Manual jog (`w`/`a`/`s`/`d`) and manual fire (`f`) always override
whatever the auto-follow/auto-fire logic wants to do that frame.

**Known limitation**: manual jog is driven by keyboard polling
(`cv2.waitKey`), not true key-up/down events like the D-pad buttons in
`Old/main.py` — a held key can feel a bit pulsed rather than perfectly
smooth. Not a bug, a known tradeoff of this simpler control scheme.

## Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `lock` | starting mode, `lock` or `raw` |
| `--conf-threshold` | `0.80` | confidence floor for any detection to count |
| `--max-box-area-frac` | `0.10` | reject detections covering more than this fraction of frame |
| `--confirm-frames` | `4` | LOCK mode: consistent cycles before auto-locking |
| `--port` | `None` | mount serial port; omit to log commands instead of sending them |
| `--mount-type` | `ESP32 DC (BTS7960)` | or `Stepper (x=/y=)` |
| `--no-display` | off | headless, no video window |
| `--no-half` | off | disable FP16 — use if hitting GPU memory errors |

Full mode explanation: [Old vs New vs Raw vs Lock](09-old-new-raw-lock.md).
Fire control detail: [Fire control and safety](10-fire-control-safety.md).

## Verify before trusting this

Optional test suite (headless, no hardware needed):
```
cd ../20RajRif
python test_tracking.py
python test_integration.py
python test_auto_acquire.py
```
Expect 13/14 (the one failure is a CPU-speed check, irrelevant on GPU
hardware), 3/3, and 8/8 respectively.
