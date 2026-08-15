# 6. Running Old/main.py

The fuller "operator station" — D-pad, FIRE/AUTO FIRE, IDMS azimuth slew,
gyro fallback, map view — with the same `tracker_core.py` identity lock
underneath, plus YOLO auto-acquire (added this session) on top.

## Launch

**Must be run from inside `20RajRif/Old/`** — icons and
`mount_config.json` are loaded relative to the working directory, not the
script's own location.

```
cd 20RajRif/Old
python main.py
```

## Dependencies specific to this app

- `tkintermapview` (map view) — `python3 -m pip install tkintermapview`
- `Pillow` — usually already pulled in transitively, but if you see
  `ModuleNotFoundError: No module named 'PIL'`, `pip install pillow`.

## First-run config

`Old/mount_config.json` auto-generates on first run if missing, with
placeholder default site coordinates baked into `load_mount_data.py`'s
source. Edit the generated JSON directly to set your real
`mount_latitude`/`mount_longitude`, `ESP32_PORT`, and `Gyro_PORT` — do not
rely on the hardcoded defaults for a real deployment.

**This file is deliberately excluded from git** (see the root `.gitignore`)
since it carries real site coordinates. It regenerates automatically —
just re-edit it after any fresh clone/checkout.

## Interface

| Control | Effect |
|---|---|
| Drag/click in video panel | manual lock, same as `tracker_gui.py` |
| D-pad | manual pan/tilt (disabled while tracking is on) |
| TRACKING button | toggles whether the lock drives the mount |
| MODE: PROFILE / BLOB | toggles lock type, clears any lock, centers mount |
| FIRE (hold) | manual trigger, no gating |
| AUTO FIRE toggle | see `10-fire-control-safety.md` — real gating, read first |
| **AUTO-ACQUIRE toggle** | added this session — YOLO originates the lock automatically; see `09-old-new-raw-lock.md` |

## AUTO-ACQUIRE specifics

- Off by default.
- Button reads **"AUTO-ACQUIRE: N/A"** and stays disabled if the weights
  failed to load — check the console for the exact reason (missing file,
  `ultralytics`/`torch` not installed, etc.).
- Defaults to `../../drone_detection_module/weights/drone_yolov11x.pt`
  relative to `Old/main.py`'s location — confirm this path resolves
  correctly for your folder layout (see `02-fetch-weights.md`).
- Works with either PROFILE or BLOB lock mode.
- **Important**: an auto-acquired lock is exactly as eligible for AUTO
  FIRE as a manually-acquired one — this was a deliberate, explicitly
  confirmed decision, not an oversight. Read `10-fire-control-safety.md`
  before enabling both together.

Next: [Old vs New vs Raw vs Lock](09-old-new-raw-lock.md) to understand
exactly what AUTO-ACQUIRE is doing under the hood, or
[Fire control and safety](10-fire-control-safety.md) before arming
anything.
