# 11. Troubleshooting

Every specific error hit this session, and its actual fix — not generic
advice.

## `RuntimeError: Numpy is not available`

Full traceback usually includes:
```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash...
```
`torch` was built against NumPy 1.x; NumPy 2.x is installed. Fix:
```
python3 -m pip install "numpy<2"
```
Don't touch `torch`/`ultralytics` — they're fine as-is.

## `Could not find the Qt platform plugin "wayland"`

Linux + Wayland desktop. `opencv-contrib-python`'s wheel bundles its own
Qt runtime, which only ships the `xcb` platform plugin — no wayland
plugin, regardless of what's installed system-wide. Fix:
```
export QT_QPA_PLATFORM=xcb
```
Works via XWayland, present on essentially every modern Linux desktop.

## `ModuleNotFoundError: No module named 'tracker_core'`

`detect_and_track.py` imports `tracker_core.py` from `../20RajRif` via a
relative path — `drone_tracking_module/` and `20RajRif/` must be sibling
folders under the same parent. Check your actual layout matches:
```
army/
    20RajRif/
    drone_tracking_module/
```

## `ModuleNotFoundError: No module named 'tkintermapview'`

Only affects `Old/main.py` (map view feature).
```
python3 -m pip install tkintermapview
```

## `FileNotFoundError: ... weights\drone_yolov11x.pt`

Model weights aren't in git (114MB > GitHub's 100MB limit) — only a
placeholder ships with a fresh clone. See `02-fetch-weights.md` and
actually download the file before running anything.

## `py` / `python3.11`: command not found (on what turned out to be Linux)

This repo's docs sometimes say `py -3.11 ...`, which is a **Windows-only**
launcher syntax. On Linux/macOS just use `python3` or `python`.

## Camera opens but the video window shows nothing / hangs on open

Set `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` **before** any
`cv2.VideoCapture` is constructed — without it, MSMF backend camera opens
can take up to ~40 seconds *per device*, which is easy to mistake for a
hang. Already set automatically inside `tracker_gui.py`, `Old/main.py`,
`detect_and_track.py`, and `probe_cameras.py` — but if you write any new
script that opens a camera, set this first.

## `AttributeError: module 'cv2' has no attribute 'TrackerCSRT_create'`

`opencv-python` (not the contrib build), or too new a version of
`opencv-contrib-python`, doesn't ship the legacy tracker API this repo's
`tracker_core.py` uses. Fix:
```
python3 -m pip uninstall opencv-python
python3 -m pip install "opencv-contrib-python==4.10.0.84"
```

## Auto-acquire / LOCK mode never confirms a lock

Watch the console — with logging enabled you should see one of:
- `Auto-acquire: candidate box WxH (X% of frame) - streak N/4` — it's
  seeing something, just hasn't hit the confirm bar yet. Normal if it's
  climbing.
- `Auto-acquire: watching, no detection above conf 0.80 yet` — model is
  running but nothing is clearing the confidence floor. Check lighting/
  distance/whether a drone is actually in frame.
- **Nothing at all, ever** — the detector likely never loaded. Check for
  a `could not load YOLO weights` line earlier in the console.

If streak count keeps resetting to 1 and never climbing on real footage,
this can be a genuine timing/tuning issue — a fast-moving drone at a slow
polling rate can measure low IoU between cycles even when it's genuinely
the same object. Try `--confirm-frames 2` or `--confirm-frames 3` as a
quick mitigation, or switch to `--mode raw` live with `m` for a
zero-confirmation-delay fallback.

## Mount doesn't move despite detection/lock apparently working

In order of likelihood:
1. **`--port` wasn't given, or was wrong.** Without a valid port, the
   script silently logs `[MOUNT] pan=... tilt=... (no serial port - not
   sent)` to console instead of erroring — easy to miss.
2. **BTS7960 modules have no separate motor power.** USB alone powers the
   ESP32's logic but not the motor driver stage — commands can be sent
   and acknowledged with zero physical movement.
3. **Nothing is actually locked/detected yet.** `mount.stop()` is called
   (correctly) when there's no aim point — this isn't a bug.
4. **Commanded duty is below your rig's real stiction floor.** Generic
   defaults may not match your actual hardware; see `12-technical-specs.md`
   for the measured values from a working session, and re-measure with
   `20RajRif/find_min_duty.py` if unsure.
5. **Wrong `--mount-type`.** Must match your actual hardware (`ESP32 DC
   (BTS7960)` vs `Stepper (x=/y=)`).

See `03-esp32-firmware-setup.md` for verifying the firmware itself
responds before suspecting the Python side at all.

## ESP32 firmware won't compile / doesn't work on my board

Confirm it's a genuine ESP32, not an Arduino Nano (classic ATmega328) —
the firmware uses ESP32-only PWM hardware (`ledc*` functions) that
doesn't exist on AVR chips at all, and GPIO numbers that don't exist on a
Nano's pinout. See `03-esp32-firmware-setup.md`'s opening section.

## "The weights must be 5GB, this can't be complete"

They're not — 109–115MB is the correct, complete size for a real YOLOv11x
model, confirmed against Ultralytics' own officially published weights.
If a folder *appears* to be several GB, that's almost always the Python
virtual environment (`torch`'s bundled CUDA libraries alone commonly run
2-4GB) sitting next to a much smaller model file, not the model itself.
Check with:
```
du -sh drone_detection_module/*
```

## A detected drone false-positives on something else (face, background)

Already mitigated by the confidence (`0.80`) and box-area (`10%`) filters
— validated against a real incident where a webcam face triggered at
confidence 0.25–0.68 with a box spanning 35–60% of frame, versus real
drone detections measured at 0.85–0.95 confidence and ≤2.6% of frame. If
you see a *new* false-positive pattern that slips through these filters,
that's real signal worth capturing (see `12-technical-specs.md`'s training
roadmap notes) — the underlying model is single-class and has no explicit
"not a drone" signal, so this can never be fully eliminated by filtering
alone.
