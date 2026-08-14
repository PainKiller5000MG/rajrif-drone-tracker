# drone_tracking_module

Detect-and-follow, loading the YOLO weights directly. A clone of
`drone_detection_module` with mount-following added on top.
`detect_drone.py` in this folder is untouched — still the original
read-only detection script.

`detect_and_track.py` is the new piece: it can actually move the mount.
`tracker_core.py`, `mount.py`, and the PD control law (`_axis_cmd`) are
imported from `../20RajRif` at runtime, not copied — this folder must stay
a sibling of `20RajRif` (same parent folder) for that to work.

## 1. Install (one-time)

Use a virtual environment, not your system/global Python — avoids version
conflicts with other projects on the same machine:

```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Then install everything:

```
python3 -m pip install ultralytics torch "numpy<2" opencv-contrib-python pyserial
```

Notes:
- **`"numpy<2"`** — pinned deliberately. Some `torch` builds are compiled
  against NumPy 1.x and crash with NumPy 2.x installed
  (`RuntimeError: Numpy is not available`). If you ever see that error,
  re-run `python3 -m pip install "numpy<2"`.
- **`opencv-contrib-python`**, not plain `opencv-python` — the CSRT tracker
  this whole system uses only exists in the "contrib" build. If
  `opencv-python` is already installed, remove it first:
  ```
  python3 -m pip uninstall opencv-python
  python3 -m pip install opencv-contrib-python
  ```
- **tkinter is required**, even though this script has no GUI panel of its
  own — it imports `_axis_cmd` from `tracker_gui.py`, which imports
  `tkinter` at the top of the file. On Linux this usually means installing
  a system package (e.g. Arch: `sudo pacman -S tk`; Debian/Ubuntu:
  `sudo apt install python3-tk`). macOS/Windows Python installers normally
  include it already.
- **Linux + Wayland desktop only**: the video window may fail to open with
  a `Could not find the Qt platform plugin "wayland"` error. Fix by setting,
  once per terminal session:
  ```
  export QT_QPA_PLATFORM=xcb
  ```

## 2. Folder layout required

```
army/
    20RajRif/                  <- must exist, unmodified
    drone_tracking_module/     <- this folder
        weights/drone_yolov11x.pt
        detect_and_track.py
        detect_drone.py
```

Both folders must sit next to each other. If you only copied
`drone_tracking_module` to a new machine, copy `20RajRif` there too.

## 3. Run

Basic (prompts you for the camera device index, no mount control - safe
for a first test):

```
cd drone_tracking_module
python3 detect_and_track.py --weights weights/drone_yolov11x.pt
```

Full mount control on the real rig:

```
python3 detect_and_track.py \
    --weights weights/drone_yolov11x.pt \
    --port /dev/ttyUSB0 \
    --mount-type "ESP32 DC (BTS7960)" \
    --mode lock
```

(Windows: `--port COM3` or similar.)

Test against a recorded clip with no mount, before touching real hardware:

```
python3 detect_and_track.py \
    --weights weights/drone_yolov11x.pt \
    --source ../20RajRif/test_out/drone_flight.mp4
```

## 4. While it's running

| Key | Action |
|---|---|
| `q` | Quit |
| `m` | Switch between LOCK and RAW mode, live |
| `c` | Clear the current lock (LOCK mode only) |
| `w` / `s` | Manual tilt up / down |
| `a` / `d` | Manual pan left / right |
| `space` | Manual stop |

The control list is also drawn on the video window itself, bottom-left,
so it doesn't need to be memorised.

**Manual jog (w/a/s/d) always overrides auto-follow** for that frame, same
priority rule as the main tracker GUI's jog controls - useful to nudge the
mount by hand at any time, in either mode, without switching anything off
first. It's driven by keyboard polling rather than true key-up/down events,
so holding a key down can feel a little pulsed rather than perfectly smooth
(a known limitation, not a bug) - unlike the D-pad buttons in `Old/main.py`,
which use real press/release events. Requires the video window to have
focus, and isn't available with `--no-display`.

**LOCK mode** (default, stable): YOLO only originates a fresh lock after
~4 consistent detections; `tracker_core.py`'s identity-lock then owns
tracking, same as the main tracker GUI.

**RAW mode**: every frame, whatever YOLO detects directly drives the
mount - no persistence, no identity check, more responsive but also more
prone to a spurious detection nudging the mount. Switch to this live with
`m` if LOCK mode isn't catching fast enough.

No fire control, no ARM, no trigger of any kind - detection and following
only.

## 5. Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--weights` | `weights/drone_yolov11x.pt` | Path to YOLO weights |
| `--source` | (prompts you) | Camera index or video file path |
| `--mode` | `lock` | Starting mode: `lock` or `raw` |
| `--conf-threshold` | `0.80` | Confidence floor for any detection to count |
| `--max-box-area-frac` | `0.10` | Reject detections covering more than this fraction of frame |
| `--confirm-frames` | `4` | LOCK mode: consistent cycles needed before auto-locking |
| `--port` | `None` | Mount serial port; omit to log commands instead of sending them |
| `--mount-type` | `ESP32 DC (BTS7960)` | Or `Stepper (x=/y=)` |
| `--no-display` | off | Headless, no video window (for automated testing) |
| `--no-half` | off | Disable FP16 (use if you hit GPU memory errors) |

## 6. Troubleshooting

- **`RuntimeError: Numpy is not available`** → `python3 -m pip install "numpy<2"`
- **`Could not find the Qt platform plugin "wayland"`** → `export QT_QPA_PLATFORM=xcb`
- **`ModuleNotFoundError: No module named 'tkinter'`** → install your OS's tkinter package (see Install section)
- **Detector loads but never confirms a lock (LOCK mode)** → try
  `--confirm-frames 2` or `--confirm-frames 3`, or switch to `--mode raw`
  live with `m`. Watch the console - it logs every detection cycle and
  streak count so you can see exactly what's happening.
- **Weights warning about 6GB VRAM** → not a concern on an 8GB+ GPU; type
  `y` to continue.
