# 1. Prerequisites

## Python environment

Use a virtual environment, not the system/global Python — avoids version
conflicts with anything else installed on the machine.

```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

## Install everything

```
python3 -m pip install ultralytics torch "numpy<2" opencv-contrib-python pyserial tkintermapview
```

### Why each pin/exclusion matters

- **`"numpy<2"`** — some `torch` builds are compiled against NumPy 1.x and
  crash with NumPy 2.x installed:
  ```
  RuntimeError: Numpy is not available
  ```
  If you ever see that error, re-run `python3 -m pip install "numpy<2"`.

- **`opencv-contrib-python`, never plain `opencv-python`** — the CSRT
  correlation tracker this whole system depends on only exists in the
  "contrib" build. If `opencv-python` is already installed, remove it
  first — having both installed causes real conflicts:
  ```
  python3 -m pip uninstall opencv-python
  python3 -m pip install opencv-contrib-python
  ```

- **`tkinter` is required even for headless-feeling scripts.**
  `detect_and_track.py` has no GUI panel of its own, but it imports
  `_axis_cmd` from `tracker_gui.py`, which imports `tkinter` at module
  level — so it's a hard dependency everywhere in this repo, not just the
  GUI apps.
  - Linux: install your distro's tk package (Arch: `sudo pacman -S tk`;
    Debian/Ubuntu: `sudo apt install python3-tk`).
  - macOS/Windows: normally bundled with the standard Python installer
    already.

- **`tkintermapview`** — only needed for `Old/main.py`'s map view. Skip it
  if you're only using `tracker_gui.py` or `detect_and_track.py`.

## Linux + Wayland desktop only

OpenCV's video window can fail to open with:
```
qt.qpa.plugin: Could not find the Qt platform plugin "wayland"
```
This happens because `opencv-contrib-python`'s wheel bundles its own Qt
runtime, and that bundle only ships the `xcb` (X11) platform plugin — no
wayland plugin — regardless of what's installed system-wide. Fix, once per
terminal session:
```
export QT_QPA_PLATFORM=xcb
```
Works fine under Wayland via XWayland, present on virtually every modern
Linux desktop.

## Verifying the install

```
python3 -c "import cv2, torch, serial; from ultralytics import YOLO; print('all core deps import OK')"
```

If that line prints cleanly, move on to
[fetching the model weights](02-fetch-weights.md).
