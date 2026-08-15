# rajrif-drone-tracker

Pan/tilt drone detection, tracking, and turret control for a DC-motor mount,
built for 15 RAJRIF & 222 FD WKSP. Private repository.

**Start here: [`instructions/`](instructions/README.md)** — full setup,
install, run, and troubleshooting docs, in order, from a bare machine to a
working hardware-connected tracker.

## Layout

```
20RajRif/                  the pan/tilt tracker: lock-on, mount control, firmware
    tracker_gui.py          main control panel (manual lock + optional YOLO auto-acquire)
    tracker_core.py          the identity lock: acquisition, search, re-lock
    blob_lock.py              alternate lock: no appearance profile, pure dark-blob
    mount.py                   serial link + ESP32/stepper mount backends
    yolo_autoacquire.py         optional YOLO-driven auto-lock, off by default
    Old/main.py                 fuller operator station: D-pad, FIRE/AUTO FIRE, IDMS, map
    test_tracking.py, test_integration.py, test_auto_acquire.py   test suites
    CLAUDE.md                   detailed engineering notes - read this first for any change

drone_detection_module/    standalone, read-only drone detection (no tracking/mount/fire)
    detect_drone.py          camera in, detections out, to screen and a log file

drone_tracking_module/     detect-and-follow, loading the YOLO model directly
    detect_and_track.py      RAW/LOCK auto-follow modes, manual jog, fire control
    probe_cameras.py          camera device enumerator
```

## Setup

Each module has its own dependencies; a single venv covering all of them
works fine:

```
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python3 -m pip install ultralytics torch "numpy<2" opencv-contrib-python pyserial tkintermapview
```

**Model weights are not in this repo** (114MB, exceeds GitHub's 100MB
limit). Fetch them from HuggingFace and place them in both modules'
`weights/` folders:

```
mkdir -p drone_detection_module/weights drone_tracking_module/weights
curl -L -o drone_detection_module/weights/drone_yolov11x.pt \
  "https://huggingface.co/doguilmak/Drone-Detection-YOLOv11x/resolve/main/weight/best.pt"
cp drone_detection_module/weights/drone_yolov11x.pt drone_tracking_module/weights/
```

**Linux + Wayland desktop**: also set, once per terminal session:
```
export QT_QPA_PLATFORM=xcb
```

Full setup/run instructions for every module are in
[`instructions/`](instructions/README.md). For engineering detail on the
tracker itself, see `20RajRif/README.md` and `20RajRif/CLAUDE.md`.

## Safety-relevant code

Two places in this repo can autonomously drive the mount or fire the
weapon. Read `20RajRif/CLAUDE.md` and the module docstrings before
touching either:

- **`20RajRif/tracker_gui.py`** and **`20RajRif/Old/main.py`** - optional
  YOLO auto-acquire (off by default), and `Old/main.py` has AUTO FIRE
  (ARM + confirmed-lock + centered-target gated).
- **`drone_tracking_module/detect_and_track.py`** - RAW/LOCK auto-follow,
  plus manual (`f`) and auto (`r`-armed) fire control with the same three
  gates.

`tracker_core.py`'s identity-lock is the safety-critical piece all of the
above builds on: it exists specifically to prevent locking clouds/wrong
objects, per the incident history in `20RajRif/CLAUDE.md`. Do not bypass
it or duplicate its logic elsewhere.
