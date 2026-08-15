# Instructions index

Everything needed to go from a bare machine to a running, hardware-connected
drone tracker. Read in order the first time; use as reference after that.

1. [Prerequisites](01-prerequisites.md) — Python, dependencies, OS-specific fixes
2. [Fetch the model weights](02-fetch-weights.md) — not stored in git, fetched separately
3. [ESP32 firmware setup](03-esp32-firmware-setup.md) — hardware, wiring, flashing, verification
4. [Camera setup](04-camera-setup.md) — finding the right device index
5. [Running tracker_gui.py](05-run-tracker-gui.md) — the main control panel
6. [Running Old/main.py](06-run-old-main.md) — the fuller operator station
7. [Running detect_and_track.py](07-run-detect-and-track.md) — the newest module: RAW/LOCK modes, fire control
8. [Running detect_drone.py](08-run-detect-drone.md) — standalone, read-only detection
9. [Old vs New vs Raw vs Lock](09-old-new-raw-lock.md) — what each tracking mechanism actually does
10. [Fire control and safety](10-fire-control-safety.md) — every gate, every key, read before arming anything
11. [Troubleshooting](11-troubleshooting.md) — every specific error hit this session and its fix
12. [Technical specifications](12-technical-specs.md) — full spec sheet, model/tracking/hardware/control-law numbers

## Fastest path to "something is moving"

If you just want to confirm the hardware works before touching any of the
newer YOLO features:

```
cd 20RajRif
python tracker_gui.py
```

This loads `tracker_gui_settings.json`, which already has the duty floors
and gains measured on the real rig (see `12-technical-specs.md`). Just
correct the **port** field to whatever your ESP32 enumerates as today, pick
your camera index, click Start, and drag a box around any object — Black
blob mode was the last combination confirmed working on real hardware.

Only after that works should you move on to the newer auto-acquire/YOLO
modules — they build on the same mount/tracker_core layer, so if this
baseline doesn't move the motors, nothing built on top of it will either.
