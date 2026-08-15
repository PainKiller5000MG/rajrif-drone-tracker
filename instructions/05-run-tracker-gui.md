# 5. Running tracker_gui.py

The main control panel. This is the **fastest path to confirming real
hardware works**, since it auto-loads previously-saved, real-rig-measured
settings.

## Launch

```
cd 20RajRif
python tracker_gui.py
```
(Windows: `python tracker_gui.py`. Linux/Wayland: remember
`export QT_QPA_PLATFORM=xcb` first — see `01-prerequisites.md`.)

## What loads automatically

`tracker_gui_settings.json`, if present, is loaded on startup. A version
from a previously-successful real-rig session had these values already
proven working:

| Setting | Value | Why it matters |
|---|---|---|
| `mount_type` | ESP32 DC (BTS7960) | |
| `lock_mode` | Black blob | no-appearance-profile tracking, proven on this rig |
| `min_duty` / `min_duty_tilt` / `min_duty_tilt_neg` | 276 / 229 / 379 | measured stiction floors — see `12-technical-specs.md` |
| `kp_pan` / `kp_tilt` | 3.0 / 2.5 | tuned gains for this rig's actual motor speed |
| `pulse_below` | 0.2 | |

**Only the `port` field needs updating** to whatever your ESP32
enumerates as right now — saved port values go stale across machines/USB
ports. See `03-esp32-firmware-setup.md` to verify the board responds
before trusting any port number.

## In the panel

1. Set **Camera** to the confirmed index (`04-camera-setup.md`).
2. Set **Port** to your ESP32's actual port.
3. Confirm **Mount type** matches your hardware.
4. Click **Start**.

## In the video window

| Action | Control |
|---|---|
| Lock onto an object | drag a box around it |
| Lock onto an object (alternate) | single click — snaps to edges if found |
| Clear the lock | right-click, or `c` |
| Quit | `q` |

## Optional: YOLO auto-acquire

Off by default. Tick **"Auto-acquire lock from YOLO detections"** in the
panel to let YOLO originate the lock automatically instead of requiring a
manual drag/click — see `09-old-new-raw-lock.md` for exactly how this
works, and `10-fire-control-safety.md` before combining it with anything
fire-related.

## Verifying it actually worked

- The video window shows a colored box and a state label (`LOCKED`,
  `HIDDEN - predicting`, `LOST - searching`) once something is tracked.
- **The mount should physically move** when the locked target moves off
  center. If it doesn't:
  - Confirm the ESP32 responds to `V`/`P` outside this app first
    (`03-esp32-firmware-setup.md`).
  - Confirm the BTS7960 modules have their own motor power, not just USB.
  - Check the console for connection errors.

Next: [Running Old/main.py](06-run-old-main.md) for the fuller operator
station, or [Running detect_and_track.py](07-run-detect-and-track.md) for
the newest module.
