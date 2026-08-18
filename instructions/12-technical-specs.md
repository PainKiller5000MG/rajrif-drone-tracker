# 12. Technical specifications

## Detection model

| Spec | Value |
|---|---|
| Architecture | YOLOv11x (Ultralytics) |
| Parameters | 56.9M (source repo states 56,828,179 — fused-model count, matches) |
| Layers | 631 modules (190 fused) |
| GFLOPs | 194.4 |
| Classes | 1 (`drone`) — single-class model |
| Input resolution | 640×640 (training), configurable at inference |
| Weights file | `drone_yolov11x.pt`, 114,382,930 bytes (109.1 MB) |
| Source | `doguilmak/Drone-Detection-YOLOv11x` on HuggingFace |
| Precision | FP32 (CPU) / FP16 optional (CUDA) |

**Training (per source repo):** 1,012 training / 347 validation images,
YOLO-format annotations, 32 epochs, batch 16, AdamW, lr0=0.001, NVIDIA L4
GPU (Colab Pro+), PyTorch 2.6.0+cu124.
**mAP50: 0.905, mAP50-95: 0.546.** Precision 0.922, Recall 0.831
(validation, 369 instances). Reference inference speed: 8.9ms/image.

## Detection-layer safety filters

**As of 2026-08-16, the confidence and box-size filters are effectively
disabled by default, at explicit user request, understanding the
tradeoff.** History: 0.80 (initial) → 0.60 (2026-08-15, still above the
observed false-positive band) → 0.01 / 1.0 (2026-08-16, both filters
removed). This applies everywhere: `detect_drone.py`,
`detect_and_track.py` (both modes), `yolo_autoacquire.py` (feeds
auto-acquire in both `tracker_gui.py` and `Old/main.py`).

| Filter | Current default | Prior default | Rationale for prior value (no longer applied) |
|---|---|---|---|
| Confidence threshold | **0.01** (effectively unfiltered — Ultralytics refuses literal 0.0) | 0.60, originally 0.80 | observed false positives measured 0.25–0.68; real detections 0.85–0.95 |
| Max box area fraction | **1.0** (disabled) | 0.10 (10%) | real detections ≤2.6% of frame; false positives measured 35–60% |
| Confirm cycles (LOCK) | 4 consecutive (**unchanged**) | — | ~1s before a detection is trusted enough to originate a lock — the only remaining gate against a single noisy frame, not a confidence filter |
| Detector poll interval | 0.12s (~8Hz, unchanged) | — | decoupled from camera/control loop |
| IoU consistency threshold | 0.35 (unchanged) | — | tolerates real inter-frame drone motion (measured 0.0–0.94) |
| Streak decay | soft (-1 per miss, unchanged) | — | a single missed/inconsistent cycle doesn't erase prior progress |
| Staleness gap | 8× poll interval (≥1.0s, unchanged) | — | old streak reference discarded past this |

**Practical effect**: LOCK mode's auto-acquire will now originate a lock
on any detection YOLO returns at all — including clouds, birds, or
background noise repeated consistently for ~1s (the confirm-cycles gate).
Where AUTO FIRE is armed, such a lock is exactly as fire-eligible as a
real one (see `10-fire-control-safety.md`). To restore filtering, pass
`--conf-threshold 0.5` (or similar) and `--max-box-area-frac 0.1`
explicitly on the command line, or raise the equivalent fields in
`tracker_gui.py`'s Auto-acquire panel / `Old/main.py`'s
`AUTO_ACQUIRE_CONF`/`AUTO_ACQUIRE_MAX_AREA_FRAC` constants.

## Tracking (`tracker_core.py`)

| Component | Detail |
|---|---|
| Identity lock | `LockTracker` — appearance-memorizing, refuses mismatches |
| Alternate lock | `blob_lock.BlobTracker` — no appearance profile, dark-blob pursuit |
| Correlation tracker | CSRT (~25–65ms/frame, no GPU) or KCF (~4.6–9ms/frame, measured on real rig) |
| Lock states | `idle` → `lock` / `coast` / `search` |
| Re-lock confirmation | 3-frame default, velocity-aware drift tolerance |
| Distractor guard | blacklists look-alikes seen near the real target |
| Contrast veto | rejects near-featureless regions (blocks sky/cloud locking) |

| DEFAULTS (generic) | Value |
|---|---|
| track_thresh | 0.42 |
| relock_thresh | 0.60 |
| relock_frames | 3 |
| relock_margin | 0.08 |
| coast_s | 0.5 |
| max_jump_frac | 0.22 |
| RECENT_SIM_MIN | 0.42 |
| CONTRAST_VETO_MIN | 0.06 |

## Control law (`_axis_cmd`)

PD controller on normalized image error → signed speed fraction [-1, 1].

| Parameter | Generic default | **Measured, real rig (2026-08-09)** |
|---|---|---|
| kp_pan / kp_tilt | 1.6 / 1.4 | **3.0 / 2.5** |
| kd_pan / kd_tilt | 0.12 / 0.10 | 0.12 / 0.10 |
| deadzone_x / deadzone_y | 0.045 / 0.045 | 0.045 / 0.055 |
| pulse_below | — | 0.2 |
| D-term cap | 0.6× P-term | same |
| Coast scaling | 0.4× | same |

**Use the measured column for anything running on the actual rig** — the
generic defaults in `detect_and_track.py`/`build_default_cfg()` are
placeholders, not tuned. `tracker_gui.py` picks up the measured values
automatically via `tracker_gui_settings.json`.

## Mount / hardware

| Spec | Value |
|---|---|
| Motor driver | BTS7960 (×3: pan, tilt, trigger) |
| Board | genuine ESP32 (Arduino or MicroPython firmware — NOT Arduino Nano, see `03-esp32-firmware-setup.md`) |
| Control | signed PWM duty, open-loop (no position feedback) |
| PWM frequency | 5 kHz (20kHz barely moves these motors — measured) |
| Duty range | -1023 to 1023 |
| Serial protocol | `M,<pan>,<tilt>` / `S` / `F,1`/`F,0` / `P`→OK / `V`→version / legacy direction words |
| Keepalive | firmware watchdog stops motors after 600ms of serial silence; GUI resends ~5Hz |
| Alternate mount type | Stepper (absolute angle x=/y=), same interface |

**Measured duty floors and top speed, real rig, 2026-08-09 (1280×720):**

| Axis | Floor | Set to | Top speed |
|---|---|---|---|
| pan right | 240 | 276 | 155 px/s |
| pan left | 240 | 276 | 189 px/s |
| tilt up | 200 | 229 | 116 px/s |
| tilt down | 330 | 379 | 99 px/s |

Tilt-down needing a higher floor than tilt-up is real and reproducible
(confirmed not a swapped pin), not a tuning artifact.

**Confirmed working port/site values from a prior real-rig session** (do
not reuse literally — these are specific to that machine/session, re-check
against your own hardware):
- `tracker_gui_settings.json`: port `COM11`, camera index `0`, lock_mode
  `Black blob`
- `Old/mount_config.json`: `ESP32_PORT` `/dev/tty.usbserial-0001` (a
  different machine/OS than the COM11 session above — these two apps were
  evidently tested on different setups, not the same one)

## Fire control

See `10-fire-control-safety.md` for full detail. Summary:

| Mode | Gating |
|---|---|
| Manual (`f` / FIRE button) | none — operator judgment only |
| Auto (armed) | ALL of: armed + `mode=="lock"`/PROFILE + `state=="lock"` + centered (zone<30%) |
| Panic stop (`space`) | unconditional disarm + fire-off + movement-stop |
| Fail-safe | `mount.fire(False)` guaranteed on every exit path |

## Software modules

| Module | Purpose | Fire control | Auto-follow |
|---|---|---|---|
| `20RajRif/tracker_gui.py` | main control panel | no | manual lock + optional YOLO auto-acquire |
| `20RajRif/Old/main.py` | full operator station | yes (pre-existing + auto-acquire eligible) | manual lock + optional YOLO auto-acquire |
| `drone_detection_module/detect_drone.py` | standalone read-only detection | no | no |
| `drone_tracking_module/detect_and_track.py` | detect-and-follow, direct model load | yes (added this session) | RAW or LOCK, live-switchable |
| `20RajRif/yolo_autoacquire.py` | shared threaded YOLO detector/filter | — | used by all three tracker-integrated apps |

## Dependencies

```
ultralytics, torch, numpy<2, opencv-contrib-python, pyserial,
tkinter (system package), tkintermapview (Old/main.py only)
```

## Test coverage

| Suite | Result |
|---|---|
| `test_tracking.py` | 13/14 (the one failure is a CPU-only-hardware speed check, not a logic defect) |
| `test_integration.py` | 3/3 |
| `test_auto_acquire.py` | 8/8 |

## Repository

- Private GitHub repo, weights excluded (114MB > GitHub's 100MB hard
  limit) — see `02-fetch-weights.md`
- `.gitignore` also excludes: venvs, `__pycache__`, runtime logs,
  `Old/mount_config.json` (real site coordinates)
