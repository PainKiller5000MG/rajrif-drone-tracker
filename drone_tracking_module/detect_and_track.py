#!/usr/bin/env python3
"""
Standalone drone detect-and-follow, loading the YOLO weights directly.

This is a clone of drone_detection_module with tracking/mount-follow added
on top (detect_drone.py in this folder is untouched, left as the read-only
detection-only script it always was). Everything here originates from the
YOLO model directly - there is no manual click/drag acquisition, unlike
tracker_gui.py / Old/main.py.

Two switchable follow modes, live-toggle with 'm':

  RAW   Every frame, whatever YOLO detects (after the confidence + max-box-
        area filters) directly drives the mount toward it. No persistence,
        no identity check. This is deliberately close to the pre-2026-08-07
        Old/main_pre_lock_backup.py design that was removed from both
        tracker rigs for repeatedly locking clouds and losing banking
        drones (see 20RajRif/CLAUDE.md, "Known gaps and failures"). It is
        offered here because it was explicitly asked for, not because the
        failure mode has been fixed - expect the same risk.

  LOCK  YOLO only originates a fresh lock (same filters, plus a short run
        of consistent detections before it's trusted - see
        yolo_autoacquire.py), then tracker_core.py's identity-lock state
        machine owns tracking exactly as it does in tracker_gui.py and
        Old/main.py. This is the stable mode; default on start.

tracker_core.py, mount.py, and the PD control law (_axis_cmd) are imported
from ../20RajRif, never copied - see 20RajRif/CLAUDE.md on why duplicating
those files is exactly what NOT to do.

Fire control (added 2026-08-14, explicitly requested and confirmed - this
reverses the "no fire trigger of any kind" scope every other module this
session was built under):

  Manual FIRE  'f' toggles it directly, any time, either mode - pure
               operator judgement, no gating at all, same as a manual
               trigger has always worked elsewhere in this repo.

  AUTO FIRE    'r' arms/disarms it. While armed, it only ever fires when
               ALL of the following hold, checked every frame:
                 - mode == "lock" (never in RAW mode - RAW has no identity
                   check at all, the same reason Old/main.py disables AUTO
                   FIRE in BLOB mode: "locked" there doesn't mean "this is
                   confirmed the same target")
                 - tracker.state == "lock" (never coast/search - never
                   fires on a prediction or a search guess)
                 - the target is centered (zone < 30, same formula and
                   threshold as Old/main.py's legacy condition)
               These are the same three gates Old/main.py's AUTO FIRE
               already uses; nothing about that proven design was changed,
               only re-implemented here for this script's own state.

Both fire paths always call mount.fire(False) on any exit path (quit,
Ctrl+C, crash) - see the finally block.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
RAJRIF_DIR = os.path.join(HERE, "..", "20RajRif")
sys.path.insert(0, os.path.abspath(RAJRIF_DIR))

# Same env-var-before-cv2-import requirement as tracker_gui.py/Old/main.py.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")


def parse_args():
    p = argparse.ArgumentParser(
        description="Detect and follow a drone directly from the YOLO model.")
    p.add_argument("--weights", default=os.path.join(HERE, "weights", "drone_yolov11x.pt"),
                    help="Path to YOLO weights (default: this folder's cloned weights).")
    p.add_argument("--source", default=None,
                    help="Camera device index or a video file path. If omitted, "
                         "you'll be prompted to confirm the device index.")
    p.add_argument("--mode", choices=["raw", "lock"], default="lock",
                    help="Starting follow mode (default: lock, the stable one). "
                         "Press 'm' at runtime to switch live.")
    p.add_argument("--conf-threshold", type=float, default=0.01,
                    help="Confidence floor for a detection to be used at all (default 0.01 - "
                         "effectively unfiltered; 0.0 itself is refused by Ultralytics). "
                         "WARNING: this feeds auto-acquire/LOCK-mode targeting and, if "
                         "AUTO FIRE is armed, fire eligibility - at this setting expect "
                         "false locks. See instructions/10-fire-control-safety.md.")
    p.add_argument("--max-box-area-frac", type=float, default=1.0,
                    help="Reject detections covering more than this fraction of the "
                         "frame (default 1.0 - disabled). Set below 1.0 to guard "
                         "against frame-spanning false positives in both modes.")
    p.add_argument("--confirm-frames", type=int, default=4,
                    help="LOCK mode only: consecutive consistent detector cycles "
                         "needed before YOLO originates a fresh lock (default 5).")
    p.add_argument("--half", dest="half", action="store_true", default=True)
    p.add_argument("--no-half", dest="half", action="store_false")
    p.add_argument("--device", default=None, help="Torch device, e.g. 'cuda:0' or 'cpu'.")
    p.add_argument("--port", default="None", help="Mount serial port, or 'None' for no serial.")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--mount-type", default="ESP32 DC (BTS7960)",
                    choices=["ESP32 DC (BTS7960)", "Stepper (x=/y=)"])
    p.add_argument("--frame-width", type=int, default=1280)
    p.add_argument("--frame-height", type=int, default=720)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--log-path", default=None)
    return p.parse_args()


def confirm_device_index():
    print("\nNo --source was given.")
    print("Enter the OpenCV VideoCapture device index (0, 1, 2, ...) or a video "
          "file path. Device indices are OS/setup-dependent - not guessed.")
    while True:
        raw = input("Source: ").strip()
        if raw == "":
            continue
        try:
            return int(raw)
        except ValueError:
            return raw


def check_weights_size(weights_path):
    """Same check as drone_detection_module/detect_drone.py, duplicated here
    (not imported) so this script stays a single self-contained file."""
    name = Path(weights_path).stem.lower()
    looks_heavy = any(m in name for m in ("11x", "yolo11x", "v11x", "-x", "_x"))
    size_mb = os.path.getsize(weights_path) / (1024 * 1024) if os.path.exists(weights_path) else None
    if size_mb is not None and size_mb > 80:
        looks_heavy = True
    if looks_heavy:
        print("\n" + "=" * 70)
        print("WARNING: These look like YOLOv11x (extra-large) weights"
              + (" (%.1f MB)" % size_mb if size_mb else ""))
        print("Heavy for small GPUs - confirm before proceeding.")
        print("=" * 70)
        if input("Continue anyway? [y/N]: ").strip().lower() != "y":
            print("Aborting.")
            sys.exit(1)


def build_default_cfg(args):
    """Minimal cfg dict for tracker_core.LockTracker and mount.make_mount -
    same keys tracker_gui.py's Start button builds, defaults only."""
    return {
        "tracker": "CSRT",
        "give_up_s": 0.0,
        "track_thresh": 0.42, "relock_thresh": 0.60, "relock_frames": 3,
        "relock_margin": 0.08, "relock_color_min": 0.35, "relock_struct_min": 0.34,
        "distractor_guard": True, "coast_s": 0.5, "max_jump_frac": 0.22,
        "aim_smooth": 0.45,
        "min_duty": 320, "max_duty": 1023, "min_duty_neg": 320, "max_duty_tilt": 1023,
        "min_duty_tilt": 320, "min_duty_tilt_neg": 320, "pulse_below": 0.30,
        "max_speed_x": 12.0, "max_speed_y": 20.0, "scan_min": -90, "scan_max": 90,
        "tilt_center": 0, "tilt_min": -45, "tilt_max": 45,
        "kp_pan": 1.6, "kp_tilt": 1.4, "kd_pan": 0.12, "kd_tilt": 0.10,
        "deadzone_x": 0.045, "deadzone_y": 0.045, "pan_dir": 1, "tilt_dir": 1,
        "enable_y": True,
        "jog_speed": 0.55,  # same default as tracker_gui.py's jog speed
    }


# Manual jog keys -> (pan, tilt) direction, matching the D-pad convention
# used elsewhere in this repo: +pan = right, +tilt = up.
JOG_KEYS = {
    ord('a'): (-1.0, 0.0),
    ord('d'): (1.0, 0.0),
    ord('w'): (0.0, 1.0),
    ord('s'): (0.0, -1.0),
}

MENU_TEXT = [
    "MANUAL: w/a/s/d = tilt up/pan left/tilt down/pan right, space = stop",
    "MODE: m = switch RAW/LOCK, c = clear a LOCK",
    "FIRE: f = manual fire toggle, r = arm/disarm AUTO FIRE",
    "q = quit",
]

FIRE_ZONE_MAX = 30.0  # same threshold as Old/main.py's legacy centred/zone<30 condition


def fire_zone_pct(aim, half_w, half_h):
    """% distance of aim from frame centre, 0=dead centre - same formula as
    Old/main.py's get_zone(), reimplemented here (trivial/stateless, no
    tuning risk, so not worth adding an import dependency on Old/ for)."""
    cx, cy = aim
    dist = ((cx - half_w) ** 2 + (cy - half_h) ** 2) ** 0.5
    max_dist = (half_w ** 2 + half_h ** 2) ** 0.5
    return (dist / max_dist * 100.0) if max_dist else 0.0


def main():
    args = parse_args()
    check_weights_size(args.weights)

    source = args.source if args.source is not None else confirm_device_index()
    if isinstance(source, str):
        try:
            source = int(source)
        except ValueError:
            pass

    import cv2
    import torch
    from ultralytics import YOLO

    from tracker_core import LockTracker, clip_box
    from mount import SerialLink, make_mount
    from tracker_gui import _axis_cmd
    from yolo_autoacquire import AutoAcquireDetector

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    use_half = args.half and device.startswith("cuda")
    print("Weights: %s | device: %s | half: %s" % (args.weights, device, use_half))

    print("Loading YOLO model directly...")
    model = YOLO(args.weights)
    class_names = model.names
    print("Model loaded.")

    cfg = build_default_cfg(args)
    link = SerialLink(args.port, args.baud, log=print) if args.port != "None" else None
    mount = make_mount(args.mount_type, link, cfg) if link is not None else None
    if mount is None:
        print("No serial port given - mount commands will be logged, not sent.")

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)
    if not cap.isOpened():
        print("ERROR: could not open source %r" % (source,))
        sys.exit(1)
    ret, frame = cap.read()
    if not ret:
        print("ERROR: source opened but returned no frames")
        sys.exit(1)
    frame_h, frame_w = frame.shape[:2]
    half_w, half_h = frame_w / 2.0, frame_h / 2.0
    print("Source %r ready at %dx%d" % (source, frame_w, frame_h))

    # ---- LOCK-mode machinery ----
    tracker = LockTracker(cfg, log=lambda m: print("[LOCK]", m))
    detector = AutoAcquireDetector(
        args.weights, conf_thresh=args.conf_threshold,
        max_box_area_frac=args.max_box_area_frac,
        confirm_frames=args.confirm_frames,
        log=lambda m: print("[AUTO-ACQUIRE]", m))
    if detector.ready:
        detector.start()

    # ---- logging ----
    log_path = args.log_path or str(Path(HERE) / ("track_log_%s.csv" %
                                                    datetime.now().strftime("%Y%m%d_%H%M%S")))
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["timestamp_utc", "mode", "class", "confidence",
                          "x1", "y1", "x2", "y2", "pan_cmd", "tilt_cmd",
                          "firing", "fire_reason"])
    print("Log file: %s" % log_path)

    mode = args.mode
    prev_err = [0.0, 0.0]
    in_dz = [False, False]
    last_t = time.time()
    last_raw_seen_t = 0.0
    last_key = -1  # updated by cv2.waitKey each frame; -1 = nothing pressed
    manual_fire = False   # 'f' toggle - operator's own judgement, no gating
    armed = False          # 'r' toggle - arms AUTO FIRE; starts disarmed
    fire_active = False    # current physical fire state (edge-triggered logging)
    RAW_STALE_S = 0.5  # no detection for this long -> stop, matches the old
                        # pre-port app's stale-out window

    print("Mode: %s" % mode.upper())
    for line in MENU_TEXT:
        print("  " + line)
    if args.no_display:
        print("(--no-display: manual jog needs the video window focused, "
              "so it's unavailable in headless mode)")

    frame_count = 0
    try:
        while True:
            now = time.time()
            dt = max(1e-3, min(now - last_t, 0.25))
            last_t = now

            ret, frame = cap.read()
            if not ret:
                print("End of stream / frame read error.")
                break
            frame_count += 1

            if mode == "lock":
                detector.submit_frame(frame)
                if tracker.state == "idle":
                    auto_box = detector.get_confirmed_box()
                    if auto_box is not None:
                        print("[AUTO-ACQUIRE] locking on YOLO detection (%dx%d)" %
                              (auto_box[2], auto_box[3]))
                        tracker.select(frame, auto_box)

                st = tracker.update(frame, (), now=now)
                aim = st.aim if st.aim is not None and st.state in ("lock", "coast") else None
                state_txt = st.state
                cls_name, conf = "drone", st.score
                box_for_log = st.box
            else:
                # RAW: run YOLO directly this frame, no persistence at all -
                # see module docstring for why this mode carries real risk.
                results = model.predict(frame, conf=args.conf_threshold,
                                         half=use_half, device=device, verbose=False)
                best_box, best_area, best_conf, best_cls = None, 0, 0.0, "drone"
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                        w, h = x2 - x1, y2 - y1
                        area = w * h
                        if frame_w * frame_h and (area / (frame_w * frame_h)) > args.max_box_area_frac:
                            continue
                        if area > best_area:
                            best_area = area
                            best_box = (x1, y1, w, h)
                            best_conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            best_cls = class_names.get(cls_id, str(cls_id))
                if best_box is not None:
                    last_raw_seen_t = now
                    aim = (best_box[0] + best_box[2] / 2.0, best_box[1] + best_box[3] / 2.0)
                    state_txt = "raw-detect"
                elif now - last_raw_seen_t < RAW_STALE_S:
                    aim = None  # coast briefly, same stale window as the old app
                    state_txt = "raw-stale"
                else:
                    aim = None
                    state_txt = "raw-none"
                cls_name, conf = best_cls, best_conf
                box_for_log = best_box

            pan_cmd = tilt_cmd = 0.0
            jog = JOG_KEYS.get(last_key)
            if last_key == ord(' '):
                # Explicit manual stop: overrides auto-follow same as a jog
                # does, AND is a full fire kill-switch - forces manual fire
                # off and disarms AUTO FIRE unconditionally, regardless of
                # their current state. One key, no ambiguity, always safe.
                prev_err = [0.0, 0.0]
                state_txt = "manual-stop"
                if manual_fire or armed:
                    print("[FIRE] SPACE pressed - forcing manual fire off and disarming")
                manual_fire = False
                armed = False
                if mount is not None:
                    mount.stop()
            elif jog is not None:
                # Manual jog always takes priority over auto-follow, same
                # convention as tracker_gui.py's jog-overrides-tracking rule.
                # NOTE: driven by cv2.waitKey polling, not real key-up/down
                # events (unlike the D-pad buttons elsewhere in this repo),
                # so held-key movement can feel a bit pulsed rather than
                # perfectly smooth - a known limitation of this approach.
                pan_cmd = jog[0] * cfg["jog_speed"]
                tilt_cmd = jog[1] * cfg["jog_speed"]
                prev_err = [0.0, 0.0]
                state_txt = "manual-jog"
                if mount is not None:
                    mount.drive(pan_cmd, tilt_cmd, dt)
                else:
                    print("[MOUNT] pan=%.2f tilt=%.2f (no serial port - not sent)" %
                          (pan_cmd, tilt_cmd))
            elif aim is not None:
                err_x = (aim[0] - half_w) / half_w
                err_y = (half_h - aim[1]) / half_h
                pan_cmd = _axis_cmd(err_x, prev_err, 0, in_dz, dt,
                                     cfg["kp_pan"], cfg["kd_pan"], cfg["deadzone_x"], cfg["pan_dir"])
                tilt_cmd = _axis_cmd(err_y, prev_err, 1, in_dz, dt,
                                      cfg["kp_tilt"], cfg["kd_tilt"], cfg["deadzone_y"], cfg["tilt_dir"])
                if state_txt in ("coast", "raw-stale"):
                    pan_cmd *= 0.4
                    tilt_cmd *= 0.4
                if mount is not None:
                    mount.drive(pan_cmd, tilt_cmd, dt)
                else:
                    print("[MOUNT] pan=%.2f tilt=%.2f (no serial port - not sent)" %
                          (pan_cmd, tilt_cmd))
            else:
                prev_err = [0.0, 0.0]
                if mount is not None:
                    mount.stop()

            # ---- FIRE (added 2026-08-14, explicitly requested) ----
            # Manual: pure operator judgement via 'f', no gating whatsoever.
            # Auto: only while armed ('r'), and only when ALL three of the
            # same gates Old/main.py's AUTO FIRE already uses hold - see the
            # module docstring for the reasoning behind each one.
            auto_gate_mode = (mode == "lock")
            auto_gate_state = (mode == "lock" and tracker.state == "lock")
            zone = fire_zone_pct(aim, half_w, half_h) if aim is not None else None
            auto_gate_centered = (zone is not None and zone < FIRE_ZONE_MAX)
            auto_should_fire = armed and auto_gate_mode and auto_gate_state and auto_gate_centered

            should_fire = manual_fire or auto_should_fire
            if should_fire and not fire_active:
                reason = "manual" if manual_fire else "auto (armed, locked, centered zone=%.0f%%)" % zone
                print("[FIRE] STARTING - %s" % reason)
            elif fire_active and not should_fire:
                print("[FIRE] STOPPING")
            fire_active = should_fire
            fire_reason = "manual" if manual_fire else ("auto" if auto_should_fire else "")
            if mount is not None:
                mount.fire(should_fire)
            elif should_fire:
                print("[FIRE] would fire now (%s) - no serial port, not sent" % fire_reason)

            # Log whenever there's a detection to record, OR whenever fire is
            # active - manual fire has no detection gating at all, so a
            # firing frame with no box must still land in the audit trail.
            if box_for_log is not None or fire_active:
                if box_for_log is not None:
                    x1, y1, w, h = box_for_log
                    box_fields = [int(x1), int(y1), int(x1 + w), int(y1 + h)]
                else:
                    box_fields = ["", "", "", ""]
                log_writer.writerow([datetime.now(timezone.utc).isoformat(), mode,
                                      cls_name if box_for_log is not None else "",
                                      "%.4f" % conf if box_for_log is not None else "",
                                      *box_fields,
                                      "%.3f" % pan_cmd, "%.3f" % tilt_cmd,
                                      int(fire_active), fire_reason])
                log_file.flush()

            if not args.no_display:
                if box_for_log is not None:
                    x1, y1, w, h = [int(v) for v in box_for_log]
                    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                    cv2.putText(frame, "%s %.2f" % (cls_name, conf), (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "MODE: %s  STATE: %s" % (mode.upper(), state_txt),
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                fire_col = (0, 0, 255) if fire_active else ((0, 165, 255) if armed else (180, 180, 180))
                cv2.putText(frame, "FIRE: %s  ARM: %s%s" %
                            ("ON" if fire_active else "off",
                             "ARMED" if armed else "safe",
                             " (manual)" if manual_fire else ""),
                            (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fire_col, 2)
                for i, line in enumerate(MENU_TEXT):
                    cv2.putText(frame, line, (8, frame_h - 12 - (len(MENU_TEXT) - 1 - i) * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                cv2.imshow("Detect and Track", frame)
                key = cv2.waitKey(1) & 0xFF
                last_key = key
                if key == ord('q'):
                    break
                if key == ord('m'):
                    mode = "raw" if mode == "lock" else "lock"
                    tracker.clear()
                    prev_err = [0.0, 0.0]
                    if mount is not None:
                        mount.stop()
                    print("Switched to mode: %s" % mode.upper())
                if key == ord('c') and mode == "lock":
                    tracker.clear()
                    print("Lock cleared.")
                if key == ord('f'):
                    manual_fire = not manual_fire
                    print("[FIRE] Manual fire %s" % ("ON" if manual_fire else "OFF"))
                if key == ord('r'):
                    armed = not armed
                    print("[FIRE] AUTO FIRE %s" % ("ARMED" if armed else "DISARMED"))
            else:
                last_key = -1  # no window, no keyboard input possible

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # mount.stop() only affects pan/tilt - fire is a separate command
        # and must be explicitly stopped here too, on every exit path
        # (quit, Ctrl+C, or an unexpected crash falling through to here).
        if mount is not None:
            print("[FIRE] Ensuring fire is off before exit.")
            try:
                mount.fire(False)
            except Exception:
                pass
            mount.stop()
        if link is not None:
            link.close()
        detector.stop()
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        log_file.close()
        print("Processed %d frames. Log: %s" % (frame_count, log_path))


if __name__ == "__main__":
    main()
