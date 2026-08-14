"""Pan/tilt tracker - control panel.

Video lives in an OpenCV window; this panel picks the mount, port, camera and
gains. Heavy imports (cv2, serial) happen on Start so the window opens
instantly.

The operator's drag/click is still the primary way a target is acquired
(removed 2026-08-07, with Scan360's, then reintroduced as an optional,
off-by-default assist on 2026-08-13 for a specific demo - see
yolo_autoacquire.py and the "Auto-acquire (YOLO)" panel). When enabled it
only ever proposes a box to the same tracker.select() a manual drag calls;
tracker_core.py's identity/state machine is unchanged and untouched.

In the video window:
    drag left button   lock onto whatever is inside the box
    single left click  lock onto the object under the cursor - its own edges
                       if they can be found, else a default-sized box
    right click / c    clear the lock
    q                  quit
"""

import json
import os
import queue
import threading
import time

# Without this, opening the camera through the MSMF backend takes ~40 s
# (OpenCV hardware-transform negotiation bug). Must be set before cv2
# creates the VideoCapture.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import tkinter as tk
from tkinter import ttk

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(HERE, "tracker_gui_settings.json")

BAUD_DEFAULT = 115200
MOUNT_TYPES = ["ESP32 DC (BTS7960)", "Stepper (x=/y=)"]
WINDOW_NAME = "RajRif Tracker"
REASON_LOG_S = 1.0      # how often the "why not re-locking" line may repeat

# Bumped when a saved key changes meaning. v1 gains were degrees-per-frame and
# v1 deadzones were degrees; the same names now hold a speed fraction and a
# fraction of the half-frame. Loading a v1 file blind would leave the mount
# with a deadzone of 1.0 - i.e. it would never move - so those keys are
# dropped back to defaults instead.
SETTINGS_VERSION = 2
RESCALED_IN_V2 = {"kp_pan", "kp_tilt", "deadzone_x", "deadzone_y"}


# ================== TRACKER THREAD ==================

def run_tracker(cfg, stop_event, ui, log):
    import cv2

    from mount import SerialLink, make_mount
    from tracker_core import (LockTracker, clip_box, estimate_object_box,
                              snap_selection)

    link = SerialLink(cfg["port"], cfg["baud"], log=log)
    mount = make_mount(cfg["mount_type"], link, cfg)
    log("Mount: %s" % cfg["mount_type"])

    # ---- camera ----
    log("Opening camera index %s ..." % cfg["cam"])
    # MSMF is essential for 1280x720@30: DSHOW only offers YUY2 at this
    # resolution (~5 fps over USB) and ignores MJPG requests.
    cap = cv2.VideoCapture(cfg["cam"], cv2.CAP_MSMF)
    if not cap.isOpened():
        log("MSMF failed, retrying with the default backend...")
        cap = cv2.VideoCapture(cfg["cam"])
    if not cap.isOpened():
        log("ERROR: could not open camera")
        link.close()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["frame_w"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["frame_h"])
    cap.set(cv2.CAP_PROP_FPS, 30)

    ret, frame = cap.read()
    if not ret:
        log("ERROR: camera opened but returns no frames")
        cap.release()
        link.close()
        return
    frame_h, frame_w = frame.shape[:2]
    log("Camera %dx%d ready." % (frame_w, frame_h))

    # ---- tracker ----
    # Two locks, operator's choice. Profile memorises what the object looks
    # like and refuses mismatches - safe, slower to re-take a changed target.
    # Black blob knows nothing but "the dark blob against the sky where the
    # target should be" - catches instantly, and will follow a bird if the
    # bird is the best blob. The trigger is manual behind ARM either way.
    if cfg.get("lock_mode") == "Black blob":
        from blob_lock import BlobTracker
        tracker = BlobTracker(cfg, log=log)
        log("Lock mode: BLACK BLOB - no appearance profile, pure dark-blob "
            "pursuit.")
    else:
        tracker = LockTracker(cfg, log=log)

    # ---- auto-acquire (optional, off by default) ----
    # Runs YOLO on its own thread, decoupled from this loop's cadence, and
    # only ever proposes a box for tracker.select() below - it never touches
    # tracker_core's state machine directly and never overrides an existing
    # lock. See yolo_autoacquire.py's module docstring for why.
    detector = None
    if cfg.get("auto_acquire"):
        try:
            from yolo_autoacquire import AutoAcquireDetector
            detector = AutoAcquireDetector(
                cfg.get("auto_acquire_weights", ""),
                conf_thresh=cfg.get("auto_acquire_conf", 0.80),
                max_box_area_frac=cfg.get("auto_acquire_max_area_frac", 0.10),
                confirm_frames=cfg.get("auto_acquire_confirm_frames", 5),
                log=log)
            if detector.ready:
                detector.start()
                log("Auto-acquire: watching for a drone to lock onto.")
        except Exception as exc:
            log("Auto-acquire: disabled (%s)" % exc)
            detector = None

    # ---- mouse selection ----
    sel = {"drag": False, "p0": None, "p1": None, "box": None,
           "click": None, "clear": False}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            sel["drag"] = True
            sel["p0"] = (x, y)
            sel["p1"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and sel["drag"]:
            sel["p1"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and sel["drag"]:
            sel["drag"] = False
            sel["p1"] = (x, y)
            x0, y0 = sel["p0"]
            w, h = abs(x - x0), abs(y - y0)
            if w < 12 or h < 12:
                sel["click"] = (x, y)      # a click, not a drag
            else:
                sel["box"] = (min(x0, x), min(y0, y), w, h)
        elif event == cv2.EVENT_RBUTTONDOWN:
            sel["clear"] = True

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # ---- control state ----
    prev_err = [0.0, 0.0]
    in_dz = [False, False]
    half_w = frame_w / 2.0
    half_h = frame_h / 2.0
    fps_t, fps_n, fps = time.time(), 0, 0.0
    last_t = time.time()
    last_reason, last_reason_t = "", 0.0

    log("Running. Drag a box round the target in the video window to lock on.")

    try:
        while not stop_event.is_set():
            now = time.time()
            dt = max(1e-3, min(now - last_t, 0.25))
            last_t = now

            ok, frame = cap.read()
            if not ok:
                log("Camera frame error")
                break
            link.drain()

            if detector is not None:
                detector.submit_frame(frame)

            # ---- selection events ----
            if sel["clear"] or ui.get("clear"):
                sel["clear"] = False
                ui["clear"] = False
                if tracker.locked:
                    log("Lock cleared.")
                tracker.clear()
                mount.stop()
            if sel["box"] is not None:
                box, sel["box"] = sel["box"], None
                if cfg.get("click_snap_edges", True):
                    # A generous drag mostly contains desk, and the identity
                    # is snapshotted from exactly what is in the box - a
                    # mostly-desk identity stays "locked" on the desk after
                    # the object is carried away. Shrink to the object when
                    # its edges can be found; the operator's box stands
                    # otherwise.
                    box, snapped = snap_selection(frame, box)
                    if snapped:
                        log("Drag-snap: locking the %dx%d object inside "
                            "the drag" % (box[2], box[3]))
                tracker.select(frame, box)
            if sel["click"] is not None:
                cx, cy = sel["click"]
                sel["click"] = None
                hit, how = None, ""
                s = float(cfg["click_box"])
                if cfg.get("click_snap_edges", True):
                    # No detection under the cursor: find the object's own
                    # edges. A fixed square is the right size for exactly one
                    # object at one range - on anything bigger it locks a patch
                    # of the middle, which is what the operator sees as "it did
                    # not grab what I clicked".
                    hit = estimate_object_box(frame, cx, cy, hint=s)
                    if hit is not None:
                        how = "edges (%dx%d)" % (hit[2], hit[3])
                if hit is None:
                    hit = (cx - s / 2.0, cy - s / 2.0, s, s)
                    how = "fixed %dpx box (no edges found)" % s
                log("Click-lock: %s" % how)
                tracker.select(frame, clip_box(hit, frame_w, frame_h))

            # Only ever originates a fresh lock while idle - never fights or
            # overrides a manual or already-auto-acquired lock, and never
            # bypasses tracker.select()'s own identity/size checks.
            if detector is not None and tracker.state == "idle":
                auto_box = detector.get_confirmed_box()
                if auto_box is not None:
                    log("Auto-acquire: locking on YOLO detection (%dx%d)" %
                        (auto_box[2], auto_box[3]))
                    tracker.select(frame, auto_box)

            st = tracker.update(frame, (), now=now)

            # Why the re-lock has not happened yet, at most once every
            # REASON_LOG_S and only when it changes: a box on the object plus a
            # tracker that will not take it is otherwise indistinguishable from
            # a tracker that cannot see it at all.
            if st.note and now - last_reason_t >= REASON_LOG_S:
                last_reason_t = now
                if st.note != last_reason:
                    last_reason = st.note
                    log("Not re-locking: %s" % st.note)

            # ---- control ----
            jog = ui.get("jog") or (0.0, 0.0)
            if jog[0] or jog[1]:
                pan_cmd = jog[0] * cfg["jog_speed"]
                tilt_cmd = jog[1] * cfg["jog_speed"]
                mount.drive(pan_cmd, tilt_cmd, dt)
                prev_err = [0.0, 0.0]
                mode_txt = "MANUAL JOG"
            elif st.aim is not None and st.state in ("lock", "coast"):
                err_x = (st.aim[0] - half_w) / half_w
                err_y = (half_h - st.aim[1]) / half_h       # + = target is high
                pan_cmd = _axis_cmd(err_x, prev_err, 0, in_dz, dt,
                                    cfg["kp_pan"], cfg["kd_pan"],
                                    cfg["deadzone_x"], cfg["pan_dir"])
                tilt_cmd = _axis_cmd(err_y, prev_err, 1, in_dz, dt,
                                     cfg["kp_tilt"], cfg["kd_tilt"],
                                     cfg["deadzone_y"], cfg["tilt_dir"]) \
                    if cfg["enable_y"] else 0.0
                if st.state == "coast":
                    # aiming at a guess: creep, do not slew
                    pan_cmd *= 0.4
                    tilt_cmd *= 0.4
                mount.drive(pan_cmd, tilt_cmd, dt)
                mode_txt = "TRACKING" if st.state == "lock" else "COASTING"
            else:
                mount.stop()
                prev_err = [0.0, 0.0]
                idle_txt = ("NO TARGET - auto-scanning (YOLO)"
                            if detector is not None else "NO TARGET - drag a box to lock")
                mode_txt = {"idle": idle_txt,
                            "search": "TARGET LOST - searching"}.get(st.state, "")

            mount.fire(bool(ui.get("fire")))

            # ---- overlay ----
            _draw(cv2, frame, st, sel, mode_txt,
                  fps, mount.status(), frame_w, frame_h)

            fps_n += 1
            if now - fps_t >= 1.0:
                fps = fps_n / (now - fps_t)
                fps_t, fps_n = now, 0

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                log("Quit via 'q' in the video window.")
                break
            if key == ord('c'):
                tracker.clear()
                mount.stop()
                log("Lock cleared.")
    except Exception as exc:
        import traceback
        log("Tracker error: %s" % exc)
        log(traceback.format_exc().strip().splitlines()[-1])
    finally:
        if detector is not None:
            detector.stop()
        try:
            mount.fire(False)
            mount.stop()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        link.close()
        log("Stopped. Camera and serial closed.")


def _axis_cmd(err, prev_err, idx, in_dz, dt, kp, kd, deadzone, direction):
    """PD on the normalised image error -> speed fraction in [-1, 1].

    The deadzone has hysteresis: once inside, the axis stays quiet until the
    error grows well past it again. Without that, a mount that overshoots by
    a hair oscillates around the target forever.
    """
    limit = deadzone * (1.8 if in_dz[idx] else 1.0)
    if abs(err) < limit:
        in_dz[idx] = True
        prev_err[idx] = err
        return 0.0
    in_dz[idx] = False
    # The error can step discontinuously - a re-lock puts the target somewhere
    # new in one frame - and a raw derivative would answer that with a full
    # speed kick in whatever direction. The D term is capped at a fraction of
    # the P term, so it can only ever slow the approach down: the mount never
    # drives away from the target because of a one-frame jump.
    d = (err - prev_err[idx]) / dt
    prev_err[idx] = err
    p = kp * err
    lim = 0.6 * abs(p)
    damping = max(-lim, min(lim, kd * d))
    return max(-1.0, min(1.0, direction * (p + damping)))


def _draw(cv2, frame, st, sel, mode_txt, fps, mount_txt, fw, fh):
    cx, cy = fw // 2, fh // 2
    cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 22, 1)

    if st.search_box is not None:
        x, y, w, h = [int(v) for v in st.search_box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 1)

    colors = {"lock": (0, 255, 0), "coast": (0, 210, 255), "search": (0, 120, 255)}
    if st.box is not None and st.state != "idle":
        x, y, w, h = [int(v) for v in st.box]
        col = colors.get(st.state, (200, 200, 200))
        cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
        if st.aim is not None:
            ax, ay = int(st.aim[0]), int(st.aim[1])
            cv2.line(frame, (cx, cy), (ax, ay), col, 1)
            cv2.circle(frame, (ax, ay), 4, col, -1)

    if sel["drag"] and sel["p0"] and sel["p1"]:
        cv2.rectangle(frame, sel["p0"], sel["p1"], (255, 255, 0), 1)

    bar = "%s | %s" % (st.state.upper(), mode_txt) if mode_txt else st.state.upper()
    cv2.putText(frame, bar, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                colors.get(st.state, (255, 255, 255)), 2)
    if st.state != "idle":
        info = "match %.2f (colour %.2f / shape %.2f)" % (st.score, st.color, st.struct)
        if st.lost_s > 0:
            info += "  lost %.1fs" % st.lost_s
        cv2.putText(frame, info, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1)
    # The gate that is holding the re-lock up, in the operator's eyeline.
    if st.note:
        cv2.putText(frame, st.note, (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 200, 255), 1)
    cv2.putText(frame, "%.0f fps  %s" % (fps, mount_txt), (10, fh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


# ================== GUI ==================

class TrackerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Pan/Tilt Tracker Control")
        root.resizable(False, False)

        self.thread = None
        self.test_thread = None
        self._jog_thread = None
        self._jog_quit = False
        self.stop_event = threading.Event()
        self.log_queue = queue.Queue()
        self.ui = {"jog": (0.0, 0.0), "fire": False, "clear": False}

        pad = {"padx": 6, "pady": 3}
        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="n")
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="n")

        # ---- Connection ----
        conn = ttk.LabelFrame(left, text="Connection")
        conn.pack(fill="x", **pad)

        ttk.Label(conn, text="Mount:").grid(row=0, column=0, sticky="w", **pad)
        self.mount_var = tk.StringVar(value=MOUNT_TYPES[0])
        ttk.Combobox(conn, textvariable=self.mount_var, width=22, state="readonly",
                     values=MOUNT_TYPES).grid(row=0, column=1, columnspan=2,
                                              sticky="w", **pad)

        ttk.Label(conn, text="COM port:").grid(row=1, column=0, sticky="w", **pad)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(conn, textvariable=self.port_var, width=22)
        self.port_combo.grid(row=1, column=1, **pad)
        ttk.Button(conn, text="Refresh", command=self.refresh_ports).grid(
            row=1, column=2, **pad)

        ttk.Label(conn, text="Baud:").grid(row=2, column=0, sticky="w", **pad)
        self.baud_var = tk.IntVar(value=BAUD_DEFAULT)
        ttk.Entry(conn, textvariable=self.baud_var, width=10).grid(
            row=2, column=1, sticky="w", **pad)

        ttk.Label(conn, text="Camera index:").grid(row=3, column=0, sticky="w", **pad)
        self.cam_var = tk.IntVar(value=0)
        ttk.Spinbox(conn, from_=0, to=5, textvariable=self.cam_var, width=5).grid(
            row=3, column=1, sticky="w", **pad)

        ttk.Label(conn, text="Resolution:").grid(row=4, column=0, sticky="w", **pad)
        self.res_var = tk.StringVar(value="1280x720")
        ttk.Combobox(conn, textvariable=self.res_var, width=10, state="readonly",
                     values=["640x360", "800x600", "1280x720"]).grid(
            row=4, column=1, sticky="w", **pad)

        # ---- Lock-on ----
        lock = ttk.LabelFrame(left, text="Lock-on (drag a box in the video window)")
        lock.pack(fill="x", **pad)

        # The correlation tracker is CSRT, no longer a choice. Measured on
        # the rig's own pale mug: CSRT frames it at score ~0.89, KCF (fixed
        # box size) at ~0.58 and MOSSE in between - so with any realistic
        # keep-lock threshold CSRT is the only one with real margin, and the
        # cheaper trackers read as "not able to track". Cost is ~25 ms/frame
        # (a ~20 Hz loop); if motion looks steppy, drop the resolution.

        # Profile = the identity lock (memorises the object's look, refuses
        # mismatches). Black blob = blob_lock.py, no appearance at all: follow
        # the dark blob against the sky, catch the strongest one when lost.
        self.lock_mode_var = tk.StringVar(value="Profile (identity)")
        ttk.Label(lock, text="Lock mode:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(lock, textvariable=self.lock_mode_var, width=16,
                     state="readonly",
                     values=["Profile (identity)", "Black blob"]).grid(
            row=0, column=1, sticky="w", **pad)

        self.stabilise_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(lock, text="Vibration compensation",
                        variable=self.stabilise_var).grid(row=0, column=2,
                                                          columnspan=2, sticky="w", **pad)

        def num(parent, label, var, r, c, width=7):
            ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w", **pad)
            ttk.Entry(parent, textvariable=var, width=width).grid(
                row=r, column=c + 1, sticky="w", **pad)

        self.track_thresh_var = tk.DoubleVar(value=0.42)
        self.relock_thresh_var = tk.DoubleVar(value=0.60)
        self.relock_frames_var = tk.IntVar(value=3)
        self.relock_margin_var = tk.DoubleVar(value=0.08)
        self.coast_var = tk.DoubleVar(value=0.5)
        self.giveup_var = tk.DoubleVar(value=0.0)
        self.max_jump_var = tk.DoubleVar(value=0.22)
        self.aim_smooth_var = tk.DoubleVar(value=0.45)
        self.click_box_var = tk.IntVar(value=70)
        # A re-lock must clear the colour and shape floors separately, not just
        # the combined score - a box at the wrong scale sweeps in background and
        # keeps a high colour score while its shape is ruined. These are the two
        # numbers the "Not re-locking" message quotes, so they belong on the
        # panel: reading "colour 0.00<0.35" with no way to change the 0.35 is
        # the reason they were added.
        self.relock_color_var = tk.DoubleVar(value=0.35)
        self.relock_struct_var = tk.DoubleVar(value=0.34)

        num(lock, "Keep-lock score:", self.track_thresh_var, 1, 0)
        num(lock, "Re-lock score:", self.relock_thresh_var, 1, 2)
        num(lock, "Re-lock frames:", self.relock_frames_var, 2, 0)
        num(lock, "Rival margin:", self.relock_margin_var, 2, 2)
        num(lock, "Re-lock colour floor:", self.relock_color_var, 3, 0)
        num(lock, "Re-lock shape floor:", self.relock_struct_var, 3, 2)
        num(lock, "Coast (s):", self.coast_var, 4, 0)
        num(lock, "Give up after (s):", self.giveup_var, 4, 2)
        num(lock, "Max jump (frac):", self.max_jump_var, 5, 0)
        num(lock, "Aim smoothing:", self.aim_smooth_var, 5, 2)
        num(lock, "Click box (px):", self.click_box_var, 6, 0)
        ttk.Label(lock, text="0 = never give up").grid(row=6, column=2,
                                                       columnspan=2, sticky="w", **pad)

        self.click_edges_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(lock, text="Snap a click to the object's edges",
                        variable=self.click_edges_var).grid(
            row=7, column=0, columnspan=4, sticky="w", **pad)

        # Appearance alone cannot separate two identical objects; history can.
        # Anything standing elsewhere while the real target was locked is
        # remembered and refused later. Switchable because it can also refuse
        # the genuine target if it comes back to a place a look-alike stood.
        self.distractor_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(lock, text="Remember look-alikes and refuse them",
                        variable=self.distractor_var).grid(
            row=8, column=0, columnspan=4, sticky="w", **pad)

        # ---- Auto-acquire (YOLO) ----
        # Optional, off by default: skips the manual drag/click and lets a
        # YOLO detection originate the lock instead, via the same
        # tracker.select() a manual drag calls. Never overrides an existing
        # lock. See yolo_autoacquire.py's module docstring for the full
        # rationale and CLAUDE.md for why this needed asking first.
        auto = ttk.LabelFrame(left, text="Auto-acquire (YOLO)")
        auto.pack(fill="x", **pad)

        self.auto_acquire_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(auto, text="Auto-acquire lock from YOLO detections",
                        variable=self.auto_acquire_var).grid(
            row=0, column=0, columnspan=4, sticky="w", **pad)

        default_weights = os.path.join(
            HERE, "..", "drone_detection_module", "weights", "drone_yolov11x.pt")
        default_weights = default_weights if os.path.exists(default_weights) else ""
        self.auto_acquire_weights_var = tk.StringVar(value=default_weights)
        ttk.Label(auto, text="Weights:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(auto, textvariable=self.auto_acquire_weights_var, width=32).grid(
            row=1, column=1, columnspan=2, sticky="w", **pad)

        def browse_weights():
            from tkinter import filedialog
            path = filedialog.askopenfilename(
                title="Select YOLO weights (.pt)",
                filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
            if path:
                self.auto_acquire_weights_var.set(path)

        ttk.Button(auto, text="Browse...", command=browse_weights).grid(
            row=1, column=3, sticky="w", **pad)

        self.auto_acquire_conf_var = tk.DoubleVar(value=0.80)
        self.auto_acquire_max_area_var = tk.DoubleVar(value=0.10)
        self.auto_acquire_confirm_var = tk.IntVar(value=4)
        num(auto, "Confidence floor:", self.auto_acquire_conf_var, 2, 0)
        num(auto, "Max box (frac of frame):", self.auto_acquire_max_area_var, 2, 2)
        num(auto, "Confirm cycles:", self.auto_acquire_confirm_var, 3, 0)
        ttk.Label(auto, text="~0.12s/cycle").grid(row=3, column=2,
                                                 columnspan=2, sticky="w", **pad)

        # ---- Aiming ----
        mot = ttk.LabelFrame(right, text="Aiming")
        mot.pack(fill="x", **pad)

        self.enable_y_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mot, text="Enable tilt axis",
                        variable=self.enable_y_var).grid(row=0, column=0,
                                                         columnspan=4, sticky="w", **pad)

        self.kp_pan_var = tk.DoubleVar(value=1.6)
        self.kp_tilt_var = tk.DoubleVar(value=1.4)
        self.kd_pan_var = tk.DoubleVar(value=0.12)
        self.kd_tilt_var = tk.DoubleVar(value=0.10)
        self.deadzone_x_var = tk.DoubleVar(value=0.045)
        self.deadzone_y_var = tk.DoubleVar(value=0.055)
        self.pan_dir_var = tk.DoubleVar(value=1.0)
        self.tilt_dir_var = tk.DoubleVar(value=1.0)

        num(mot, "Kp pan:", self.kp_pan_var, 1, 0)
        num(mot, "Kp tilt:", self.kp_tilt_var, 1, 2)
        num(mot, "Kd pan:", self.kd_pan_var, 2, 0)
        num(mot, "Kd tilt:", self.kd_tilt_var, 2, 2)
        num(mot, "Deadzone X:", self.deadzone_x_var, 3, 0)
        num(mot, "Deadzone Y:", self.deadzone_y_var, 3, 2)
        num(mot, "Pan dir (+1/-1):", self.pan_dir_var, 4, 0)
        num(mot, "Tilt dir (+1/-1):", self.tilt_dir_var, 4, 2)
        ttk.Label(mot, text="Deadzone is a fraction of the half-frame "
                            "(0.045 ~ 29 px at 1280)").grid(
            row=5, column=0, columnspan=4, sticky="w", **pad)

        # ---- DC motor drive ----
        dc = ttk.LabelFrame(right, text="DC drive (ESP32 / BTS7960)")
        dc.pack(fill="x", **pad)
        self.min_duty_var = tk.IntVar(value=340)
        self.max_duty_var = tk.IntVar(value=1023)
        self.min_duty_neg_var = tk.IntVar(value=340)
        self.min_duty_tilt_var = tk.IntVar(value=360)
        self.max_duty_tilt_var = tk.IntVar(value=1023)
        self.min_duty_tilt_neg_var = tk.IntVar(value=360)
        self.pulse_below_var = tk.DoubleVar(value=0.30)
        self.jog_speed_var = tk.DoubleVar(value=0.55)
        num(dc, "Min duty pan right:", self.min_duty_var, 0, 0)
        num(dc, "Min duty pan left:", self.min_duty_neg_var, 0, 2)
        num(dc, "Min duty tilt up:", self.min_duty_tilt_var, 1, 0)
        num(dc, "Min duty tilt down:", self.min_duty_tilt_neg_var, 1, 2)
        num(dc, "Max duty pan:", self.max_duty_var, 2, 0)
        num(dc, "Max duty tilt:", self.max_duty_tilt_var, 2, 2)
        num(dc, "Pulse below:", self.pulse_below_var, 3, 0)
        num(dc, "Jog speed:", self.jog_speed_var, 3, 2)
        ttk.Label(dc, text="Min duty = the least that direction actually turns "
                           "at - they differ (gravity, stiff bearings, a tired "
                           "half-bridge).\nRun find_min_duty.py to measure all "
                           "four.").grid(
            row=4, column=0, columnspan=4, sticky="w", **pad)

        # ---- Stepper-only ----
        stp = ttk.LabelFrame(right, text="Stepper mount only")
        stp.pack(fill="x", **pad)
        self.max_speed_x_var = tk.DoubleVar(value=12.0)
        self.max_speed_y_var = tk.DoubleVar(value=20.0)
        self.scan_min_var = tk.DoubleVar(value=-180.0)
        self.scan_max_var = tk.DoubleVar(value=180.0)
        self.tilt_center_var = tk.DoubleVar(value=90.0)
        self.tilt_min_var = tk.DoubleVar(value=0.0)
        self.tilt_max_var = tk.DoubleVar(value=180.0)
        num(stp, "Speed X (deg/s):", self.max_speed_x_var, 0, 0)
        num(stp, "Speed Y (deg/s):", self.max_speed_y_var, 0, 2)
        num(stp, "Pan min:", self.scan_min_var, 1, 0)
        num(stp, "Pan max:", self.scan_max_var, 1, 2)
        num(stp, "Tilt centre:", self.tilt_center_var, 2, 0)
        num(stp, "Tilt min:", self.tilt_min_var, 2, 2)
        num(stp, "Tilt max:", self.tilt_max_var, 3, 0)

        # ---- Manual jog ----
        jog = ttk.LabelFrame(right, text="Manual jog (while running)")
        jog.pack(fill="x", **pad)
        grid = ttk.Frame(jog)
        grid.pack(pady=4)
        self._jog_btn(grid, "UP", 0, 1, (0.0, 1.0))
        self._jog_btn(grid, "LEFT", 1, 0, (-1.0, 0.0))
        ttk.Button(grid, text="CLEAR\nLOCK", width=8,
                   command=self.clear_lock).grid(row=1, column=1, padx=2, pady=2)
        self._jog_btn(grid, "RIGHT", 1, 2, (1.0, 0.0))
        self._jog_btn(grid, "DOWN", 2, 1, (0.0, -1.0))

        fire_row = ttk.Frame(jog)
        fire_row.pack(pady=4)
        self.arm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fire_row, text="ARM trigger",
                        variable=self.arm_var).pack(side="left", padx=6)
        self.fire_btn = tk.Button(fire_row, text="FIRE (hold)", width=12,
                                  bg="#cc2222", fg="white")
        self.fire_btn.pack(side="left", padx=6)
        self.fire_btn.bind("<ButtonPress-1>", lambda _e: self._fire(True))
        self.fire_btn.bind("<ButtonRelease-1>", lambda _e: self._fire(False))

        # ---- Buttons + status ----
        btns = ttk.Frame(root)
        btns.grid(row=1, column=0, columnspan=2, sticky="ew", **pad)
        self.start_btn = ttk.Button(btns, text="Start", command=self.start)
        self.start_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.test_btn = ttk.Button(btns, text="Test Motors", command=self.test_motors)
        self.test_btn.pack(side="left", padx=6)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(btns, textvariable=self.status_var).pack(side="left", padx=12)

        self.log_text = tk.Text(root, height=10, width=104, state="disabled",
                                font=("Consolas", 9))
        self.log_text.grid(row=2, column=0, columnspan=2, sticky="ew",
                           padx=6, pady=(0, 6))

        self._init_settings_persistence()
        self.refresh_ports()
        self.root.after(100, self.poll_log)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---- jog ----
    def _jog_btn(self, parent, text, r, c, vec):
        b = tk.Button(parent, text=text, width=8, height=2)
        b.grid(row=r, column=c, padx=2, pady=2)
        b.bind("<ButtonPress-1>", lambda _e, v=vec: self._jog_press(v))
        b.bind("<ButtonRelease-1>", lambda _e: self.ui.update(jog=(0.0, 0.0)))
        return b

    def _jog_press(self, vec):
        self.ui["jog"] = vec
        # With the tracker running, its loop owns the serial port and picks
        # this up. With it stopped, nothing was reading it and the buttons did
        # nothing at all - so drive the mount from here instead. Aiming the
        # mount should not require starting the camera.
        if self.thread is None:
            self._start_manual_jog()

    def _start_manual_jog(self):
        if self._jog_thread is not None and self._jog_thread.is_alive():
            return
        if self.test_thread is not None and self.test_thread.is_alive():
            self.log("Jog: motor test is using the port, wait for it to finish.")
            return
        if "None" in self.port_var.get():
            self.log("Jog: select a real COM port first.")
            return
        self._jog_quit = False
        # Tk variables may only be read on the main thread - doing it inside
        # the worker raises "main thread is not in main loop" and kills it
        # silently. Collect everything here and hand it over.
        self._jog_thread = threading.Thread(
            target=self._run_manual_jog,
            args=(self._mount_cfg(), self.port_var.get().split(" ")[0],
                  self.baud_var.get(), self.mount_var.get(),
                  self.jog_speed_var.get()),
            daemon=True)
        self._jog_thread.start()

    def _mount_cfg(self):
        """Mount settings snapshot. Main thread only."""
        return {
            "min_duty": self.min_duty_var.get(),
            "max_duty": self.max_duty_var.get(),
            "min_duty_neg": self.min_duty_neg_var.get(),
            "min_duty_tilt_neg": self.min_duty_tilt_neg_var.get(),
            "min_duty_tilt": self.min_duty_tilt_var.get(),
            "max_duty_tilt": self.max_duty_tilt_var.get(),
            "pulse_below": self.pulse_below_var.get(),
            "max_speed_x": self.max_speed_x_var.get(),
            "max_speed_y": self.max_speed_y_var.get(),
            "scan_min": self.scan_min_var.get(),
            "scan_max": self.scan_max_var.get(),
            "tilt_min": self.tilt_min_var.get(),
            "tilt_max": self.tilt_max_var.get(),
            "tilt_center": self.tilt_center_var.get(),
        }

    def _run_manual_jog(self, cfg, port, baud, mount_type, speed):
        from mount import SerialLink, make_mount
        self.log("Jog: connecting (the board resets when the port opens)...")
        link = SerialLink(port, baud, log=self.log)
        mount = make_mount(mount_type, link, cfg)
        last = time.time()
        idle_since = None
        self.log("Jog ready - hold a direction button.")
        try:
            while not self._jog_quit and self.thread is None:
                now = time.time()
                dt = max(1e-3, now - last)
                last = now
                jog = self.ui.get("jog") or (0.0, 0.0)
                if jog[0] or jog[1]:
                    idle_since = None
                    mount.drive(jog[0] * speed, jog[1] * speed, dt)
                else:
                    mount.stop()
                    if idle_since is None:
                        idle_since = now
                    elif (now - idle_since) > 2.0:
                        break      # released a while ago - give the port back
                link.drain()
                time.sleep(0.03)
        except Exception as exc:
            self.log("Jog error: %s" % exc)
        finally:
            try:
                mount.stop()
            except Exception:
                pass
            link.close()

    def _stop_manual_jog(self):
        self._jog_quit = True
        self.ui["jog"] = (0.0, 0.0)
        if self._jog_thread is not None:
            self._jog_thread.join(timeout=2.0)
            self._jog_thread = None

    def _fire(self, on):
        if on and not self.arm_var.get():
            self.log("FIRE ignored - tick ARM trigger first.")
            return
        self.ui["fire"] = bool(on)

    def clear_lock(self):
        self.ui["clear"] = True

    # ---- settings persistence ----
    def _init_settings_persistence(self):
        self._settings_vars = {
            "mount_type": self.mount_var,
            "port": self.port_var,
            "baud": self.baud_var,
            "camera": self.cam_var,
            "resolution": self.res_var,
            "lock_mode": self.lock_mode_var,
            "stabilise": self.stabilise_var,
            "track_thresh": self.track_thresh_var,
            "relock_thresh": self.relock_thresh_var,
            "relock_frames": self.relock_frames_var,
            "relock_margin": self.relock_margin_var,
            "relock_color_min": self.relock_color_var,
            "relock_struct_min": self.relock_struct_var,
            "distractor_guard": self.distractor_var,
            "coast_s": self.coast_var,
            "give_up_s": self.giveup_var,
            "max_jump_frac": self.max_jump_var,
            "aim_smooth": self.aim_smooth_var,
            "click_box": self.click_box_var,
            "click_snap_edges": self.click_edges_var,
            "auto_acquire": self.auto_acquire_var,
            "auto_acquire_weights": self.auto_acquire_weights_var,
            "auto_acquire_conf": self.auto_acquire_conf_var,
            "auto_acquire_max_area_frac": self.auto_acquire_max_area_var,
            "auto_acquire_confirm_frames": self.auto_acquire_confirm_var,
            "enable_y": self.enable_y_var,
            "kp_pan": self.kp_pan_var,
            "kp_tilt": self.kp_tilt_var,
            "kd_pan": self.kd_pan_var,
            "kd_tilt": self.kd_tilt_var,
            "deadzone_x": self.deadzone_x_var,
            "deadzone_y": self.deadzone_y_var,
            "pan_dir": self.pan_dir_var,
            "tilt_dir": self.tilt_dir_var,
            "min_duty": self.min_duty_var,
            "max_duty": self.max_duty_var,
            "min_duty_neg": self.min_duty_neg_var,
            "min_duty_tilt_neg": self.min_duty_tilt_neg_var,
            "min_duty_tilt": self.min_duty_tilt_var,
            "max_duty_tilt": self.max_duty_tilt_var,
            "pulse_below": self.pulse_below_var,
            "jog_speed": self.jog_speed_var,
            "max_speed_x": self.max_speed_x_var,
            "max_speed_y": self.max_speed_y_var,
            "scan_min": self.scan_min_var,
            "scan_max": self.scan_max_var,
            "tilt_center": self.tilt_center_var,
            "tilt_min": self.tilt_min_var,
            "tilt_max": self.tilt_max_var,
        }
        self._save_pending = None
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            if saved.get("settings_version", 1) < SETTINGS_VERSION:
                dropped = [k for k in RESCALED_IN_V2 if k in saved]
                for key in dropped:
                    del saved[key]
                if dropped:
                    self.log("Settings file was from the old stepper build - "
                             "reset %s to the new defaults."
                             % ", ".join(sorted(dropped)))
            for key, var in self._settings_vars.items():
                if key in saved:
                    try:
                        var.set(saved[key])
                    except tk.TclError:
                        pass
            if saved.get("settings_version", 1) < SETTINGS_VERSION:
                self._save_settings()   # migrate on disk, don't warn again
        except (OSError, ValueError):
            pass  # first run or unreadable file - keep defaults
        for var in self._settings_vars.values():
            var.trace_add("write", self._on_setting_changed)

    def _on_setting_changed(self, *_):
        # Entries fire on every keystroke; write 400 ms after the last change.
        if self._save_pending is not None:
            self.root.after_cancel(self._save_pending)
        self._save_pending = self.root.after(400, self._save_settings)

    def _save_settings(self):
        self._save_pending = None
        data = {"settings_version": SETTINGS_VERSION}
        for key, var in self._settings_vars.items():
            try:
                data[key] = var.get()
            except tk.TclError:
                pass  # field mid-edit (e.g. a lone "-"); skip it
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError:
            pass

    # ---- helpers ----
    # Descriptions of the USB-serial bridges these boards use, best first.
    KNOWN_BRIDGES = ("cp210", "ch340", "ch910", "ft232", "usb-serial", "usb serial")

    def refresh_ports(self):
        try:
            from serial.tools import list_ports
            infos = list(list_ports.comports())
        except Exception:
            infos = []
        values = [p.device for p in infos]
        values.append("None (no serial)")

        saved = self.port_var.get()
        if saved and saved not in values:
            # Keep the remembered port on the list even while the board is
            # unplugged. Silently replacing it used to overwrite the saved
            # setting, so one run with the mount disconnected lost it.
            values.insert(0, saved)
        self.port_combo["values"] = values

        if not saved:
            self.port_var.set(self._guess_port(infos) or values[0])

    def _guess_port(self, infos):
        """First run only: pick the port that looks like the mount."""
        for info in infos:
            desc = ("%s %s" % (info.device, info.description or "")).lower()
            if any(k in desc for k in self.KNOWN_BRIDGES):
                return info.device
        return None

    def log(self, msg):
        self.log_queue.put(msg)

    def poll_log(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.configure(state="normal")
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass
        running = self.thread is not None and self.thread.is_alive()
        testing = self.test_thread is not None and self.test_thread.is_alive()
        if self.thread is not None and not running:
            self.thread = None
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.status_var.set("Idle")
        if self.test_thread is not None and not testing:
            self.test_thread = None
        self.test_btn.configure(state="disabled" if (running or testing) else "normal")
        self.root.after(100, self.poll_log)

    # ---- motor test ----
    def test_motors(self):
        if self.thread is not None or self.test_thread is not None:
            return
        if "None" in self.port_var.get():
            self.log("Test motors: select a real COM port first.")
            return
        self._stop_manual_jog()   # only one owner of the port at a time
        # Tk variables are main-thread only; snapshot them before handing off.
        params = {
            "mount_type": self.mount_var.get(),
            "tilt_center": self.tilt_center_var.get(),
            "min_duty": self.min_duty_var.get(),
            "min_duty_tilt": self.min_duty_tilt_var.get(),
        }
        self.test_thread = threading.Thread(
            target=self._run_motor_test,
            args=(self.port_var.get().split(" ")[0], self.baud_var.get(),
                  params),
            daemon=True)
        self.test_thread.start()
        self.test_btn.configure(state="disabled")
        self.status_var.set("Testing motors...")

    def _run_motor_test(self, port, baud, params):
        from mount import SerialLink
        stepper = params["mount_type"].startswith("Stepper")
        link = SerialLink(port, baud, log=self.log)
        try:
            if stepper:
                centre = params["tilt_center"]
                seq = [("x=15", 2.0), ("x=-15", 3.0), ("x=0", 2.0),
                       ("y=%.1f" % (centre + 15), 2.0),
                       ("y=%.1f" % (centre - 15), 3.0),
                       ("y=%.1f" % centre, 2.0)]
            else:
                p = int(params["min_duty"]) + 150
                t = int(params["min_duty_tilt"]) + 150
                seq = [("V", 0.5), ("M,%d,0" % p, 1.0), ("M,%d,0" % -p, 1.0),
                       ("M,0,0", 0.5), ("M,0,%d" % t, 1.0), ("M,0,%d" % -t, 1.0),
                       ("S", 0.5)]
            for cmd, wait in seq:
                link.write_line(cmd)
                self.log("Test motors: sent %s" % cmd)
                end = time.time() + wait
                while time.time() < end:
                    link.drain()
                    time.sleep(0.02)
                    # DC mount: keep the watchdog fed while the step runs
                    if not stepper:
                        link.write_line(cmd)
            self.log("Test motors: done.")
        except Exception as exc:
            self.log("Test motors ERROR: %s" % exc)
        finally:
            try:
                link.write_line("S")
            except Exception:
                pass
            link.close()

    # ---- start / stop ----
    def start(self):
        if self.thread is not None:
            return
        # the standalone jog holds the port open; hand it over before the
        # tracker tries to claim it
        self._stop_manual_jog()
        w, h = self.res_var.get().split("x")
        port = self.port_var.get().split(" ")[0] \
            if "None" not in self.port_var.get() else "None (no serial)"
        cfg = {
            "mount_type": self.mount_var.get(),
            "port": port,
            "baud": self.baud_var.get(),
            "cam": self.cam_var.get(),
            "frame_w": int(w),
            "frame_h": int(h),
            # tracker core
            "lock_mode": self.lock_mode_var.get(),
            "tracker": "CSRT",
            "stabilise": self.stabilise_var.get(),
            "track_thresh": self.track_thresh_var.get(),
            "relock_thresh": self.relock_thresh_var.get(),
            "relock_frames": max(1, self.relock_frames_var.get()),
            "relock_margin": self.relock_margin_var.get(),
            "relock_color_min": self.relock_color_var.get(),
            "relock_struct_min": self.relock_struct_var.get(),
            "distractor_guard": self.distractor_var.get(),
            "coast_s": self.coast_var.get(),
            "give_up_s": self.giveup_var.get(),
            "max_jump_frac": self.max_jump_var.get(),
            "aim_smooth": self.aim_smooth_var.get(),
            "click_box": max(20, self.click_box_var.get()),
            "click_snap_edges": self.click_edges_var.get(),
            # detection assist
            "auto_acquire": self.auto_acquire_var.get(),
            "auto_acquire_weights": self.auto_acquire_weights_var.get(),
            "auto_acquire_conf": self.auto_acquire_conf_var.get(),
            "auto_acquire_max_area_frac": self.auto_acquire_max_area_var.get(),
            "auto_acquire_confirm_frames": max(1, self.auto_acquire_confirm_var.get()),
            # aiming
            "enable_y": self.enable_y_var.get(),
            "kp_pan": self.kp_pan_var.get(),
            "kp_tilt": self.kp_tilt_var.get(),
            "kd_pan": self.kd_pan_var.get(),
            "kd_tilt": self.kd_tilt_var.get(),
            "deadzone_x": self.deadzone_x_var.get(),
            "deadzone_y": self.deadzone_y_var.get(),
            "pan_dir": self.pan_dir_var.get(),
            "tilt_dir": self.tilt_dir_var.get(),
            # dc drive
            "min_duty": self.min_duty_var.get(),
            "max_duty": self.max_duty_var.get(),
            "min_duty_neg": self.min_duty_neg_var.get(),
            "min_duty_tilt_neg": self.min_duty_tilt_neg_var.get(),
            "min_duty_tilt": self.min_duty_tilt_var.get(),
            "max_duty_tilt": self.max_duty_tilt_var.get(),
            "pulse_below": self.pulse_below_var.get(),
            "jog_speed": self.jog_speed_var.get(),
            # stepper
            "max_speed_x": self.max_speed_x_var.get(),
            "max_speed_y": self.max_speed_y_var.get(),
            "scan_min": self.scan_min_var.get(),
            "scan_max": self.scan_max_var.get(),
            "tilt_center": self.tilt_center_var.get(),
            "tilt_min": self.tilt_min_var.get(),
            "tilt_max": self.tilt_max_var.get(),
        }
        self.ui.update(jog=(0.0, 0.0), fire=False, clear=False)
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=run_tracker, args=(cfg, self.stop_event, self.ui, self.log),
            daemon=True)
        self.thread.start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("Running on %s" % cfg["port"])

    def stop(self):
        self.stop_event.set()

    def on_close(self):
        if self._save_pending is not None:
            self.root.after_cancel(self._save_pending)
        self._save_settings()
        self._stop_manual_jog()
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    gui = TrackerGUI(root)
    root.mainloop()
