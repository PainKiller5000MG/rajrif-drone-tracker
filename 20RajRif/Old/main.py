"""Legacy CUAS operator station, re-eyed with the identity lock.

This is the original 15 RAJRIF / 222 FD WKSP app - D-pad, FIRE, IDMS azimuth
slew, gyro fallback, map view - with its YOLO biggest-box detector replaced by
the shared identity lock (../tracker_core.py). The operator drags a box round
the target in the video panel, exactly as in the new tracker GUI, and the lock
holds that object through banks, jumps and disappearance rather than chasing
whatever detection is largest this frame.

What did NOT change: the serial protocol (legacy direction words with a
,<speed> suffix - the firmware still answers them), send_command's zone-based
speed mapping, the IDMS/gyro/map integrations, and the fire controls. AUTO
FIRE kept its legacy behaviour but is additionally gated on the lock being in
the 'lock' state - it will never fire on a coasted prediction or a search
guess, only on the confirmed, identity-checked target.

The drag/click remains the primary way a target is acquired. An optional
AUTO-ACQUIRE button (off by default, added 2026-08-13 for a specific demo -
see ../yolo_autoacquire.py and CLAUDE.md's "Do not reintroduce it without
asking" note) lets a YOLO detection originate the lock instead, through the
exact same tracker.select() call the drag uses - it never bypasses
tracker_core/blob_lock's own state machine. AUTO FIRE's existing gates
(lock state + PROFILE mode + ARM) are unchanged and apply the same way to
an auto-acquired lock as to a manual one; that combination is a real risk
increase and is a deliberate, asked-for tradeoff, not an oversight.

The pre-port app is main_pre_lock_backup.py, byte-for-byte.
"""

import math
import os
import sys
import threading
import time
import tkinter as tk

# Without this, opening the camera through MSMF takes ~40 s. Must be set
# before cv2 creates a VideoCapture.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import serial
from PIL import Image, ImageTk

# The shared vision core lives one directory up and is byte-identical with
# Scan360's - import it, never copy it here, or the copies drift.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracker_core import LockTracker, clip_box, estimate_object_box
from blob_lock import BlobTracker

from idms_map_view import DroneTrackerMap
from gyro_data import get_roll_yaw
from idms import get_IDMS_tgt_data
from load_mount_data import load_mount_config

# data will be refreshed from the file mount_config.json
mount_config = load_mount_config()

# ---------------- CONFIG ----------------
SERIAL_PORT = "/dev/tty.usbserial-0001"
if mount_config["ESP32_PORT"] is not None:
    SERIAL_PORT = mount_config["ESP32_PORT"]
BAUD_RATE = 115200
TOLERANCE = 50
COMMAND_DELAY = 0.05
# KCF, not CSRT: measured 2026-08-09 on recorded flights, KCF held the lock
# as well or better (clip 1: 100% vs 97.9%) at 4.6 ms/frame against CSRT's
# 30 - and this app shares its loop with tkinter, so the cheap tracker is
# what keeps the video smooth.
TRACKER_ALGO = "KCF"
COAST_ZONE_CAP = 40     # aiming at a prediction: creep, do not slew

# Optional YOLO auto-acquire (off by default - see AUTO-ACQUIRE button).
# Same defaults as tracker_gui.py's panel: confidence floor sits above the
# 0.25-0.68 range measured on a real false positive, max box area is well
# above real drone detections (<=2.6% of frame, measured) but far below that
# false positive's 35-60%, and confirm_frames requires ~1s of a consistent
# detection before it is trusted - see yolo_autoacquire.py's docstring.
AUTO_ACQUIRE_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "drone_detection_module", "weights", "drone_yolov11x.pt")
AUTO_ACQUIRE_CONF = 0.80
AUTO_ACQUIRE_MAX_AREA_FRAC = 0.10
AUTO_ACQUIRE_CONFIRM_FRAMES = 4

STATE_COLORS = {"lock": (0, 255, 0), "coast": (0, 210, 255),
                "search": (0, 120, 255)}


# ---------------- PATH ----------------
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# ---------------- SERIAL ----------------
def init_serial():
    try:
        esp = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("ESP32 Connected")
        return esp
    except Exception as e:
        print("ESP32 not connected:", e)
        return None


def send_command(esp, command, zone=100):
    if not esp:
        return
    try:
        new_min = 350
        if mount_config["motor_speed_min"] is not None:
            new_min = mount_config["motor_speed_min"]

        old_min, old_max = 1, 100
        new_max = 1000
        speed = new_min + (zone - old_min) * (new_max - new_min) / (old_max - old_min)
        motor_speed = str(int(speed))

        cmd = command + "," + motor_speed
        print(cmd, flush=True)
        esp.write((cmd + "\n").encode())
    except Exception:
        pass


# ---------------- CAMERA ----------------
def init_camera():
    # MSMF first for the same reason as the new GUI; fall back if it refuses.
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


# ---------------- MOVEMENT ----------------
def get_movement(cx, cy, center_x, center_y):
    dx, dy = "", ""

    if cx < center_x - TOLERANCE:
        dx = "LEFT"
    elif cx > center_x + TOLERANCE:
        dx = "RIGHT"

    if cy < center_y - TOLERANCE:
        dy = "UP"
    elif cy > center_y + TOLERANCE:
        dy = "DOWN"

    return " ".join(filter(None, [dy, dx])) or "CENTERED"


def get_zone(cx, cy, center_x, center_y, frame_width, frame_height):
    dist = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
    max_dist = math.sqrt(center_x ** 2 + center_y ** 2)
    return dist / max_dist * 100


# ---------------- APP ----------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("15 RAJRIF & 222 FD WKSP")

        self.tracking_enabled = False
        self.auto_fire_enabled = False

        self.left = tk.Frame(root, width=300, bg="#1e1e1e")
        self.left.pack(side="left", fill="y")

        self.right = tk.Frame(root, bg="black")
        self.right.pack(side="right")

        self.video_label = tk.Label(self.right)
        self.video_label.pack()

        self.esp = init_serial()
        self.cap = init_camera()

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2

        # ---- the lock (replaces the YOLO thread) ----
        # PROFILE = the shared identity lock: memorises the object's look,
        # never takes the wrong one. BLOB = blob_lock: no appearance at all,
        # follow the dark blob, catch the strongest when lost - and because
        # a blob can be a bird, AUTO FIRE only arms in PROFILE mode.
        self.lock_mode = "PROFILE"
        self.tracker = self._make_tracker()

        # ---- optional auto-acquire (off by default) ----
        # Runs YOLO on its own thread, decoupled from this app's ~15ms
        # update_frame() cadence, and only ever proposes a box for
        # tracker.select() - never bypasses the lock's own state machine,
        # never overrides an existing lock (see the idle check in
        # update_frame). See yolo_autoacquire.py's module docstring.
        self.auto_acquire_enabled = False
        self.detector = None
        try:
            from yolo_autoacquire import AutoAcquireDetector
            self.detector = AutoAcquireDetector(
                AUTO_ACQUIRE_WEIGHTS,
                conf_thresh=AUTO_ACQUIRE_CONF,
                max_box_area_frac=AUTO_ACQUIRE_MAX_AREA_FRAC,
                confirm_frames=AUTO_ACQUIRE_CONFIRM_FRAMES,
                log=lambda m: print("[AUTO-ACQUIRE]", m))
        except Exception as exc:
            print("[AUTO-ACQUIRE] could not load (%s) - button will stay disabled" % exc)
            self.detector = None

        # drag state for the selection rubber band, in frame coordinates
        self.sel = {"drag": False, "p0": None, "p1": None,
                    "box": None, "click": None, "clear": False}
        self.video_label.bind("<ButtonPress-1>", self._sel_down)
        self.video_label.bind("<B1-Motion>", self._sel_move)
        self.video_label.bind("<ButtonRelease-1>", self._sel_up)
        self.video_label.bind("<ButtonPress-3>", self._sel_clear)

        self.rp_color = (0, 0, 0)
        self.last_command = None
        self.last_fire_command = None
        self.last_sent_time = 0
        self.droneDetectedTime = time.time()

        self.build_controls()
        self.set_manual_enabled(True)
        self.track_btn.config(text="TRACKING: OFF", bg="gray")

        self.running = True

        self.idms_data = {
            "azimuth": None, "drone_distance": None, "source": None,
            "latitude": None, "longitude": None, "raw": None,
            "timestamp": None,
        }
        self.idms_tgt_dist_to_consider = mount_config["idms_tgt_dist_to_consider"]
        self.mount_latitude = mount_config["mount_latitude"]
        self.mount_longitude = mount_config["mount_longitude"]
        self.default_mount_roll = mount_config["default_mount_roll"]
        self.default_mount_yaw = mount_config["default_mount_yaw"]
        self.idms_reported_en_drone = None

        self.idms_lock = threading.Lock()
        self.idms_running = True
        threading.Thread(target=self.idms_loop, daemon=True).start()

        self.map_view = DroneTrackerMap(
            self.root, self.mount_latitude, self.mount_longitude)

        self.update_frame()
        self.update_map_view()

    def _make_tracker(self):
        if self.lock_mode == "BLOB":
            return BlobTracker({"coast_s": 0.5, "relock_frames": 3,
                                "aim_smooth": 0.45},
                               log=lambda m: print("[BLOB]", m))
        return LockTracker({"tracker": TRACKER_ALGO, "give_up_s": 0.0},
                           log=lambda m: print("[LOCK]", m))

    def toggle_lock_mode(self):
        self.lock_mode = "BLOB" if self.lock_mode == "PROFILE" else "PROFILE"
        self.tracker = self._make_tracker()      # fresh - any lock is cleared
        send_command(self.esp, "CENTERED")
        self.last_command = "CENTERED"
        self.mode_btn.config(
            text=f"MODE: {self.lock_mode}",
            fg="purple" if self.lock_mode == "BLOB" else "blue")
        if self.lock_mode == "BLOB" and self.auto_fire_enabled:
            self.toggle_auto_fire()              # blob mode may follow a bird
        self.auto_btn.config(
            state="disabled" if self.lock_mode == "BLOB" else "normal")
        print("[MODE] %s%s" % (self.lock_mode,
                               "  (AUTO FIRE unavailable in BLOB)"
                               if self.lock_mode == "BLOB" else ""))

    # ---------------- SELECTION (replaces the YOLO thread) ----------------
    def _sel_down(self, e):
        self.sel["drag"] = True
        self.sel["p0"] = (e.x, e.y)
        self.sel["p1"] = (e.x, e.y)

    def _sel_move(self, e):
        if self.sel["drag"]:
            self.sel["p1"] = (e.x, e.y)

    def _sel_up(self, e):
        if not self.sel["drag"]:
            return
        self.sel["drag"] = False
        x0, y0 = self.sel["p0"]
        w, h = abs(e.x - x0), abs(e.y - y0)
        if w < 12 or h < 12:
            self.sel["click"] = (e.x, e.y)     # a click, not a drag
        else:
            self.sel["box"] = (min(x0, e.x), min(y0, e.y), w, h)

    def _sel_clear(self, _e):
        self.sel["clear"] = True

    # ---------------- UI ----------------
    def build_controls(self):
        dpad = tk.Frame(self.left, bg="#1e1e1e")
        dpad.pack(pady=20)

        btn_style = {"width": 8, "height": 3}

        self.btn_up = tk.Button(dpad, text="UP", command=lambda: self.manual("UP"), **btn_style)
        self.btn_up.grid(row=0, column=1)

        self.btn_left = tk.Button(dpad, text="LEFT", command=lambda: self.manual("LEFT"), **btn_style)
        self.btn_left.grid(row=1, column=0)

        self.btn_stop = tk.Button(dpad, text="STOP", command=lambda: self.manual("CENTERED"), **btn_style)
        self.btn_stop.grid(row=1, column=1)

        self.btn_right = tk.Button(dpad, text="RIGHT", command=lambda: self.manual("RIGHT"), **btn_style)
        self.btn_right.grid(row=1, column=2)

        self.btn_down = tk.Button(dpad, text="DOWN", command=lambda: self.manual("DOWN"), **btn_style)
        self.btn_down.grid(row=2, column=1)

        fire_frame = tk.Frame(self.left, bg="#1e1e1e")
        fire_frame.pack(pady=20)

        fire_btn = tk.Button(fire_frame, text="FIRE", bg="#ff0000", fg="red", width=10, height=3)
        fire_btn.grid(row=0, column=0, padx=5)
        fire_btn.bind("<ButtonPress-1>", lambda e: self.fire_start())
        fire_btn.bind("<ButtonRelease-1>", lambda e: self.fire_stop())

        self.auto_btn = tk.Button(fire_frame, text="AUTO FIRE: OFF",
                                  command=self.toggle_auto_fire,
                                  fg="orange", width=14, height=3)
        self.auto_btn.grid(row=0, column=1, padx=5)

        self.track_btn = tk.Button(self.left, text="TRACKING: OFF", bg="gray", fg="blue",
                                   command=self.toggle_tracking, width=16, height=3)
        self.track_btn.pack(pady=10)

        self.mode_btn = tk.Button(self.left, text="MODE: PROFILE", fg="blue",
                                  command=self.toggle_lock_mode, width=16, height=2)
        self.mode_btn.pack(pady=4)

        auto_acquire_ready = self.detector is not None and self.detector.ready
        self.auto_acquire_btn = tk.Button(
            self.left,
            text="AUTO-ACQUIRE: OFF" if auto_acquire_ready else "AUTO-ACQUIRE: N/A",
            command=self.toggle_auto_acquire, fg="orange", width=16, height=2,
            state="normal" if auto_acquire_ready else "disabled")
        self.auto_acquire_btn.pack(pady=4)

        # ----------- LABELS -----------
        angle_frame = tk.Frame(self.left, bg="#1e1e1e")
        angle_frame.pack(pady=5)

        self.tgt_label = tk.Label(
            angle_frame, text="TGT  : 0°",
            fg="white", bg="#1e1e1e", font=("Arial", 12), anchor="w", width=15)
        self.tgt_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.mount_label = tk.Label(
            angle_frame, text="MOUNT : 0°",
            fg="white", bg="#1e1e1e", font=("Arial", 12), anchor="w", width=15)
        self.mount_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        self.distance_label = tk.Label(
            angle_frame, text="DIST  : ---",
            fg="white", bg="#1e1e1e", font=("Arial", 12), anchor="w", width=15)
        self.distance_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.source_label = tk.Label(
            angle_frame, text="SRC      : ---",
            fg="white", bg="#1e1e1e", font=("Arial", 12), anchor="w", width=15)
        self.source_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")

    def set_manual_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for btn in [self.btn_up, self.btn_down, self.btn_left, self.btn_right, self.btn_stop]:
            btn.config(state=state)

    # ---------------- ACTIONS ----------------
    def manual(self, cmd):
        if self.tracking_enabled:
            return
        send_command(self.esp, cmd)

    def fire_start(self):
        send_command(self.esp, "START_FIRE")
        self.rp_color = (0, 0, 255)

    def fire_stop(self):
        send_command(self.esp, "STOP_FIRE")
        self.rp_color = (0, 0, 0)

    def toggle_auto_fire(self):
        self.auto_fire_enabled = not self.auto_fire_enabled
        self.auto_btn.config(
            text=f"AUTO FIRE: {'ON' if self.auto_fire_enabled else 'OFF'}",
            fg="red" if self.auto_fire_enabled else "orange")

    def toggle_auto_acquire(self):
        if self.detector is None or not self.detector.ready:
            return
        self.auto_acquire_enabled = not self.auto_acquire_enabled
        if self.auto_acquire_enabled:
            self.detector.start()
        else:
            self.detector.stop()
        self.auto_acquire_btn.config(
            text=f"AUTO-ACQUIRE: {'ON' if self.auto_acquire_enabled else 'OFF'}",
            fg="red" if self.auto_acquire_enabled else "orange")

    def toggle_tracking(self):
        self.tracking_enabled = not self.tracking_enabled
        self.track_btn.config(
            text=f"TRACKING: {'ON' if self.tracking_enabled else 'OFF'}",
            fg="green" if self.tracking_enabled else "blue",
            bg="green" if self.tracking_enabled else "gray")
        self.set_manual_enabled(not self.tracking_enabled)
        if not self.tracking_enabled:
            send_command(self.esp, "CENTERED")
            self.last_command = "CENTERED"

    # --------- Polling idms data --------
    def idms_loop(self):
        while self.idms_running:
            try:
                result = get_IDMS_tgt_data(self.mount_latitude, self.mount_longitude)

                if result and isinstance(result, dict) and \
                        result.get("drone_distance") <= self.idms_tgt_dist_to_consider:
                    with self.idms_lock:
                        self.idms_data = {
                            "azimuth": result.get("azimuth"),
                            "drone_distance": result.get("drone_distance"),
                            "latitude": result.get("latitude"),
                            "longitude": result.get("longitude"),
                            "source": result.get("source"),
                            "raw": result,
                            "timestamp": time.time(),
                        }
                    self.idms_reported_en_drone = True
                else:
                    with self.idms_lock:
                        self.idms_data = {
                            "azimuth": None, "drone_distance": None,
                            "latitude": None, "longitude": None,
                            "source": None, "raw": None, "timestamp": None,
                        }
                    if self.idms_reported_en_drone:   # for repositioning the en drone
                        self.idms_reported_en_drone = False
                        self.droneDetectedTime = time.time()

                self.root.after(0, self.update_tgt_label)
            except Exception as e:
                print("IDMS error:", e)

            time.sleep(2)

    def update_tgt_label(self):
        with self.idms_lock:
            azimuth = self.idms_data.get("azimuth")
            source = self.idms_data.get("source")
            distance = self.idms_data.get("drone_distance")

        if azimuth is not None:
            self.tgt_label.config(text=f"TGT  : {float(azimuth):.2f}°", fg="white")
        else:
            self.tgt_label.config(text="TGT  : ---", fg="grey")

        if distance is not None:
            self.distance_label.config(text=f"DIST : {float(distance):.2f}", fg="white")
        else:
            self.distance_label.config(text="DIST : ---", fg="grey")

        if source is not None:
            self.source_label.config(text=f"SRC      : {source}", fg="white")
        else:
            self.source_label.config(text="SRC       : ---", fg="grey")

    def update_map_view(self):
        if not self.running:
            return
        with self.idms_lock:
            lat = self.idms_data.get("latitude")
            lon = self.idms_data.get("longitude")
        self.map_view.update_drone(lat, lon)
        self.root.after(1000, self.update_map_view)

    # ---------------- MAIN LOOP ----------------
    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_frame)
            return

        current_time = time.time()

        if self.detector is not None and self.auto_acquire_enabled:
            self.detector.submit_frame(frame)

        # -------- MOUNT ANGLE (GYRO) --------
        roll, yaw = get_roll_yaw()
        if yaw is not None:
            self.mount_label.config(text=f"MOUNT : {yaw:.2f}°")

        # -------- SELECTION EVENTS --------
        if self.sel["clear"]:
            self.sel["clear"] = False
            if self.tracker.locked:
                print("[LOCK] cleared")
            self.tracker.clear()
        if self.sel["box"] is not None:
            box, self.sel["box"] = self.sel["box"], None
            self.tracker.select(frame, clip_box(tuple(float(v) for v in box),
                                                self.frame_width, self.frame_height))
        if self.sel["click"] is not None:
            cx, cy = self.sel["click"]
            self.sel["click"] = None
            hit = estimate_object_box(frame, cx, cy, hint=70.0)
            if hit is None:
                hit = (cx - 35.0, cy - 35.0, 70.0, 70.0)
            self.tracker.select(frame, clip_box(hit, self.frame_width,
                                                self.frame_height))

        # Only ever originates a fresh lock while idle - never fights or
        # overrides a manual or already-auto-acquired lock.
        if self.detector is not None and self.auto_acquire_enabled \
                and self.tracker.state == "idle":
            auto_box = self.detector.get_confirmed_box()
            if auto_box is not None:
                print("[AUTO-ACQUIRE] locking on YOLO detection (%dx%d)" %
                      (auto_box[2], auto_box[3]))
                self.tracker.select(frame, auto_box)

        # -------- THE LOCK (replaces YOLO) --------
        st = self.tracker.update(frame, (), now=current_time)
        target_live = st.state in ("lock", "coast") and st.aim is not None
        if target_live:
            self.droneDetectedTime = current_time

        drone_distance = self.idms_data.get("drone_distance")

        if target_live and self.tracking_enabled:
            cx, cy = int(st.aim[0]), int(st.aim[1])
            movement = get_movement(cx, cy, self.center_x, self.center_y)
            zone = get_zone(cx, cy, self.center_x, self.center_y,
                            self.frame_width, self.frame_height)
            if st.state == "coast":
                # aiming at a guess: creep, do not slew
                zone = min(zone, COAST_ZONE_CAP)

            if zone <= 100:
                h, w = frame.shape[:2]
                fx, fy = w // 2, h // 2
                for x, y in [(0, fy), (w, fy), (fx, 0), (fx, h)]:
                    cv2.line(frame, (fx, fy), (x, y), self.rp_color, 2)
                cv2.circle(frame, (fx, fy), int(zone), self.rp_color, 2)

            if movement != self.last_command and \
                    (current_time - self.last_sent_time > COMMAND_DELAY):
                send_command(self.esp, movement, zone)
                self.last_command = movement
                self.last_sent_time = current_time

            # AUTO FIRE: legacy condition, plus the lock must actually BE
            # locked - never on a coasted prediction - and only in PROFILE
            # mode, where "locked" means the identity-checked target. In
            # blob mode "locked" merely means "a dark blob", which can be a
            # bird; the button is disabled there and this guard backs it up.
            if self.auto_fire_enabled and st.state == "lock" \
                    and self.lock_mode == "PROFILE":
                if movement == "CENTERED" or zone < 30:
                    if self.last_fire_command is None:
                        send_command(self.esp, "START_FIRE")
                        self.last_fire_command = "START_FIRE"
                        self.rp_color = (0, 0, 255)
                else:
                    self.rp_color = (0, 0, 0)
            elif self.last_fire_command is not None:
                send_command(self.esp, "STOP_FIRE")
                self.last_fire_command = None
                self.rp_color = (0, 0, 0)

        elif self.tracking_enabled and drone_distance is not None and \
                drone_distance <= self.idms_tgt_dist_to_consider:
            if roll is not None and yaw is not None:
                tgt_azimuth = self.idms_data.get("azimuth")
                tgt_azimuth = (tgt_azimuth + 180) % 360 - 180

                if not (tgt_azimuth - 10 < yaw < tgt_azimuth + 10) and (-180 < yaw < 180):
                    movement = "LEFT" if yaw > tgt_azimuth else "RIGHT"
                else:
                    movement = "CENTERED"

                if movement != self.last_command and \
                        (current_time - self.last_sent_time > COMMAND_DELAY):
                    send_command(self.esp, movement)
                    self.last_command = movement
                    self.last_sent_time = current_time

        elif self.tracking_enabled:
            if self.last_fire_command is not None:
                send_command(self.esp, "STOP_FIRE")
                self.last_fire_command = None

            diff = current_time - self.droneDetectedTime

            if 5 < diff < 15:   # fallback of mount to its default pose
                if roll is not None and yaw is not None:
                    if not (self.default_mount_roll - 5 < roll < self.default_mount_roll + 5):
                        movement = "DOWN" if roll > self.default_mount_roll else "UP"
                    elif not (self.default_mount_yaw - 10 < yaw < self.default_mount_yaw + 10):
                        movement = "LEFT" if yaw > self.default_mount_yaw else "RIGHT"
                    else:
                        movement = "CENTERED"

                    if movement != self.last_command and \
                            (current_time - self.last_sent_time > COMMAND_DELAY):
                        send_command(self.esp, movement)
                        self.last_command = movement
                        self.last_sent_time = current_time
            else:
                if self.last_command != "CENTERED":
                    send_command(self.esp, "CENTERED")
                    self.last_command = "CENTERED"

        # -------- OVERLAY --------
        col = STATE_COLORS.get(st.state, (200, 200, 200))
        if st.box is not None and st.state != "idle":
            x, y, w, h = [int(v) for v in st.box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
        if st.search_box is not None:
            x, y, w, h = [int(v) for v in st.search_box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 1)
        if self.sel["drag"] and self.sel["p0"] and self.sel["p1"]:
            cv2.rectangle(frame, self.sel["p0"], self.sel["p1"], (255, 255, 0), 1)
        if st.state != "idle":
            head = {"lock": "LOCKED", "coast": "HIDDEN - predicting",
                    "search": "LOST - searching"}.get(st.state, st.state)
            cv2.putText(frame, "%s %s  %.2f" % (self.lock_mode, head, st.score),
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            if st.note:
                cv2.putText(frame, st.note, (8, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)
        else:
            cv2.putText(frame, "drag a box round the target", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(15, self.update_frame)

    # ---------------- CLEANUP ----------------
    def stop(self):
        self.running = False
        self.idms_running = False
        if self.detector is not None:
            self.detector.stop()
        self.cap.release()
        if self.esp:
            self.esp.close()


# ---------------- RUN ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)

    def on_close():
        send_command(app.esp, "CENTERED")
        app.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
