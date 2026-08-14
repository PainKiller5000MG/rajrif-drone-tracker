import tkinter as tk
import cv2
from PIL import Image, ImageTk
import time
import cvzone
from ultralytics import YOLO
import serial
import sys
import os

import threading
import math
import os
import sys

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


# ---------------- PATH ----------------
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except:
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
        new_min=350

        if mount_config["motor_speed_min"] is not None:
            new_min = mount_config["motor_speed_min"]

        old_min, old_max = 1, 100
        new_max = 1000
        speed = new_min + (zone - old_min) * (new_max - new_min) / (old_max - old_min)
        motor_speed = str(int(speed))


        # ---------------- SEND ----------------
        cmd= command + "," + motor_speed
        print(cmd, flush=True)
        esp.write((cmd + "\n").encode())

    except:
        pass


# ---------------- CAMERA ----------------
def init_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not accessible")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return cap


# ---------------- MODEL ----------------
def load_model():
    model_path = resource_path("best.pt")

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        sys.exit(1)

    print("Loading model...")
    return YOLO(model_path)


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
    # Distance from center
    dist = math.sqrt((cx - center_x)**2 + (cy - center_y)**2)

    # Max possible distance (corner of frame)
    max_dist = math.sqrt(center_x**2 + center_y**2)

    # Normalize (0 → center, 1 → corner)
    norm = dist / max_dist

    return norm * 100

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
        self.model = load_model()

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2

        self.rp_color = (0, 0, 0)

        self.last_command = None
        self.last_fire_command = None
        self.last_sent_time = 0
        self.droneDetectedTime = time.time()

        # THREAD DATA
        self.frame_for_detection = None
        self.last_box = None
        self.last_detection_time = 0
        self.lock = threading.Lock()

        self.build_controls()

        self.set_manual_enabled(True)
        self.track_btn.config(text="TRACKING: OFF", bg="gray")

        self.running = True

        threading.Thread(target=self.detection_loop, daemon=True).start()

        self.idms_data = {
            "azimuth": None,
            "drone_distance": None,
            "source": None,
            "latitude": None,
            "longitude": None,
            "raw": None,
            "timestamp": None
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

        #For Map View
        self.map_view = DroneTrackerMap(
            self.root,
            self.mount_latitude,
            self.mount_longitude
        )

        self.update_frame()
        self.update_map_view()

    # ---------------- YOLO THREAD ----------------
    def detection_loop(self):
        while self.running:
            if not self.tracking_enabled:
                time.sleep(0.05)
                continue

            frame = None

            with self.lock:
                if self.frame_for_detection is not None:
                    frame = self.frame_for_detection.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            results = self.model(frame, conf=0.7, verbose=False)

            best_box = None
            max_area = 0

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)

                    if area > max_area:
                        max_area = area
                        best_box = (x1, y1, x2, y2)

            with self.lock:
                if best_box:
                    self.last_box = best_box
                    self.last_detection_time = time.time()

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

        # ----------- LABELS -----------
        angle_frame = tk.Frame(self.left, bg="#1e1e1e")
        angle_frame.pack(pady=5)

        # Row 0 → TGT | MOUNT
        self.tgt_label = tk.Label(
            angle_frame, text="TGT  : 0°",
            fg="white", bg="#1e1e1e", font=("Arial", 12),
            anchor="w", width=15  # left align + fixed width
        )
        self.tgt_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.mount_label = tk.Label(
            angle_frame, text="MOUNT : 0°",
            fg="white", bg="#1e1e1e", font=("Arial", 12),
            anchor="w", width=15
        )
        self.mount_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        # Row 1 → DIST | SRC
        self.distance_label = tk.Label(
            angle_frame, text="DIST  : ---",
            fg="white", bg="#1e1e1e", font=("Arial", 12),
            anchor="w", width=15
        )
        self.distance_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.source_label = tk.Label(
            angle_frame, text="SRC      : ---",
            fg="white", bg="#1e1e1e", font=("Arial", 12),
            anchor="w", width=15
        )
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
            fg="red" if self.auto_fire_enabled else "orange"
        )

    def toggle_tracking(self):
        self.tracking_enabled = not self.tracking_enabled

        self.track_btn.config(
            text=f"TRACKING: {'ON' if self.tracking_enabled else 'OFF'}",
            fg="green" if self.tracking_enabled else "blue",
            bg="green" if self.tracking_enabled else "gray"
        )

        self.set_manual_enabled(not self.tracking_enabled)

    #---------Polling idms data--------
    def idms_loop(self):
        while self.idms_running:
            try:
                result = get_IDMS_tgt_data(self.mount_latitude,self.mount_longitude)

                if result and isinstance(result, dict) and result.get("drone_distance") <=self.idms_tgt_dist_to_consider:
                    with self.idms_lock:
                        self.idms_data = {
                            "azimuth": result.get("azimuth"),
                            "drone_distance": result.get("drone_distance"),
                            "latitude": result.get("latitude"),
                            "longitude": result.get("longitude"),
                            "source": result.get("source"),
                            "raw": result,
                            "timestamp": time.time()
                        }
                    self.idms_reported_en_drone=True
                else:
                    with self.idms_lock:
                        self.idms_data = {
                            "azimuth": None,
                            "drone_distance": None,
                            "latitude": None,
                            "longitude": None,
                            "source": None,
                            "raw": None,
                            "timestamp": None
                        }
                    if self.idms_reported_en_drone: #For Repositioning the en drone
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

        # --- AZIMUTH ---
        if azimuth is not None:
            self.tgt_label.config(text=f"TGT  : {float(azimuth):.2f}°",fg="white")
        else:
            self.tgt_label.config(text="TGT  : ---",fg="grey")

        # --- DISTANCE ---
        if distance is not None:
            self.distance_label.config(text=f"DIST : {float(distance):.2f}",fg="white")
        else:
            self.distance_label.config(text="DIST : ---",fg="grey")

        # --- SOURCE ---
        if source is not None:
            self.source_label.config(text=f"SRC      : {source}",fg="white")
        else:
            self.source_label.config(text="SRC       : ---",fg="grey")

    def update_map_view(self):
        if not self.running:
            return

        with self.idms_lock:
            lat = self.idms_data.get("latitude")
            lon = self.idms_data.get("longitude")

        self.map_view.update_drone(lat, lon)

        self.root.after(1000, self.update_map_view)  # update every 1 sec

    # ---------------- MAIN LOOP ----------------
    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_frame)
            return

        current_time = time.time()

        # -------- MOUNT ANGLE (GYRO) --------
        roll, yaw = get_roll_yaw()
        if yaw is not None:

            #Update Mount Label
            self.mount_label.config(text=f"MOUNT : {yaw:.2f}°")

        with self.lock:
            self.frame_for_detection = frame
            last_box = self.last_box
            last_detection_time = self.last_detection_time

        if current_time - last_detection_time > 0.5:
            last_box = None

        drone_distance = self.idms_data.get("drone_distance")# For using distance in else of next if Else

        if last_box:
            self.droneDetectedTime = current_time

            x1, y1, x2, y2 = last_box
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            movement = get_movement(cx, cy, self.center_x, self.center_y)

            zone = get_zone(cx, cy, self.center_x, self.center_y,
                            self.frame_width, self.frame_height)

            # Optional: display on frame
            '''
            cv2.putText(frame, f"{int(zone)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            '''

            if zone <= 100:
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                for x, y in [(0, cy), (w, cy), (cx, 0), (cx, h)]:
                    cv2.line(frame, (cx, cy), (x, y), self.rp_color, 2)
                cv2.circle(frame, (cx, cy), int(zone), self.rp_color, 2)

            if movement != self.last_command and (current_time - self.last_sent_time > COMMAND_DELAY):
                send_command(self.esp, movement, zone)
                self.last_command = movement
                self.last_sent_time = current_time

            #Setting Auto Fire
            if self.auto_fire_enabled:
                if movement == "CENTERED" or zone<30:
                    if self.last_fire_command is None:
                        send_command(self.esp, "START_FIRE")
                        self.last_fire_command = "START_FIRE"
                        self.rp_color=(0, 0, 255)
                else:
                    self.rp_color = (0, 0, 0)

            elif self.last_fire_command is not None:
                send_command(self.esp, "STOP_FIRE")
                self.last_fire_command = None
                self.rp_color=(0, 0, 0)

        elif drone_distance is not None and drone_distance <= self.idms_tgt_dist_to_consider:
            if roll is not None and yaw is not None:
                tgt_azimuth = self.idms_data.get("azimuth")
                tgt_azimuth = (tgt_azimuth + 180) % 360 - 180

                if not (tgt_azimuth-10 < yaw < tgt_azimuth+10) and (-180 < yaw < 180):
                    movement = "LEFT" if yaw > tgt_azimuth else "RIGHT"
                else:
                    movement = "CENTERED"

                if movement != self.last_command and (current_time - self.last_sent_time > COMMAND_DELAY):
                    send_command(self.esp, movement)
                    self.last_command = movement
                    self.last_sent_time = current_time

        else:
            if self.last_fire_command is not None:
                send_command(self.esp, "STOP_FIRE")
                self.last_fire_command = None

            diff = current_time - self.droneDetectedTime

            if 5 < diff < 15: # For fallback of mount
                if roll is not None and yaw is not None:
                    if not (self.default_mount_roll-5 < roll < self.default_mount_roll+5):
                        movement = "DOWN" if roll > self.default_mount_roll else "UP"
                    elif not (self.default_mount_yaw-10 < yaw < self.default_mount_yaw+10):
                        movement = "LEFT" if yaw > self.default_mount_yaw else "RIGHT"
                    else:
                        movement = "CENTERED"

                    if movement != self.last_command and (current_time - self.last_sent_time > COMMAND_DELAY):
                        send_command(self.esp, movement)
                        self.last_command = movement
                        self.last_sent_time = current_time
            else:
                if self.last_command != "CENTERED":
                    send_command(self.esp, "CENTERED")
                    self.last_command = "CENTERED"

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_frame)

    # ---------------- CLEANUP ----------------
    def stop(self):
        self.running = False
        self.idms_running = False
        self.cap.release()
        if self.esp:
            self.esp.close()


# ---------------- RUN ----------------
root = tk.Tk()
app = App(root)


def on_close():
    send_command(app.esp, "CENTERED")
    app.stop()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()