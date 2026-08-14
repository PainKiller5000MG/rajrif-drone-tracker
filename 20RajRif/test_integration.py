"""End-to-end check of the tracker loop with a fake camera and fake serial.

Runs the real run_tracker() - selection handling, control law, mount protocol
and overlay - against synthetic frames, and asserts the mount is commanded
toward the target with the right sign on both axes.

    py -3.11 test_integration.py
"""

import sys
import queue
import threading
import time

import cv2
import numpy as np

import test_tracking as T

FRAME_W, FRAME_H = 640, 360


class FakeCap:
    """Synthetic camera: the target sits well right of and above centre."""

    def __init__(self, *_a, **_k):
        self.patch = T.make_object((40, 40, 220), 11)
        self.i = 0

    def isOpened(self):
        return True

    def set(self, *_a):
        return True

    def read(self):
        self.i += 1
        time.sleep(0.005)
        return True, T.render([(self.patch, self.target[0], self.target[1])],
                              shake=T.shake_at(self.i, 2.0))

    def release(self):
        pass


class FakeLink:
    def __init__(self, *_a, **_k):
        self.lines = []
        self.lock = threading.Lock()

    def write_line(self, text):
        with self.lock:
            self.lines.append(text)

    def drain(self, log_lines=True):
        return []

    def close(self):
        pass


def check_standalone_jog():
    """The jog pad must drive the mount with the tracker stopped.

    It used to only feed a dict the tracker loop read, so with nothing running
    the buttons did nothing at all - and said nothing either.
    """
    import os
    import tempfile
    import tkinter as tk

    import mount
    import tracker_gui

    link = FakeLink()
    mount.SerialLink = lambda *a, **k: link

    # Never touch the user's real settings file - these tests would overwrite
    # the duty floors and gains they just spent time tuning.
    real_settings = tracker_gui.SETTINGS_FILE
    tracker_gui.SETTINGS_FILE = os.path.join(tempfile.mkdtemp(), "settings.json")

    root = tk.Tk()
    root.withdraw()
    gui = tracker_gui.TrackerGUI(root)
    gui.port_var.set("FAKECOM")
    gui.mount_var.set("ESP32 DC (BTS7960)")

    try:
        gui._jog_press((1.0, 0.0))            # hold RIGHT
        deadline = time.time() + 4
        while time.time() < deadline:
            root.update()
            with link.lock:
                drives = [l for l in link.lines if l.startswith("M,")
                          and int(l.split(",")[1]) != 0]
            if drives:
                break
            time.sleep(0.05)
        assert drives, "jog sent nothing with the tracker stopped: %s" % link.lines
        assert all(int(l.split(",")[1]) > 0 for l in drives), \
            "RIGHT drove the wrong way: %s" % drives[:3]

        gui.ui["jog"] = (0.0, 0.0)            # release
        time.sleep(0.4)
        with link.lock:
            mark = len(link.lines)
        time.sleep(0.4)
        with link.lock:
            after = [l for l in link.lines[mark:] if l.startswith("M,")]
        assert all(l == "M,0,0" for l in after), \
            "kept driving after release: %s" % after

        # and it must hand the port back rather than hold it forever
        deadline = time.time() + 5
        while gui._jog_thread.is_alive() and time.time() < deadline:
            time.sleep(0.1)
        assert not gui._jog_thread.is_alive(), "jog never released the port"
        print("   jog: drives with the tracker stopped, stops on release, "
              "releases the port")

        # Test Motors ran on a worker thread too and read Tk variables there,
        # which raises "main thread is not in main loop" and produced silence.
        with link.lock:
            link.lines.clear()
        gui.test_motors()
        deadline = time.time() + 14
        while gui.test_thread is not None and gui.test_thread.is_alive() \
                and time.time() < deadline:
            root.update()
            time.sleep(0.05)
        with link.lock:
            sent = list(link.lines)
        msgs = []
        try:
            while True:
                msgs.append(gui.log_queue.get_nowait())
        except queue.Empty:
            pass
        assert not any("ERROR" in m or "main loop" in m for m in msgs), \
            "motor test errored: %s" % [m for m in msgs if "ERROR" in m]
        moves = [l for l in sent if l.startswith("M,") and l != "M,0,0"]
        assert any(int(l.split(",")[1]) > 0 for l in moves), "no pan-right step"
        assert any(int(l.split(",")[1]) < 0 for l in moves), "no pan-left step"
        assert any(int(l.split(",")[2]) > 0 for l in moves), "no tilt-up step"
        assert sent[-1] in ("S", "M,0,0"), "test did not stop the mount: %s" % sent[-1]
        print("   test motors: pan both ways, tilt both ways, then stop")
    finally:
        gui._stop_manual_jog()
        root.destroy()
        tracker_gui.SETTINGS_FILE = real_settings
    return True


def check_port_persistence():
    """The chosen COM port must survive a restart - even one with the mount
    unplugged. refresh_ports() used to overwrite it with whatever was
    connected, so a single run without the board lost the setting."""
    import json
    import os
    import tempfile
    import tkinter as tk

    import serial.tools.list_ports as lp

    import tracker_gui

    class FakePort:
        def __init__(self, device, description=""):
            self.device = device
            self.description = description

    tmp = os.path.join(tempfile.mkdtemp(), "settings.json")
    real_settings, real_comports = tracker_gui.SETTINGS_FILE, lp.comports
    tracker_gui.SETTINGS_FILE = tmp
    try:
        # first run, mount connected: it should pick the CP210x, not COM1
        lp.comports = lambda: [FakePort("COM1", "Some other device"),
                               FakePort("COM11", "Silicon Labs CP210x UART Bridge")]
        root = tk.Tk(); root.withdraw()
        gui = tracker_gui.TrackerGUI(root)
        assert gui.port_var.get() == "COM11", \
            "did not pick the USB-serial bridge: %s" % gui.port_var.get()
        gui.port_var.set("COM11")
        gui._save_settings()
        root.destroy()
        assert json.load(open(tmp))["port"] == "COM11", "port was not saved"

        # restart with the mount UNPLUGGED
        lp.comports = lambda: [FakePort("COM1", "Some other device")]
        root = tk.Tk(); root.withdraw()
        gui = tracker_gui.TrackerGUI(root)
        assert gui.port_var.get() == "COM11", \
            "unplugged restart clobbered the saved port: %s" % gui.port_var.get()
        assert "COM11" in gui.port_combo["values"], "saved port dropped from list"
        gui._save_settings()
        root.destroy()
        assert json.load(open(tmp))["port"] == "COM11", \
            "saved file was overwritten while the board was unplugged"

        # plug it back in: still selected, no duplicate entry
        lp.comports = lambda: [FakePort("COM1", ""), FakePort("COM11", "CP210x")]
        root = tk.Tk(); root.withdraw()
        gui = tracker_gui.TrackerGUI(root)
        assert gui.port_var.get() == "COM11", "lost the port on reconnect"
        assert list(gui.port_combo["values"]).count("COM11") == 1, \
            "duplicate port entry: %s" % (gui.port_combo["values"],)
        root.destroy()
        print("   settings: COM port survives a restart with the board "
              "unplugged, and is auto-detected on first run")
    finally:
        tracker_gui.SETTINGS_FILE = real_settings
        lp.comports = real_comports
    return True


def main():
    T.W, T.H, T.BG = FRAME_W, FRAME_H, None
    FakeCap.target = (500.0, 90.0)      # right of centre and above it

    import mount
    import tracker_gui

    link = FakeLink()
    mount.SerialLink = lambda *a, **k: link

    captured = {}
    cv2.VideoCapture = lambda *a, **k: FakeCap()
    cv2.namedWindow = lambda *a, **k: None
    cv2.imshow = lambda *a, **k: None
    cv2.destroyAllWindows = lambda: None
    cv2.waitKey = lambda *_a: 255
    cv2.setMouseCallback = lambda _w, cb: captured.update(cb=cb)

    cfg = {
        "mount_type": "ESP32 DC (BTS7960)", "port": "FAKE", "baud": 115200,
        "cam": 0, "frame_w": FRAME_W, "frame_h": FRAME_H,
        "tracker": "CSRT", "stabilise": True, "track_thresh": 0.42,
        "relock_thresh": 0.60, "relock_frames": 3, "relock_margin": 0.08,
        "coast_s": 0.5, "give_up_s": 0.0, "max_jump_frac": 0.22,
        "aim_smooth": 0.45, "click_box": 70,
        "enable_y": True, "kp_pan": 1.6, "kp_tilt": 1.4, "kd_pan": 0.12,
        "kd_tilt": 0.10, "deadzone_x": 0.045, "deadzone_y": 0.055,
        "pan_dir": 1.0, "tilt_dir": 1.0,
        "min_duty": 340, "max_duty": 1023, "min_duty_tilt": 360,
        "max_duty_tilt": 1023, "pulse_below": 0.30, "jog_speed": 0.55,
        "max_speed_x": 12.0, "max_speed_y": 20.0, "scan_min": -180.0,
        "scan_max": 180.0, "tilt_center": 90.0, "tilt_min": 0.0, "tilt_max": 180.0,
    }

    logs = []
    stop = threading.Event()
    ui = {"jog": (0.0, 0.0), "fire": False, "clear": False}
    th = threading.Thread(target=tracker_gui.run_tracker,
                          args=(cfg, stop, ui, logs.append), daemon=True)
    th.start()

    # let the loop start, then drag a box round the target like the operator
    deadline = time.time() + 5
    while "cb" not in captured and time.time() < deadline:
        time.sleep(0.02)
    assert "cb" in captured, "video window never came up: %s" % logs
    time.sleep(0.3)

    cb = captured["cb"]
    tx, ty = FakeCap.target
    cb(cv2.EVENT_LBUTTONDOWN, int(tx - 32), int(ty - 32), 0, None)
    cb(cv2.EVENT_MOUSEMOVE, int(tx + 32), int(ty + 32), 0, None)
    cb(cv2.EVENT_LBUTTONUP, int(tx + 32), int(ty + 32), 0, None)

    time.sleep(1.5)
    with link.lock:
        drives = [l for l in link.lines if l.startswith("M,")]
    assert any("LOCKED" in l for l in logs), "never locked on: %s" % logs[-4:]
    assert drives, "no drive commands were sent"

    pans = [int(l.split(",")[1]) for l in drives]
    tilts = [int(l.split(",")[2]) for l in drives]
    moving = [(p, t) for p, t in zip(pans, tilts) if p or t]
    assert moving, "mount was never commanded to move"
    right = sum(1 for p, _ in moving if p > 0)
    up = sum(1 for _, t in moving if t > 0)
    print("   %d drive commands, %d/%d pan right, %d/%d tilt up"
          % (len(drives), right, len(moving), up, len(moving)))
    assert right > 0.9 * len([1 for p, _ in moving if p]), \
        "target is right of centre but the mount was told to pan left"
    assert up > 0.9 * len([1 for _, t in moving if t]), \
        "target is above centre but the mount was told to tilt down"
    assert max(abs(p) for p in pans) <= 1023, "duty out of range"

    # jog must override tracking
    with link.lock:
        mark = len(link.lines)
    ui["jog"] = (-1.0, 0.0)
    time.sleep(0.4)
    with link.lock:
        recent = [l for l in link.lines[mark + 1:] if l.startswith("M,")]
    assert recent and all(int(l.split(",")[1]) < 0 for l in recent), \
        "manual jog did not take over: %s" % recent
    ui["jog"] = (0.0, 0.0)

    # clearing the lock must stop the mount
    time.sleep(0.2)
    with link.lock:
        mark = len(link.lines)
    ui["clear"] = True
    time.sleep(0.6)
    with link.lock:
        tail = [l for l in link.lines[mark + 1:] if l.startswith(("M,", "S"))]
    assert tail and all(l in ("S", "M,0,0") for l in tail), \
        "mount kept driving after the lock was cleared: %s" % tail

    stop.set()
    th.join(timeout=5)
    print("   integration: select -> track -> jog -> clear all behave")

    check_standalone_jog()
    check_port_persistence()
    print("\n3/3 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
