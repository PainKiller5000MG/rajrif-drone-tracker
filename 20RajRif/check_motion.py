"""Measure the duty floor with the camera instead of the operator's eyes.

`find_min_duty.py` asks "did it turn?" after every step. The camera is bolted
to the mount, so it can answer that itself: if the mount turns, the *whole
image* shifts, and phase correlation on a thumbnail measures that shift to a
fraction of a pixel. Same measurement, no human in the loop, and it produces a
number rather than a yes.

    py -3.11 check_motion.py                  COM11, camera 0
    py -3.11 check_motion.py COM11 0          explicit
    py -3.11 check_motion.py COM11 0 --pan    only the pan axis

What it prints per direction is the lowest duty that moved the image further
than the noise floor, and how fast it moved at each step - px/s per duty is the
open-loop equivalent of Scan360's degrees-per-pixel calibration.

The mount must be clear and the GUI must be closed (it holds the camera).
Directions are tested in opposing pairs so the rig ends up roughly where it
started rather than walking into an end stop.
"""

import os
import sys
import time

# Must precede the cv2 import or opening the camera takes ~40 s.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np
import serial

# (label, GPIO, the GUI field it feeds) - opposing pairs, so the rig comes back
PAN = [("PAN RIGHT", 32, "min_duty"), ("PAN LEFT", 33, "min_duty_neg")]
TILT = [("TILT UP", 26, "min_duty_tilt"), ("TILT DOWN", 25, "min_duty_tilt_neg")]

LADDER = [200, 240, 280, 330, 380, 440, 520, 620, 750, 900, 1023]
PULSE_MS = 400
SETTLE_S = 0.6
THUMB_W = 320


def thumb(frame):
    """Grey thumbnail, windowed. Phase correlation needs float and a window
    or the frame edges dominate the peak."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = max(1, int(g.shape[0] * THUMB_W / g.shape[1]))
    g = cv2.resize(g, (THUMB_W, h))
    return np.float32(g)


def flush(cap, n=4):
    """Drop queued frames. MSMF buffers, so without this the 'before' frame is
    already several frames old and the first measurement is nonsense."""
    for _ in range(n):
        cap.read()


def shift_between(a, b):
    (dx, dy), _resp = cv2.phaseCorrelate(a, b)
    return float(np.hypot(dx, dy))


def measure_noise(cap, win):
    """How much the image moves when nothing is driving it: sensor noise,
    building vibration, rolling shutter. The floor has to clear this."""
    flush(cap)
    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("camera returned no frame")
    prev = thumb(prev)
    worst = 0.0
    for _ in range(12):
        ok, f = cap.read()
        if not ok:
            continue
        t = thumb(f)
        worst = max(worst, shift_between(prev, t))
        prev = t
    return worst


def pulse_and_measure(ser, cap, pin, duty):
    """Drive one direction briefly; return (px moved, px/s) in thumbnail px.

    `T` is used rather than `M` because it is not watchdog limited, so the
    pulse really lasts PULSE_MS instead of being cut at 600 ms of silence.
    """
    flush(cap)
    ok, before = cap.read()
    if not ok:
        return 0.0, 0.0
    before = thumb(before)

    ser.reset_input_buffer()
    ser.write(("T,%d,%d,%d\n" % (pin, duty, PULSE_MS)).encode())

    t0 = time.time()
    total, prev = 0.0, before
    while time.time() - t0 < (PULSE_MS / 1000.0):
        ok, f = cap.read()
        if not ok:
            continue
        cur = thumb(f)
        total += shift_between(prev, cur)      # sum, so a reversal still counts
        prev = cur
    elapsed = time.time() - t0
    time.sleep(SETTLE_S)
    ser.read(200)
    return total, total / max(1e-3, elapsed)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else "COM11"
    cam = 0
    for a in sys.argv[2:]:
        if a.isdigit():
            cam = int(a)
    axes = PAN + TILT
    if "--pan" in sys.argv:
        axes = PAN
    if "--tilt" in sys.argv:
        axes = TILT
    # Finding the floor is only half the question. The other half is how fast
    # the mount goes once it is above it: a floor of 280 is useless if 1023
    # still cannot keep up with a person walking. --sweep runs the whole ladder
    # instead of stopping at the first movement, and alternates the two
    # directions of each axis at every rung so the rig stays put.
    sweep = "--sweep" in sys.argv

    print("Opening %s ..." % port)
    ser = serial.Serial(port, 115200, timeout=0.5)
    time.sleep(2.2)                      # the ESP32 resets when the port opens
    ser.reset_input_buffer()
    ser.write(b"V\n")
    time.sleep(0.4)
    ver = ser.read(200).decode(errors="ignore").strip()
    print("Firmware: %s" % (ver or "(no reply)"))
    if "RAJRIF" not in ver.upper():
        print("That does not look like the mount firmware. Stopping.")
        ser.close()
        return 1

    print("Opening camera %d ..." % cam)
    cap = cv2.VideoCapture(cam, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print("Could not open the camera. Is the GUI still running?")
        ser.close()
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    time.sleep(1.0)

    try:
        noise = measure_noise(cap, None)
        # Generous: three times the worst still frame, and never under half a
        # thumbnail pixel. A floor that only just clears noise is not a floor.
        threshold = max(3.0 * noise, 0.5)
        print("\nStill-image noise: %.2f px worst frame-to-frame."
              "  Calling it motion above %.2f px." % (noise, threshold))
        print("Mount must be clear. Each step drives for %d ms.\n" % PULSE_MS)

        results = {}
        if sweep:
            print("%-6s | %s" % ("duty", " | ".join("%-22s" % n for n, _p, _f in axes)))
            print("-" * (8 + 25 * len(axes)))
            curve = {f: [] for _n, _p, f in axes}
            for duty in LADDER:
                cells = []
                for name, pin, field in axes:
                    moved, rate = pulse_and_measure(ser, cap, pin, duty)
                    curve[field].append((duty, moved, rate))
                    cells.append("%6.1f px %6.0f px/s" % (moved, rate))
                print("%-6d | %s" % (duty, " | ".join("%-22s" % c for c in cells)))
            print()
            for name, pin, field in axes:
                # A real floor clears the noise several times over, not by a
                # hair - a 0.6 px reading next to a 0.03 px noise floor is
                # backlash taking up or the rig ringing from the last pulse.
                solid = [d for d, m, _r in curve[field] if m > max(2.0, 6.0 * noise)]
                floor = solid[0] if solid else None
                top = max(r for _d, _m, r in curve[field])
                results[field] = min(1023, int(floor * 1.15)) if floor else None
                print("%-12s floor %s   top speed %.0f px/s"
                      % (name, floor if floor else "NONE", top))
            print()
        for name, pin, field in ([] if sweep else axes):
            print("=== %s  (GPIO %d) ===" % (name, pin))
            floor = None
            for duty in LADDER:
                moved, rate = pulse_and_measure(ser, cap, pin, duty)
                verdict = "MOVED" if moved > threshold else "-"
                print("  duty %4d  ->  %6.1f px  (%5.0f px/s)   %s"
                      % (duty, moved, rate, verdict))
                if moved > threshold:
                    floor = duty
                    break
            if floor is None:
                print("  *** never moved, even at %d. That is hardware, not a"
                      " setting:" % LADDER[-1])
                print("      check that half-bridge, its enable pin, the PWM"
                      " wire to GPIO %d," % pin)
                print("      the motor supply, and the PWM frequency"
                      " (send FREQ,5000).")
                results[field] = None
            else:
                # headroom, so the floor still works on a cold motor and at the
                # stiff end of the travel rather than only where it sits now
                results[field] = min(1023, int(floor * 1.15))
                print("  lowest duty that moved the image: %d  ->  %s = %d"
                      % (floor, field, results[field]))
            print()
    except KeyboardInterrupt:
        print("\nStopped early.")
        results = locals().get("results", {})
    finally:
        try:
            ser.write(b"S\n")
            time.sleep(0.2)
        except Exception:
            pass
        cap.release()
        ser.close()

    print("---------------- put these in the GUI ----------------")
    labels = {"min_duty": "Min duty pan right", "min_duty_neg": "Min duty pan left",
              "min_duty_tilt": "Min duty tilt up",
              "min_duty_tilt_neg": "Min duty tilt down"}
    for _n, _p, field in axes:
        v = results.get(field)
        print("  %-22s %s" % (labels[field] + ":", v if v else "(not measured)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
