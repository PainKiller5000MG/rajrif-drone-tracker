"""Measure the lowest PWM duty each direction actually turns at.

These are the most important numbers for smooth tracking. Below the floor the
motor only whistles; far above it the mount slams past the target and hunts.
The tracker lifts every non-zero demand to the floor and pulses it for anything
smaller, so getting these right is what makes a lock sit still.

All four directions are measured separately, because a mount is rarely
symmetric: tilt fights gravity going up, and a stiff slew ring or a tired
half-bridge can make one pan direction need noticeably more to break loose.

    py -3.11 find_min_duty.py            uses COM11
    py -3.11 find_min_duty.py COM5
"""

import sys
import time

import serial

# (label, pin, GUI field it feeds)
AXES = [
    ("PAN RIGHT", 32, "Min duty pan right"),
    ("PAN LEFT", 33, "Min duty pan left"),
    ("TILT UP", 26, "Min duty tilt up"),
    ("TILT DOWN", 25, "Min duty tilt down"),
]
LADDER = [1023, 900, 750, 620, 520, 440, 380, 330, 280, 240, 200, 160]
PULSE_MS = 1200


def ask(question):
    while True:
        a = input(question + " [y/n/q] ").strip().lower()
        if a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False
        if a in ("q", "quit"):
            raise KeyboardInterrupt


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM11"
    print("Opening %s..." % port)
    ser = serial.Serial(port, 115200, timeout=0.5)
    time.sleep(2.2)
    ser.reset_input_buffer()
    ser.write(b"V\n")
    time.sleep(0.4)
    ver = ser.read(200).decode(errors="ignore").strip()
    print("Firmware: %s" % (ver or "(no reply - is this the mount?)"))
    if "RAJRIF" not in ver:
        print("That does not look like the mount firmware. Stopping.")
        ser.close()
        return 1

    print("\nEach step drives one direction for about a second.")
    input("Make sure the mount is clear, then press Enter. ")

    results = {}
    try:
        for name, pin, field in AXES:
            print("\n=== %s  (GPIO %d) ===" % (name, pin))
            last_moved = None
            for duty in LADDER:
                ser.reset_input_buffer()
                ser.write(("T,%d,%d,%d\n" % (pin, duty, PULSE_MS)).encode())
                print("  duty %4d ..." % duty, end="", flush=True)
                time.sleep(PULSE_MS / 1000.0 + 0.6)
                ser.read(200)
                print(" done.")
                if ask("  did it turn?"):
                    last_moved = duty
                else:
                    break
            if last_moved is None:
                print("  *** %s never turned, even at %d." % (name, LADDER[0]))
                print("      That is a hardware fault, not a setting: check "
                      "that half-bridge,")
                print("      its enable pin and the PWM wire to GPIO %d." % pin)
                results[field] = None
            else:
                # headroom, so the floor still works on a cold motor and at the
                # stiff end of the travel rather than only where it sits now
                results[field] = min(1023, int(last_moved * 1.15))
                print("  lowest that turned: %d  ->  %s = %d"
                      % (last_moved, field, results[field]))
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        ser.write(b"S\n")
        time.sleep(0.2)
        ser.close()

    print("\n---------------- put these in the GUI ----------------")
    for _n, _p, field in AXES:
        v = results.get(field)
        print("  %-20s %s" % (field + ":", v if v else "(not measured)"))
    print("\nIf the mount still creeps past the target afterwards, lower Kp;")
    print("if it shivers when centred, raise the Deadzone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
