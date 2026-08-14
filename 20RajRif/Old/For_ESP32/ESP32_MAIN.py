"""
ESP32 (MicroPython) firmware for the pan/tilt mount - BTS7960 DC drives.

HARDWARE
    Motor 1  tilt  (UP / DOWN)     BTS7960 #1   RPWM 25  LPWM 26  R_EN 27  L_EN 14
    Motor 2  pan   (LEFT / RIGHT)  BTS7960 #2   RPWM 32  LPWM 33  R_EN 12  L_EN 13
    Trigger                        BTS7960 #3   RPWM 16  LPWM 17  R_EN  5  L_EN  4
    Status LED                     GPIO 2

SERIAL PROTOCOL  (115200 8N1, '\n' terminated ASCII)

    M,<pan>,<tilt>   proportional drive, each -1023..1023.
                     +pan = RIGHT, +tilt = UP, 0 = that axis stops.
                     The tracker streams this ~30x/s. Because the magnitude
                     is the speed, the mount can creep for a two-pixel
                     correction instead of slamming full speed and hunting
                     around the target - that is what makes a lock hold still.
    S                stop both axes (trigger untouched)
    F,1 / F,0        trigger on / off
    P                ping -> "OK"
    V                version string
    Legacy (old main.py): UP / DOWN / LEFT / RIGHT / UP LEFT / UP RIGHT /
                     DOWN LEFT / DOWN RIGHT / CENTERED / START_FIRE /
                     STOP_FIRE, each optionally followed by ",<speed>".

    NOTE the legacy speed suffix used to break this firmware: main.py sends
    "UP,600" but v1 compared the whole line against "UP", so every command
    fell through to the final else and stopped the motors. Commands are now
    split on ',' before matching, so the old GUI works too.

SAFETY
    * Comms watchdog: if no command arrives for CMD_TIMEOUT_MS the motors and
      the trigger are cut. A crashed or unplugged PC can no longer leave the
      mount slewing or firing.
    * Duty is ramped, never stepped, so the gearboxes and the supply do not
      see instant full-current reversals.
    * Any exception in the main loop stops everything before it propagates.
"""

from machine import Pin, PWM
import sys
import time

try:                      # MicroPython
    import uselect as select
except ImportError:       # CPython, for syntax checking on a PC
    import select

VERSION = "RAJRIF-MOUNT 2.0"

# ===============================
# CONSTANTS
# ===============================
PWM_FREQ = 20000          # above hearing, keeps the motors quiet
TRIG_FREQ = 1000
MAX_DUTY = 1023           # ESP32 PWM is 10-bit
CMD_TIMEOUT_MS = 600      # comms watchdog
RAMP_STEP = 70            # duty change allowed per tick
LOOP_MS = 2               # tick period -> 0..full in ~30 ms

# ===============================
# PIN GROUPS  (RPWM, LPWM, R_EN, L_EN)
# ===============================
TILT_PINS = (25, 26, 27, 14)   # motor 1
PAN_PINS = (32, 33, 12, 13)    # motor 2
TRIG_PINS = (16, 17, 5, 4)
LED_PIN = 2

# motor 1 wiring: "up" is the LPWM side, so the tilt axis is inverted
TILT_INVERT = True
PAN_INVERT = False


class Axis:
    """One BTS7960 half-bridge pair driven as a signed, ramped speed."""

    def __init__(self, pins, freq=PWM_FREQ, invert=False):
        rpwm, lpwm, r_en, l_en = pins
        self.pos = PWM(Pin(rpwm), freq=freq, duty=0)
        self.neg = PWM(Pin(lpwm), freq=freq, duty=0)
        self.en_r = Pin(r_en, Pin.OUT)
        self.en_l = Pin(l_en, Pin.OUT)
        self.en_r.value(1)
        self.en_l.value(1)
        self.sign = -1 if invert else 1
        self.target = 0
        self.cur = 0

    def set(self, duty):
        duty = int(duty) * self.sign
        if duty > MAX_DUTY:
            duty = MAX_DUTY
        elif duty < -MAX_DUTY:
            duty = -MAX_DUTY
        self.target = duty

    def stop(self):
        self.target = 0
        self.cur = 0
        self._write(0)

    def step(self):
        """Move one ramp step toward the target. Call every LOOP_MS."""
        if self.cur == self.target:
            return
        diff = self.target - self.cur
        if diff > RAMP_STEP:
            self.cur += RAMP_STEP
        elif diff < -RAMP_STEP:
            self.cur -= RAMP_STEP
        else:
            self.cur = self.target
        self._write(self.cur)

    def moving(self):
        return self.cur != 0

    def _write(self, duty):
        # Never energise both sides at once - that is a shoot-through on the
        # BTS7960. Always zero the opposite half first.
        if duty > 0:
            self.neg.duty(0)
            self.pos.duty(duty)
        elif duty < 0:
            self.pos.duty(0)
            self.neg.duty(-duty)
        else:
            self.pos.duty(0)
            self.neg.duty(0)


# Legacy direction words -> (pan sign, tilt sign)
LEGACY_DIRS = {
    "UP": (0, 1),
    "DOWN": (0, -1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "UP LEFT": (-1, 1),
    "UP RIGHT": (1, 1),
    "DOWN LEFT": (-1, -1),
    "DOWN RIGHT": (1, -1),
    "CENTERED": (0, 0),
    "STOP": (0, 0),
}


class LineReader:
    """Non-blocking newline splitter for sys.stdin.

    readline() blocks until a '\\n' actually lands, which would stall the
    ramp and the watchdog mid-command. This drains whatever bytes are in the
    buffer and only returns once a full line exists.
    """

    MAX_LEN = 96

    def __init__(self):
        self.buf = ""
        self.poll = select.poll()
        self.poll.register(sys.stdin, select.POLLIN)

    def readline(self):
        while self.poll.poll(0):
            ch = sys.stdin.read(1)
            if not ch:
                break
            if ch == "\n" or ch == "\r":
                if self.buf:
                    line = self.buf
                    self.buf = ""
                    return line
            else:
                self.buf += ch
                if len(self.buf) > self.MAX_LEN:
                    self.buf = ""       # junk / no terminator - drop it
        return None


def main():
    led = Pin(LED_PIN, Pin.OUT)
    pan = Axis(PAN_PINS, invert=PAN_INVERT)
    tilt = Axis(TILT_PINS, invert=TILT_INVERT)
    trigger = Axis(TRIG_PINS, freq=TRIG_FREQ)

    reader = LineReader()
    last_cmd = time.ticks_ms()
    stopped_by_watchdog = False

    def reply(text):
        sys.stdout.write(text + "\n")

    def handle(line):
        nonlocal stopped_by_watchdog
        parts = line.split(",")
        head = parts[0].strip().upper()
        stopped_by_watchdog = False

        if head == "M":
            # M,<pan>,<tilt> - the normal tracking command
            try:
                p = int(float(parts[1])) if len(parts) > 1 else 0
                t = int(float(parts[2])) if len(parts) > 2 else 0
            except (ValueError, IndexError):
                reply("ERR M")
                return
            pan.set(p)
            tilt.set(t)
            return

        if head in ("S", "STOP", "CENTERED"):
            pan.set(0)
            tilt.set(0)
            reply("STOPPED")
            return

        if head == "F":
            on = len(parts) > 1 and parts[1].strip() in ("1", "ON", "on")
            trigger.set(MAX_DUTY if on else 0)
            reply("FIRE " + ("1" if on else "0"))
            return

        if head == "P":
            reply("OK")
            return

        if head == "V":
            reply(VERSION)
            return

        if head == "START_FIRE":
            trigger.set(MAX_DUTY)
            reply("FIRE 1")
            return

        if head == "STOP_FIRE":
            trigger.set(0)
            reply("FIRE 0")
            return

        if head in LEGACY_DIRS:
            # Old protocol: full speed unless a ",<speed>" suffix is given.
            speed = MAX_DUTY
            if len(parts) > 1:
                try:
                    speed = max(0, min(MAX_DUTY, int(float(parts[1]))))
                except ValueError:
                    speed = MAX_DUTY
            px, ty = LEGACY_DIRS[head]
            pan.set(px * speed)
            tilt.set(ty * speed)
            reply("CMD " + head)
            return

        reply("Unknown: " + head)

    reply(VERSION + " ready")

    try:
        while True:
            while True:
                line = reader.readline()
                if line is None:
                    break
                last_cmd = time.ticks_ms()
                handle(line)

            # ---- comms watchdog ----
            if time.ticks_diff(time.ticks_ms(), last_cmd) > CMD_TIMEOUT_MS:
                if not stopped_by_watchdog:
                    pan.set(0)
                    tilt.set(0)
                    trigger.set(0)
                    stopped_by_watchdog = True

            pan.step()
            tilt.step()
            trigger.step()

            # LED on = mount at rest, off = something is driving
            led.value(0 if (pan.moving() or tilt.moving()) else 1)
            time.sleep_ms(LOOP_MS)
    finally:
        pan.stop()
        tilt.stop()
        trigger.stop()
        led.value(1)


main()
