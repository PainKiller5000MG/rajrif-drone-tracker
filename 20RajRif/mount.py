"""Mount backends.

The tracker produces one thing per axis: a speed fraction in [-1, 1]
(+pan = turn right, +tilt = look up). Everything hardware-specific lives
here, so the same control loop drives either rig:

  EspDcMount      ESP32 + BTS7960 DC gearmotors (Old/For_ESP32/ESP32_MAIN.py).
                  Open loop, no position feedback - the fraction becomes a
                  signed PWM duty.
  StepperMount    Arduino + TB6600 steppers. Closed loop on step count, takes
                  absolute angles, so the fraction is integrated into an angle.
"""

import time


class SerialLink:
    """Thin serial wrapper that degrades to a no-op instead of crashing."""

    def __init__(self, port, baud, log=print, settle=2.0):
        self.log = log
        self.ser = None
        self.dummy = True
        if not port or port.startswith("None"):
            log("Serial: dummy (no port selected)")
            return
        try:
            import serial
            log("Opening serial %s @ %s..." % (port, baud))
            self.ser = serial.Serial(port, baud, timeout=0.05)
            # ESP32/Arduino reset when the port opens; let the boot finish or
            # the first commands land in the bootloader instead of the sketch.
            time.sleep(settle)
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass
            self.dummy = False
            log("Serial ready.")
        except Exception as exc:
            log("ERROR opening serial: %s - continuing without motors." % exc)
            self.ser = None

    def write_line(self, text):
        if self.ser is None:
            return
        try:
            self.ser.write((text.strip() + "\n").encode("ascii"))
        except Exception as exc:
            self.log("Serial write failed: %s" % exc)
            self.ser = None

    def drain(self, log_lines=True):
        """Read whatever already arrived. Never blocks the control loop."""
        if self.ser is None:
            return []
        out = []
        try:
            while self.ser.in_waiting:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line:
                    out.append(line)
                    if log_lines:
                        self.log("[MOUNT] " + line)
        except Exception:
            pass
        return out

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None


class BaseMount:
    def drive(self, pan, tilt, dt):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def fire(self, on):
        pass

    def status(self):
        return ""

    def close(self):
        self.stop()


class EspDcMount(BaseMount):
    """BTS7960 DC mount. Speed fraction -> signed PWM duty.

    Two things make a DC gearmotor usable for fine aiming:

    * a stiction floor - below ~min_duty the motor just buzzes, so any
      non-zero demand is lifted to min_duty;
    * pulsing - a demand smaller than `pulse_below` would still move the
      mount at min_duty speed and overshoot, so instead min_duty is switched
      on for a matching fraction of a short period. The mount nudges a
      degree at a time and settles on the target rather than hunting.
    """

    def __init__(self, link, min_duty=320, max_duty=1023, pulse_below=0.30,
                 pulse_period=0.16, max_duty_tilt=None, min_duty_tilt=None,
                 min_duty_neg=None, min_duty_tilt_neg=None):
        self.link = link
        self.min_duty = int(min_duty)
        self.max_duty = int(max_duty)
        self.min_duty_tilt = int(min_duty_tilt if min_duty_tilt is not None else min_duty)
        self.max_duty_tilt = int(max_duty_tilt if max_duty_tilt is not None else max_duty)
        # A mount is rarely symmetric: tilt fights gravity going up, and a
        # stiff slew ring or a tired half-bridge can make one pan direction
        # need noticeably more to break loose. One floor per axis leaves the
        # weak direction buzzing without turning.
        self.min_duty_neg = int(min_duty_neg if min_duty_neg is not None else self.min_duty)
        self.min_duty_tilt_neg = int(
            min_duty_tilt_neg if min_duty_tilt_neg is not None else self.min_duty_tilt)
        self.pulse_below = float(pulse_below)
        self.pulse_period = float(pulse_period)
        self.last_sent = None
        self.last_send_time = 0.0
        self.last_fire = None
        self._t0 = time.time()

    def _duty(self, frac, min_pos, max_duty, min_neg=None):
        a = abs(float(frac))
        if a < 1e-3:
            return 0
        a = min(a, 1.0)
        min_duty = min_pos if frac > 0 else (min_pos if min_neg is None else min_neg)
        if a < self.pulse_below:
            # duty-cycle the stiction floor instead of driving continuously
            phase = ((time.time() - self._t0) % self.pulse_period) / self.pulse_period
            if phase > (a / self.pulse_below):
                return 0
            duty = min_duty
        else:
            span = max_duty - min_duty
            duty = min_duty + span * (a - self.pulse_below) / max(1e-6, 1.0 - self.pulse_below)
        return int(duty if frac > 0 else -duty)

    def drive(self, pan, tilt, dt):
        p = self._duty(pan, self.min_duty, self.max_duty, self.min_duty_neg)
        t = self._duty(tilt, self.min_duty_tilt, self.max_duty_tilt,
                       self.min_duty_tilt_neg)
        now = time.time()
        # Re-send even an unchanged command a few times a second: the firmware
        # watchdog cuts the motors after 600 ms of silence.
        if (p, t) != self.last_sent or (now - self.last_send_time) > 0.2:
            self.link.write_line("M,%d,%d" % (p, t))
            self.last_sent = (p, t)
            self.last_send_time = now

    def stop(self):
        if self.last_sent != (0, 0):
            self.link.write_line("S")
            self.last_sent = (0, 0)
            self.last_send_time = time.time()
        elif (time.time() - self.last_send_time) > 0.2:
            self.link.write_line("M,0,0")
            self.last_send_time = time.time()

    def fire(self, on):
        on = bool(on)
        if on == self.last_fire:
            return          # the loop calls this every frame; only send edges
        self.last_fire = on
        self.link.write_line("F,1" if on else "F,0")

    def status(self):
        if not self.last_sent:
            return "pan=0 tilt=0"
        return "pan=%d tilt=%d" % self.last_sent


class StepperMount(BaseMount):
    """TB6600 stepper mount driven by absolute angle commands.

    The firmware re-zeros at its current physical pose when the port opens,
    so we adopt x=0 / y=tilt_center and send nothing until the tracker asks
    for a move - Start must not jerk the turret.
    """

    SEND_THRESHOLD_DEG = 0.05

    def __init__(self, link, max_speed_x=12.0, max_speed_y=20.0,
                 pan_min=-180.0, pan_max=180.0,
                 tilt_min=0.0, tilt_max=180.0, tilt_center=90.0):
        self.link = link
        self.max_speed_x = float(max_speed_x)
        self.max_speed_y = float(max_speed_y)
        self.pan_min, self.pan_max = float(pan_min), float(pan_max)
        self.tilt_min, self.tilt_max = float(tilt_min), float(tilt_max)
        self.x = 0.0
        self.y = float(tilt_center)
        self.last_x = None
        self.last_y = None
        self.stopped = False

    def drive(self, pan, tilt, dt):
        dt = max(1e-3, min(dt, 0.25))
        if pan or tilt:
            self.stopped = False
        if pan:
            self.x = max(self.pan_min,
                         min(self.pan_max, self.x + pan * self.max_speed_x * dt))
            if self.last_x is None or abs(self.x - self.last_x) >= self.SEND_THRESHOLD_DEG:
                self.link.write_line("x=%.2f" % self.x)
                self.last_x = self.x
        if tilt:
            self.y = max(self.tilt_min,
                         min(self.tilt_max, self.y + tilt * self.max_speed_y * dt))
            if self.last_y is None or abs(self.y - self.last_y) >= self.SEND_THRESHOLD_DEG:
                self.link.write_line("y=%.2f" % self.y)
                self.last_y = self.y

    def stop(self):
        # Called every frame while there is no target - only send the edge,
        # or the link fills with "stop" and starves the real commands.
        if self.stopped:
            return
        self.link.write_line("stop")
        self.stopped = True
        self.last_x = self.last_y = None

    def status(self):
        return "x=%.2f y=%.2f" % (self.x, self.y)


def make_mount(kind, link, cfg):
    if kind.startswith("Stepper"):
        return StepperMount(
            link,
            max_speed_x=cfg.get("max_speed_x", 12.0),
            max_speed_y=cfg.get("max_speed_y", 20.0),
            pan_min=cfg.get("scan_min", -180.0),
            pan_max=cfg.get("scan_max", 180.0),
            tilt_min=cfg.get("tilt_min", 0.0),
            tilt_max=cfg.get("tilt_max", 180.0),
            tilt_center=cfg.get("tilt_center", 90.0),
        )
    return EspDcMount(
        link,
        min_duty=cfg.get("min_duty", 320),
        max_duty=cfg.get("max_duty", 1023),
        min_duty_tilt=cfg.get("min_duty_tilt", cfg.get("min_duty", 320)),
        max_duty_tilt=cfg.get("max_duty_tilt", cfg.get("max_duty", 1023)),
        min_duty_neg=cfg.get("min_duty_neg"),
        min_duty_tilt_neg=cfg.get("min_duty_tilt_neg"),
        pulse_below=cfg.get("pulse_below", 0.30),
    )
