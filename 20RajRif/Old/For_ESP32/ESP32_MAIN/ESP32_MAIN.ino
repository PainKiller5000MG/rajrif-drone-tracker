/*
  ESP32 firmware for the pan/tilt mount - BTS7960 DC drives.

  Same protocol as the MicroPython version in ../ESP32_MAIN.py; use whichever
  matches what is on your board. A board running this sketch answers "V" with
  the version string below, so you can always tell which one is loaded.

  HARDWARE
    Motor 1  tilt  (UP / DOWN)     BTS7960 #1   RPWM 25  LPWM 26  R_EN 27  L_EN 14
    Motor 2  pan   (LEFT / RIGHT)  BTS7960 #2   RPWM 32  LPWM 33  R_EN 12  L_EN 13
    Trigger                        BTS7960 #3   RPWM 16  LPWM 17  R_EN  5  L_EN  4
    Status LED                     GPIO 2  (lit = mount at rest)

  SERIAL PROTOCOL  (115200 8N1, '\n' terminated ASCII)
    M,<pan>,<tilt>   proportional drive, each -1023..1023.
                     +pan = RIGHT, +tilt = UP, 0 = that axis stops.
                     The tracker streams this ~30x/s. Because the magnitude is
                     the speed, the mount can creep for a two-pixel correction
                     instead of slamming full speed and hunting around the
                     target - that is what makes a lock hold still.
    S                stop both axes (trigger untouched)
    F,1 / F,0        trigger on / off
    P                ping -> "OK"
    V                version string
    Legacy (old main.py): UP / DOWN / LEFT / RIGHT / UP LEFT / UP RIGHT /
                     DOWN LEFT / DOWN RIGHT / CENTERED / START_FIRE /
                     STOP_FIRE, each optionally followed by ",<speed>".

  SAFETY
    * Comms watchdog: no command for CMD_TIMEOUT_MS and the motors and the
      trigger are cut, so a crashed or unplugged PC cannot leave the mount
      slewing or firing.
    * Duty is ramped, never stepped, so the gearboxes and the supply do not see
      instant full-current reversals.
    * The two halves of a bridge are never driven together.
*/

// The Arduino builder hoists auto-generated prototypes to the top of the
// sketch, above the class definition, so Axis has to be known before them.
class Axis;

static const char *VERSION = "RAJRIF-MOUNT 2.0 (arduino)";

// 20 kHz is inaudible but plenty of cheap BTS7960 modules cannot switch that
// fast - the input filter swallows it and the motor barely turns. 5 kHz whines
// slightly and always works. Change live with "FREQ,<hz>" while testing.
static uint32_t PWM_FREQ = 5000;
static const uint32_t TRIG_FREQ = 1000;
static const uint8_t PWM_BITS = 10;       // 0..1023, matches the protocol
static const int MAX_DUTY = 1023;
static const uint32_t CMD_TIMEOUT_MS = 600;
static const int RAMP_STEP = 70;          // duty change allowed per tick
static const uint32_t LOOP_MS = 2;        // 0 -> full in ~30 ms

static const uint8_t LED_PIN = 2;

// ---- PWM helpers: the ledc API changed in ESP32 core 3.x ----
static inline void pwmAttach(uint8_t pin, uint8_t ch, uint32_t freq) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  (void)ch;
  ledcAttach(pin, freq, PWM_BITS);
#else
  ledcSetup(ch, freq, PWM_BITS);
  ledcAttachPin(pin, ch);
#endif
}

static inline void pwmDetach(uint8_t pin, uint8_t ch) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  (void)ch;
  ledcDetach(pin);
#else
  (void)ch;
  ledcDetachPin(pin);
#endif
}

static inline void pwmWrite(uint8_t pin, uint8_t ch, uint32_t duty) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  (void)ch;
  ledcWrite(pin, duty);
#else
  (void)pin;
  ledcWrite(ch, duty);
#endif
}

// One BTS7960 half-bridge pair driven as a signed, ramped speed.
class Axis {
 public:
  Axis(uint8_t rpwm, uint8_t lpwm, uint8_t rEn, uint8_t lEn,
       uint8_t chPos, uint8_t chNeg, bool invert)
      : pinPos_(rpwm), pinNeg_(lpwm), enR_(rEn), enL_(lEn),
        chPos_(chPos), chNeg_(chNeg), sign_(invert ? -1 : 1),
        target_(0), cur_(0) {}

  void begin(uint32_t freq) {
    pinMode(enR_, OUTPUT);
    pinMode(enL_, OUTPUT);
    digitalWrite(enR_, HIGH);
    digitalWrite(enL_, HIGH);
    pwmAttach(pinPos_, chPos_, freq);
    pwmAttach(pinNeg_, chNeg_, freq);
    write(0);
  }

  void set(int duty) {
    duty *= sign_;
    if (duty > MAX_DUTY) duty = MAX_DUTY;
    else if (duty < -MAX_DUTY) duty = -MAX_DUTY;
    target_ = duty;
  }

  void stop() {
    target_ = 0;
    cur_ = 0;
    write(0);
  }

  void step() {  // one ramp step toward the target; call every LOOP_MS
    if (cur_ == target_) return;
    int diff = target_ - cur_;
    if (diff > RAMP_STEP) cur_ += RAMP_STEP;
    else if (diff < -RAMP_STEP) cur_ -= RAMP_STEP;
    else cur_ = target_;
    write(cur_);
  }

  bool moving() const { return cur_ != 0; }

  void reattach(uint32_t freq) {  // for the live FREQ command
    stop();
    pwmDetach(pinPos_, chPos_);
    pwmDetach(pinNeg_, chNeg_);
    pwmAttach(pinPos_, chPos_, freq);
    pwmAttach(pinNeg_, chNeg_, freq);
    write(0);
  }

  void enable() {  // re-assert the driver enables after a diagnostic sweep
    pinMode(enR_, OUTPUT);
    pinMode(enL_, OUTPUT);
    digitalWrite(enR_, HIGH);
    digitalWrite(enL_, HIGH);
  }

  bool ownsPin(uint8_t pin) const { return pin == pinPos_ || pin == pinNeg_; }
  void writePin(uint8_t pin, uint32_t duty) {
    if (pin == pinPos_) pwmWrite(pinPos_, chPos_, duty);
    else if (pin == pinNeg_) pwmWrite(pinNeg_, chNeg_, duty);
  }

 private:
  void write(int duty) {
    // Never energise both sides at once - that is a shoot-through on the
    // BTS7960. Always zero the opposite half first.
    if (duty > 0) {
      pwmWrite(pinNeg_, chNeg_, 0);
      pwmWrite(pinPos_, chPos_, duty);
    } else if (duty < 0) {
      pwmWrite(pinPos_, chPos_, 0);
      pwmWrite(pinNeg_, chNeg_, -duty);
    } else {
      pwmWrite(pinPos_, chPos_, 0);
      pwmWrite(pinNeg_, chNeg_, 0);
    }
  }

  uint8_t pinPos_, pinNeg_, enR_, enL_, chPos_, chNeg_;
  int sign_, target_, cur_;
};

// motor 1 wiring: "up" is the LPWM side, so the tilt axis is inverted
Axis tiltAxis(25, 26, 27, 14, 0, 1, true);
Axis panAxis(32, 33, 12, 13, 2, 3, false);
Axis trigger(16, 17, 5, 4, 4, 5, false);

// ===================== BRING-UP DIAGNOSTICS =====================
// The pin map above came from the old MicroPython file, which never actually
// drove anything (it compared the whole line against "UP", so every command
// with a speed suffix fell through to "stop"). If nothing moves, these find
// the real wiring over the serial port instead of by reflashing:
//
//   FREQ,<hz>            re-attach both axes at a new PWM frequency
//   ENALL,1 / ENALL,0    drive every safe GPIO high / low (satisfies unknown
//                        enable pins so a PWM test can actually turn a motor)
//   T,<pin>,<duty>,<ms>  pulse one GPIO, everything else left as-is
//   SCAN,<duty>          pulse each safe GPIO in turn, printing the number as
//                        it goes, with the others held high. Watch the mount:
//                        whichever pin number is printed when it twitches is
//                        an input of that driver.
//
// Excluded on purpose: 0/2/12/15 are boot straps (2 and 12 are still scanned
// because the current map uses them), 1 and 3 are the USB serial, 6-11 are the
// SPI flash - touching those crashes the chip - and 34-39 are input only.
static const uint8_t SAFE_PINS[] = {2, 4, 5, 12, 13, 14, 15, 16, 17, 18,
                                    19, 21, 22, 23, 25, 26, 27, 32, 33};
static const uint8_t SAFE_PIN_COUNT = sizeof(SAFE_PINS) / sizeof(SAFE_PINS[0]);

static uint32_t lastCmdMs = 0;
static uint32_t lastTickMs = 0;
static bool stoppedByWatchdog = false;
static char lineBuf[96];
static uint8_t lineLen = 0;

// Is this pin already one of the six axis PWM outputs? Re-attaching a live
// ledc pin upsets the timer, so those are written directly instead.
static Axis *axisOwning(uint8_t pin) {
  if (panAxis.ownsPin(pin)) return &panAxis;
  if (tiltAxis.ownsPin(pin)) return &tiltAxis;
  if (trigger.ownsPin(pin)) return &trigger;
  return nullptr;
}

static void pulsePin(uint8_t pin, int duty, uint32_t ms) {
  if (duty < 0) duty = 0;
  if (duty > MAX_DUTY) duty = MAX_DUTY;
  Axis *owner = axisOwning(pin);
  if (owner) {
    owner->writePin(pin, duty);
    delay(ms);
    owner->writePin(pin, 0);
  } else {
    pwmAttach(pin, 6, PWM_FREQ);
    pwmWrite(pin, 6, duty);
    delay(ms);
    pwmWrite(pin, 6, 0);
    pwmDetach(pin, 6);
    pinMode(pin, OUTPUT);
    digitalWrite(pin, LOW);
  }
}

static void enableAll(bool on) {
  for (uint8_t i = 0; i < SAFE_PIN_COUNT; ++i) {
    uint8_t pin = SAFE_PINS[i];
    if (axisOwning(pin)) continue;   // leave the PWM outputs alone
    pinMode(pin, OUTPUT);
    digitalWrite(pin, on ? HIGH : LOW);
  }
}

static void scanPins(int duty) {
  panAxis.stop();
  tiltAxis.stop();
  trigger.stop();
  Serial.println("SCAN start - watch the mount, note the pin it moves on");
  enableAll(true);
  for (uint8_t i = 0; i < SAFE_PIN_COUNT; ++i) {
    uint8_t pin = SAFE_PINS[i];
    if (trigger.ownsPin(pin)) {
      // Never sweep the trigger outputs - that would fire it for a second.
      // Test them deliberately with "T,<pin>,..." if you really mean to.
      Serial.print("PIN ");
      Serial.print(pin);
      Serial.println(" skipped (trigger output)");
      continue;
    }
    Serial.print("PIN ");
    Serial.println(pin);
    delay(400);            // time to look up before it moves
    pulsePin(pin, duty, 900);
    delay(700);
  }
  enableAll(false);
  panAxis.enable();       // enableAll(false) pulled the driver enables down
  tiltAxis.enable();
  trigger.enable();
  Serial.println("SCAN done");
}

static void handle(char *line) {
  // split on ',' into at most 4 fields
  char *field[4] = {line, nullptr, nullptr, nullptr};
  uint8_t n = 1;
  for (char *p = line; *p && n < 4; ++p) {
    if (*p == ',') {
      *p = '\0';
      field[n++] = p + 1;
    }
  }
  for (char *p = field[0]; *p; ++p) *p = toupper(*p);
  const char *head = field[0];
  stoppedByWatchdog = false;

  if (!strcmp(head, "M")) {  // the normal tracking command
    panAxis.set(field[1] ? atoi(field[1]) : 0);
    tiltAxis.set(field[2] ? atoi(field[2]) : 0);
    return;
  }
  if (!strcmp(head, "S") || !strcmp(head, "STOP") || !strcmp(head, "CENTERED")) {
    panAxis.set(0);
    tiltAxis.set(0);
    Serial.println("STOPPED");
    return;
  }
  if (!strcmp(head, "F")) {
    bool on = field[1] && (field[1][0] == '1' || !strcmp(field[1], "ON"));
    trigger.set(on ? MAX_DUTY : 0);
    Serial.println(on ? "FIRE 1" : "FIRE 0");
    return;
  }
  if (!strcmp(head, "P")) { Serial.println("OK"); return; }
  if (!strcmp(head, "V")) { Serial.println(VERSION); return; }

  // ---- bring-up diagnostics ----
  if (!strcmp(head, "FREQ")) {
    uint32_t hz = field[1] ? (uint32_t)atol(field[1]) : 5000;
    if (hz < 30) hz = 30;
    if (hz > 40000) hz = 40000;
    PWM_FREQ = hz;
    panAxis.reattach(PWM_FREQ);
    tiltAxis.reattach(PWM_FREQ);
    Serial.print("FREQ ");
    Serial.println(PWM_FREQ);
    return;
  }
  if (!strcmp(head, "ENALL")) {
    bool on = field[1] && field[1][0] == '1';
    enableAll(on);
    if (!on) { panAxis.enable(); tiltAxis.enable(); trigger.enable(); }
    Serial.println(on ? "ENALL 1" : "ENALL 0");
    return;
  }
  if (!strcmp(head, "T")) {
    uint8_t pin = field[1] ? (uint8_t)atoi(field[1]) : 255;
    int duty = field[2] ? atoi(field[2]) : 700;
    uint32_t ms = field[3] ? (uint32_t)atol(field[3]) : 800;
    if (ms > 5000) ms = 5000;
    bool safe = false;
    for (uint8_t i = 0; i < SAFE_PIN_COUNT; ++i)
      if (SAFE_PINS[i] == pin) safe = true;
    if (!safe) { Serial.println("ERR unsafe pin"); return; }
    Serial.print("PULSE ");
    Serial.println(pin);
    pulsePin(pin, duty, ms);
    Serial.println("PULSE done");
    return;
  }
  if (!strcmp(head, "SCAN")) {
    scanPins(field[1] ? atoi(field[1]) : 700);
    return;
  }
  if (!strcmp(head, "START_FIRE")) {
    trigger.set(MAX_DUTY);
    Serial.println("FIRE 1");
    return;
  }
  if (!strcmp(head, "STOP_FIRE")) {
    trigger.set(0);
    Serial.println("FIRE 0");
    return;
  }

  // Legacy direction words. NOTE the old GUI appends ",<speed>" - v1 of the
  // MicroPython firmware compared the whole line against "UP", so "UP,600"
  // matched nothing and fell through to the branch that stops everything.
  int px = 0, ty = 0;
  bool known = true;
  if (!strcmp(head, "UP")) ty = 1;
  else if (!strcmp(head, "DOWN")) ty = -1;
  else if (!strcmp(head, "LEFT")) px = -1;
  else if (!strcmp(head, "RIGHT")) px = 1;
  else if (!strcmp(head, "UP LEFT")) { px = -1; ty = 1; }
  else if (!strcmp(head, "UP RIGHT")) { px = 1; ty = 1; }
  else if (!strcmp(head, "DOWN LEFT")) { px = -1; ty = -1; }
  else if (!strcmp(head, "DOWN RIGHT")) { px = 1; ty = -1; }
  else known = false;

  if (known) {
    int speed = MAX_DUTY;
    if (field[1]) {
      speed = atoi(field[1]);
      if (speed < 0) speed = 0;
      if (speed > MAX_DUTY) speed = MAX_DUTY;
    }
    panAxis.set(px * speed);
    tiltAxis.set(ty * speed);
    Serial.print("CMD ");
    Serial.println(head);
    return;
  }

  Serial.print("Unknown: ");
  Serial.println(head);
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);
  panAxis.begin(PWM_FREQ);
  tiltAxis.begin(PWM_FREQ);
  trigger.begin(TRIG_FREQ);
  lastCmdMs = millis();
  Serial.print(VERSION);
  Serial.println(" ready");
}

void loop() {
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (lineLen) {
        lineBuf[lineLen] = '\0';
        lineLen = 0;
        lastCmdMs = millis();
        handle(lineBuf);
      }
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = ch;
    } else {
      lineLen = 0;  // junk / no terminator - drop it
    }
  }

  uint32_t now = millis();
  if (now - lastCmdMs > CMD_TIMEOUT_MS && !stoppedByWatchdog) {
    panAxis.set(0);
    tiltAxis.set(0);
    trigger.set(0);
    stoppedByWatchdog = true;
  }

  if (now - lastTickMs >= LOOP_MS) {
    lastTickMs = now;
    panAxis.step();
    tiltAxis.step();
    trigger.step();
    digitalWrite(LED_PIN, (panAxis.moving() || tiltAxis.moving()) ? LOW : HIGH);
  }
}
