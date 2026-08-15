# 3. ESP32 firmware setup

## This must be a genuine ESP32 — not an Arduino Nano

The firmware in this repo (`20RajRif/Old/For_ESP32/`) only runs on real
ESP32 silicon:

- **`ESP32_MAIN.ino`** uses the ESP32's **LEDC peripheral**
  (`ledcAttach`/`ledcSetup`/`ledcAttachPin`/`ledcWrite`) for PWM — this is
  a hardware register block that physically only exists on ESP32 chips. It
  will not compile for AVR/Arduino Nano at all.
- **`ESP32_MAIN.py`** uses MicroPython's `machine.Pin`/`machine.PWM` — this
  runs on ESP32 (and a few other MCUs like RP2040/ESP8266), not on a
  classic Arduino Nano's ATmega328 chip.
- The documented pin numbers (25, 26, 27, 14, 32, 33, 12, 13, 16, 17, 5, 4,
  2) are ESP32 GPIOs — several of them (25, 26, 27, 32, 33) don't exist as
  pin numbers on a Nano at all, which only has D0–D13 and A0–A7.

If you have a board labeled "Nano" in some way, confirm it's specifically
one of: a genuine ESP32 chip on a Nano-shaped breakout, or the official
**Arduino Nano ESP32** (ESP32-S3 under the hood) — not a classic
ATmega328-based Nano. If in doubt, check the exact chip printed on the
board itself.

## Hardware needed

| Item | Notes |
|---|---|
| Genuine ESP32 dev board | Standard 30/38-pin "ESP32 DevKit"-style, or equivalent |
| 3× BTS7960 motor driver modules | Pan, tilt, trigger |
| 2× DC gear motors | Pan + tilt axes |
| **Separate motor power supply** | Matched to your motors' voltage/current. **USB power alone runs the ESP32's logic but will not drive the motors** — this is the single most common "commands are sent and acknowledged but nothing physically moves" cause on this rig. |
| USB cable (data-capable) | For flashing and serial comms |

## Wiring

| Signal | GPIO |
|---|---|
| Motor 1 (tilt) RPWM / LPWM / R_EN / L_EN | 25 / 26 / 27 / 14 |
| Motor 2 (pan) RPWM / LPWM / R_EN / L_EN | 32 / 33 / 12 / 13 |
| Trigger RPWM / LPWM / R_EN / L_EN | 16 / 17 / 5 / 4 |
| Status LED (lit = mount at rest) | 2 |

## Flashing — pick ONE path

### Path A — Arduino IDE + `.ino` (recommended, simpler)

1. Install [Arduino IDE](https://www.arduino.cc/en/software) (2.x).
2. File → Preferences → Additional Board Manager URLs, add:
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
3. Tools → Board → Boards Manager → search "esp32" → install the
   Espressif package.
4. Tools → Board → select your specific board (e.g. "ESP32 Dev Module").
5. Plug in the board, select the correct port under Tools → Port.
6. Open `20RajRif/Old/For_ESP32/ESP32_MAIN/ESP32_MAIN.ino`, click Upload.

No external Arduino libraries are required — only the ESP32 core's
built-in `ledc*` functions.

### Path B — MicroPython + `.py`

1. `pip install esptool`
2. Download the MicroPython ESP32 firmware `.bin` from
   [micropython.org/download/ESP32_GENERIC](https://micropython.org/download/ESP32_GENERIC/)
3. Erase and flash:
   ```
   esptool.py --port COM3 erase_flash
   esptool.py --port COM3 write_flash -z 0x1000 <firmware>.bin
   ```
4. Use Thonny (or `ampy`/`rshell`) to copy
   `20RajRif/Old/For_ESP32/ESP32_MAIN.py` onto the board as `main.py`.

## Verify the firmware, before touching any Python

Open a serial terminal at **115200 baud** on whatever port the board
enumerates as (Arduino IDE's Serial Monitor, PuTTY, `screen /dev/ttyUSB0
115200`, etc.), and send:

| Send | Expect | Confirms |
|---|---|---|
| `V` | a version string | firmware is alive and responding |
| `P` | `OK` | ping command works |
| `S` | (silent, safe) | stop command accepted |

If both `V` and `P` respond correctly, the board is genuinely running one
of the two documented firmwares. If there's silence, garbage, or the port
won't open at all — the firmware isn't actually running, or the
port/baud is wrong. Fix this before moving on to any tracker script.

## Serial protocol reference (115200 8N1, `\n`-terminated ASCII)

```
M,<pan>,<tilt>   proportional drive, each -1023..1023
                 +pan = RIGHT, +tilt = UP, 0 = that axis stops
S                stop both axes (trigger untouched)
F,1 / F,0        trigger on / off
P                ping -> "OK"
V                version string
```
Legacy direction words (`UP`, `DOWN LEFT`, `START_FIRE`, …) also work, with
or without a `,<speed>` suffix, for `Old/main.py`'s protocol.

## Built-in safety behavior (already in the firmware, nothing to configure)

- **600ms comms watchdog** — if no command arrives for `CMD_TIMEOUT_MS`,
  motors and trigger are cut automatically. A crashed or unplugged PC
  cannot leave the mount slewing or firing.
- **Duty is ramped, never stepped** — gearboxes and supply never see an
  instant full-current reversal.
- The two halves of a single H-bridge are never driven together.

Next: [Camera setup](04-camera-setup.md).
