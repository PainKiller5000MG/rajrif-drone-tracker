# Pan/tilt object tracker

Lock the mount onto **one object the operator picks with the mouse** - any
object, no class list - and keep it there through vibration, a jump across the
frame, and a full disappearance. When the object comes back it re-locks *that*
object, not the one standing next to it. When the object is taken away it
goes orange and says why it is refusing - it does not sit green on whatever
was standing behind it.

There is no YOLO and no class detection anywhere (removed 2026-08-07): the
mouse is the only way a target is acquired, and no `.pt` model file is needed.

```
Scan360GUI.bat            or:  py -3.11 tracker_gui.py
py -3.11 test_tracking.py        vision checks (add --save to write clips)
py -3.11 test_integration.py     full loop against a fake camera + serial
```

## Using it

Pick the mount, COM port and camera in the panel, press **Start**, then in the
video window:

| action | result |
| --- | --- |
| drag the left button | lock onto the object in the box (the box shrinks to the object's edges when they can be found) |
| single left click | lock onto the object under the cursor - its own edges if they can be found, else a fixed-size box |
| right click, or `c` | clear the lock |
| `q` | quit |

A click is one point but the lock needs a box, and a fixed square is the right
size for exactly one object at one range - on anything bigger it grabs a patch
of the middle, and the identity snapshot is then a crop of the object. So a
click reads the object's own edges three ways (what
its outline encloses, a flood of similar colour, the smallest closed contour)
and takes the largest plausible answer. If none of them is plausible it says so
and keeps the fixed square, which is honest rather than confidently wrong.
Untick **Snap a click to the object's edges** to go back to the plain square.

A *drag* is snapped the same way (same tick box), because a generous drag
mostly contains desk, and an identity that is mostly desk stays "locked" on
the desk after the object is carried away - measured at 90 of 90 frames on a
3x drag. On a low-contrast object against clutter the outline honestly cannot
always be found and the drag is kept as drawn (the log says which happened) -
so drag tight when the background is busy.

Box colours: **green** locked and tracking, **amber** target not visible, aiming
on prediction, **orange** lost, searching for it. The orange outline is the
region being searched. The header shows the match score against the object you
originally picked, split into colour and shape.

When it is searching and *not* taking an object you can plainly see, the line
under the header names the gate that is refusing it - which threshold and by
how much, or that two things score too alike to choose between. The same line
goes to the log at most once a second. Without it, a box on the object plus a
tracker that will not take it is indistinguishable from one that cannot see it,
and there is nothing to tune.

## How the lock holds

* **Frame to frame** an OpenCV correlation tracker (CSRT) follows the box, but
  is never trusted blind - every box it returns is scored against the object's
  appearance before it is accepted.
* **Identity** is snapshotted when you select: an HSV colour histogram and a
  normalised grey template. Later high-scoring views join a small gallery so a
  slow turn or a lighting change is absorbed - but a view is only followed,
  and only remembered, while it still resembles the *original* selection.
  Scoring against the gallery alone lets the identity walk: each absorbed
  view lowers the bar for the next, and a chain of small slides carries the
  lock onto a different object without a single gate failing. The first view
  is a gate, not just a member.
* **A washed-out object scores by shape, not colour.** With almost no
  saturated pixels the colour histogram says only "this is pale" - and every
  pale patch in the scene says the same, which used to carry the empty desk
  over the keep-following bar after the object was removed. Colour keeps its
  weight only when the selection actually has colour.
* **What was behind the object is not the object.** The frame is snapshotted
  the moment the target goes missing; any re-lock candidate sitting on
  scenery unchanged since then is refused however well it scores, and while
  coasting the tracker box may only be taken back at full re-lock strength,
  near the prediction, because a coasting tracker is not following anything
  visible.
* **Vibration** is measured by phase correlation on a thumbnail and folded into
  the motion prediction, so a shaking mount does not read as the target moving.
  Shifts too large to be shake are ignored (a big object crossing the frame can
  otherwise fool the estimate).
* **Jumps** land far from the prediction. If the appearance also scores badly
  the box is rejected instead of followed; the search then finds the object
  again, typically inside a third of a second.
* **Size changes** are handled by widening the search scales the moment nothing
  clears all three floors, rather than waiting out a timer. An object that went
  further away comes back smaller *immediately*, and the narrow scale set was
  looking for a size it no longer is. The narrow set still runs first, because
  an object that merely jumped is the same size a frame later and extra scales
  only add rival peaks.
* **A target still moving** when it is re-found is allowed the distance it
  could plausibly have travelled since the last look. The confirm-over-several
  -frames gate used to demand it stay within 1.5x its own size, which a fast
  crossing beats every frame, so the counter reset forever and the re-lock
  never completed.
* **Re-locking** is deliberately harder than staying locked: a candidate must
  beat a higher score, clear the colour *and* shape floors separately, be a
  plausible size, hold up over several frames, and have no rival scoring nearly
  as well. Anything that was standing somewhere else *while* the real target
  was locked is remembered as a look-alike and refused. If it cannot tell two
  objects apart it keeps searching rather than guessing - it will never quietly
  swap targets.

## Tuning

Measure the duty floor first - it matters more than any gain:

```
py -3.11 find_min_duty.py        walks each axis down until it stops turning
```

| symptom | change |
| --- | --- |
| mount buzzes but does not turn | raise **Min duty pan/tilt** until it just moves |
| creeps past the target and comes back | raise **Deadzone**, lower **Kp** |
| slow to centre | raise **Kp** |
| shivers around the target | raise **Deadzone**, or **Pulse below** |
| turns the wrong way | flip **Pan dir** / **Tilt dir** to -1 |
| loses the object too easily | lower **Keep-lock score** (0.42 default) |
| grabs the wrong object after a loss | raise **Re-lock score** and **Rival margin** |

Deadzone and gains are fractions of the half-frame, not degrees - `0.045` is
about 29 px at 1280 wide. Settings persist in `tracker_gui_settings.json`; a
file written by the older stepper-only build is migrated automatically (the
gains meant something different there and would have left the mount immobile).

## Shared with Scan360

`tracker_core.py` is byte-for-byte the same file as
`Python/Scan360/Python/tracker_core.py` and contains no hardware at all - no
serial, no angles, no duty. Vision and lock changes belong there and should be
copied across whole. Everything hardware lives on this side: `mount.py`, the
duty-fraction control law in `tracker_gui.py`, `find_min_duty.py` and the ESP32
firmware. Scan360's `calibration.py` is *not* ported: it measures
degrees-per-pixel by dithering a turret that takes absolute angle commands, and
this mount has no angles to dither.

The identical-twin refusal (`refuse-identical-twin` in `test_tracking.py`)
used to fail here: a blacklist entry re-centred itself onto any weak
candidate that landed near it and random-walked off the twin it was guarding.
Fixed 2026-08-07 in the shared core - entries no longer move - and the check
passes (zero frames locked on the twin).

## Mounts

**ESP32 DC (BTS7960)** - open loop, the default. Two interchangeable builds,
identical protocol; use whichever suits the board:

* `Old/For_ESP32/ESP32_MAIN/ESP32_MAIN.ino` - **Arduino sketch**. Open in the
  Arduino IDE, board "ESP32 Dev Module", upload. Use this one if the board is
  not running MicroPython.
* `Old/For_ESP32/ESP32_MAIN.py` - MicroPython, copy to the board as `main.py`
  (`pip install mpremote`, then `mpremote connect COM11 fs cp ESP32_MAIN.py :main.py`).

Send `V` on the serial port to see which is loaded - both answer with a version
string ending in `(arduino)` or not.

PWM runs at **5 kHz**. 20 kHz is inaudible and was the original setting, but
these BTS7960 modules will not switch that fast - the motors barely twitched
until it came down. Change it live with `FREQ,<hz>` rather than reflashing.

Bring-up commands, for when nothing moves (Serial Monitor at 115200, line
ending "Newline"):

```
T,<pin>,<duty>,<ms>   pulse one GPIO - not watchdog limited, unlike M
SCAN,<duty>           pulse each safe GPIO in turn, printing the number, to
                      find the real wiring (skips 16/17, the trigger)
FREQ,<hz>  ENALL,1/0
```

```
Motor 1  tilt   BTS7960 #1   RPWM 25  LPWM 26  R_EN 27  L_EN 14
Motor 2  pan    BTS7960 #2   RPWM 32  LPWM 33  R_EN 12  L_EN 13
Trigger         BTS7960 #3   RPWM 16  LPWM 17  R_EN  5  L_EN  4
Status LED      GPIO 2   (lit = mount at rest)
```

Protocol, 115200 8N1, newline terminated:

```
M,<pan>,<tilt>   signed duty -1023..1023, +pan = right, +tilt = up
S                stop both axes
F,1  F,0         trigger on / off
P                ping -> OK          V -> version
```

Old direction words (`UP`, `DOWN LEFT`, `CENTERED`, `START_FIRE`, ...) still
work, with or without a `,<speed>` suffix, so `Old/main.py` can still drive the
board. That suffix used to break v1 of the firmware: it compared the whole line
against `"UP"`, so `"UP,600"` matched nothing and fell through to the branch
that stops everything.

Two safety behaviours are new: the firmware **stops the motors and the trigger
if no command arrives for 600 ms**, so a crashed or unplugged PC cannot leave
the mount slewing, and duty is ramped rather than stepped so the gearboxes do
not see instant full-current reversals. The GUI keeps the link alive by
re-sending the current command five times a second.

**Stepper (x=/y=)** - the TB6600 rig that took absolute angle commands. Select
it in the Mount box; the same tracking and the same control law drive it, with
the speed integrated into an angle instead of a PWM duty.

The trigger is manual only and gated behind the **ARM** tick box. There is no
auto-fire.
