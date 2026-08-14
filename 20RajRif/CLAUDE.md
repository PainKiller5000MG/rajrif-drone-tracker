# 20RajRif — pan/tilt tracker on a DC mount

Lock onto **one object the operator picks with the mouse** — any object, no
class list — and hold it through vibration, a jump across the frame and a full
disappearance, re-locking *that* object rather than the one standing next to
it. `README.md` is the operator's document; this file is for whoever works on
the code next.

```
Scan360GUI.bat                 or: py -3.11 tracker_gui.py
py -3.11 test_tracking.py      vision + mount + firmware checks (--save writes clips)
py -3.11 test_integration.py   whole loop against a fake camera and fake serial
py -3.11 find_min_duty.py      walks each axis down until it stops turning
```

Both test scripts run headless on this PC with **no hardware attached**. Run
them before and after any change; there is no other safety net.

## This is not Scan360

The two rigs share a filename, a look and a code lineage, and that has already
caused mistakes. What is actually shared is **one file**:

| | 20RajRif | Scan360 |
| --- | --- | --- |
| motors | BTS7960 DC gearmotors | TB6600 steppers |
| control | signed PWM duty, open loop | absolute angle (`x=`/`y=`) |
| feedback | **none** — no position at all | commanded angle is known |
| board | ESP32 | ESP32 |
| vision | `tracker_core.py` | the same `tracker_core.py` |

`tracker_core.py` is **byte-for-byte identical** to
`Python/Scan360/Python/tracker_core.py` and contains no hardware whatsoever —
no serial, no angles, no duty, no FOV. Keep it that way. Any vision or lock
change belongs in that file and gets copied across **whole**, in either
direction, not merged by hand.

```bash
md5sum tracker_core.py ../Scan360/Python/tracker_core.py
```

Use `md5sum`, not `diff`. A `diff` run from the wrong working directory prints
nothing and reads exactly like "identical" — that mistake was made on
2026-08-07 and cost a wrong answer to the user.

Everything hardware lives on this side and must **not** drift toward Scan360:
`mount.py`, the duty-fraction control law and `_axis_cmd` in `tracker_gui.py`,
`find_min_duty.py`, and the firmware in `Old/For_ESP32/`.

## Layout

```
tracker_gui.py            control panel + the tracker thread + overlay
tracker_core.py           the lock: identity, search, re-lock  (SHARED — see above)
blob_lock.py              the OTHER lock: totally-blob, no appearance profile
mount.py                  SerialLink + the two mount backends
find_min_duty.py          measures the stiction floor per axis and direction
check_motion.py           the same measurement with the camera answering, not you
test_tracking.py          vision scenarios, mount duty mapping, control law, firmware
test_integration.py       select -> track -> jog -> clear against fakes
Old/For_ESP32/            the firmware, two interchangeable builds
Old/                      the CUAS operator station — see below
```

`Old/*` is ignored by `.gitignore` with `!Old/For_ESP32/`. That is deliberate:
`Old/mount_config.json` carries a site latitude/longitude and this is a public
repo. Do not un-ignore it.

### Old/main.py — the operator station, re-eyed (2026-08-09)

No longer retired: at the user's request `Old/main.py` was ported to the
shared identity lock. It keeps everything the new tracker GUI does not have —
D-pad, FIRE / AUTO FIRE, IDMS azimuth slew, gyro fallback to a default pose,
the map view, and the legacy direction-word protocol with zone-based speed —
and replaces only its eyes: the YOLO biggest-box thread is gone, the operator
drags a box in the video panel, `tracker_core` holds the lock (KCF, measured
better and 6x cheaper than CSRT on the recorded flights).

Things to know before touching it:

* It **imports `../tracker_core.py` and `../blob_lock.py` via sys.path** —
  never copy either into `Old/`, the copies would drift apart silently.
* A **MODE: PROFILE / BLOB** button toggles between the two locks, same
  semantics as the main GUI's dropdown. Toggling clears any lock and centres
  the mount.
* **AUTO FIRE gained two gates**: it fires only in the `lock` state — never
  on a coasted prediction or a search guess — and only in PROFILE mode,
  because in blob mode "locked" merely means "a dark blob", which can be a
  bird. The button is disabled in BLOB and the condition backs it up. The
  legacy centred/zone<30 condition still applies on top. Do not remove
  either gate.
* `gyro_data.py` used to open its serial port at import with no guard, so a
  missing FC killed the app before the window opened. It now degrades to
  (None, None), which the callers already handled.
* The pre-port app is `Old/main_pre_lock_backup.py`, byte-for-byte, and
  `best.pt` stays on disk unused.
* All of this is **untracked by git** (the `Old/*` ignore above) — the
  backups are local only, so copy before overwriting anything here.

## Mount and firmware

Two builds of the same protocol — an Arduino sketch and a MicroPython file. Send
`V` to see which is loaded; both answer with a version string.

```
Motor 1  tilt   BTS7960 #1   RPWM 25  LPWM 26  R_EN 27  L_EN 14
Motor 2  pan    BTS7960 #2   RPWM 32  LPWM 33  R_EN 12  L_EN 13
Trigger         BTS7960 #3   RPWM 16  LPWM 17  R_EN  5  L_EN  4
Status LED      GPIO 2   (lit = mount at rest)

M,<pan>,<tilt>   signed duty -1023..1023, +pan = right, +tilt = up
S                stop both axes
F,1 / F,0        trigger on / off
P -> OK          V -> version
T,<pin>,<duty>,<ms>   pulse one GPIO, not watchdog limited
SCAN,<duty>      pulse each safe GPIO in turn (skips 16/17)
FREQ,<hz>        change PWM frequency live
```

Things that cost time to find, so do not undo them:

* **PWM is 5 kHz, not 20 kHz.** 20 kHz is inaudible and was the original
  setting, but these BTS7960 modules will not switch that fast — the motors
  barely twitched. Change it live with `FREQ,<hz>` rather than reflashing.
* **The firmware stops motors and trigger after 600 ms of silence.** A crashed
  or unplugged PC therefore cannot leave the mount slewing. The GUI keeps the
  link alive by re-sending the current command five times a second — if you
  restructure the loop, keep that cadence or the mount will stutter.
* **Duty is ramped, not stepped**, so the gearboxes never see an instant
  full-current reversal.
* The legacy direction words (`UP`, `DOWN LEFT`, `START_FIRE`, …) still work,
  with or without a `,<speed>` suffix. v1 of the firmware compared the whole
  line against `"UP"`, so `"UP,600"` matched nothing and fell through to the
  branch that stopped everything.
* The trigger is manual only, gated behind the **ARM** tick box. There is no
  auto-fire, and nothing in the tracker may add one.

## Two lock modes (panel dropdown, 2026-08-09)

**Profile (identity)** — `tracker_core.LockTracker`, the shared lock: memorises
the object's appearance and refuses mismatches. Never takes the wrong object
(0 wrong-object frames across every recorded flight) at the price of slower
re-acquisition of a changed target.

**Black blob** — `blob_lock.BlobTracker`, built at the operator's request
("make it a black blob instead of a profile - easier to catch if lost"). No
appearance at all: follow the dark contrasting blob nearest the prediction,
and when lost take the strongest blob that holds still relative to itself for
`relock_frames` frames. Its detector is the winner of the 2026-08-09 race
(multi-scale DoG + ring-brightness sky test, ~2 ms, stateless — safe on a
panning mount). Measured on the recorded flights: clip 1 **100 % locked /
0 wrong at 2.3 ms/frame**; clip 2 93 % locked with the best on-drone count of
any variant (22) *and* 13 wrong-blob frames — it will follow a bird if the
bird is the best dark blob. That trade is the point; do not bolt identity
checks onto it, that is what Profile mode is. `blob_lock.py` is 20RajRif-side
(not shared); Scan360 may copy it whole.

## Control law

`_axis_cmd` in `tracker_gui.py`: PD on the normalised image error, output a
speed fraction in [-1, 1], which `mount.py` maps onto duty.

* **Gains and deadzones are fractions of the half-frame, not degrees.**
  `0.045` is about 29 px at 1280 wide. This is the single most dangerous
  difference from Scan360, whose same-named keys are degrees.
* **The deadzone has hysteresis** — once inside, the axis stays quiet until the
  error grows well past it. Without that a mount that overshoots by a hair
  oscillates forever.
* **The D term is capped at 0.6× the P term.** A re-lock moves the target
  across the frame in one step and a raw derivative answers that with a full
  speed kick in whatever direction — the mount drove *away* from the target.
  The cap means D can only ever slow the approach.
* **Coasting creeps**: while aiming at a prediction the command is scaled to
  0.4. It is a guess, so do not slew on it.
* **Measure the duty floor before touching any gain.** `find_min_duty.py` asks
  you; `check_motion.py` asks the camera. The camera is bolted to the mount, so
  if the mount turns the whole image shifts, and phase correlation measures that
  to a fraction of a pixel — a number instead of a yes. Use `--sweep`: stopping
  at the first movement gives you a floor and nothing else, and single readings
  a hair over the noise are backlash, not rotation. The floors are per axis
  *and* per direction (`min_duty`, `min_duty_neg`, `min_duty_tilt`,
  `min_duty_tilt_neg`).

### Measured on the rig, 2026-08-09 (camera, 1280x720)

Floors, and speed at full duty. Pan is roughly linear in duty above the floor.

| | floor | set to | top speed |
| --- | --- | --- | --- |
| pan right | 240 | 276 | 155 px/s |
| pan left | 240 | 276 | 189 px/s |
| tilt up | 200 | 229 | 116 px/s |
| tilt down | 330 | 379 | 99 px/s |

Tilt *down* needing more than up is real and reproducible, not a swapped pin —
the signed shift confirms GPIO 32/33/26/25 are pan right / pan left / tilt up /
tilt down as labelled, and that `pan_dir` and `tilt_dir` of +1 are both correct.
Cross-axis coupling is under 2 px.

`kp_pan` 3.0 / `kp_tilt` 2.5 / `pulse_below` 0.20 follow from those speeds: the
mount tops out around 155 px/s, so it has to reach high duty *early* or a
walking person simply outruns it. At 1280 wide that gives quiet under 29 px,
pulsed nudging to ~45 px, 78 px/s at 100 px off, and full duty from 220 px out.
Raising `kp` further is safe on this rig — at full speed and ~20 Hz the mount
covers ~8 px per frame against a 29 px deadzone, so it cannot overshoot into
oscillation.

**The top speed is the real limit and no gain fixes it.** A person walking a few
metres away crosses the frame faster than 155 px/s. That is gearing, not tuning.

## Settings version trap

`SETTINGS_VERSION = 2`. In v1 `kp_pan`/`kp_tilt` were degrees-per-frame and
`deadzone_x`/`deadzone_y` were degrees. The same names now hold a speed
fraction and a fraction of the half-frame. A v1 file loaded blind gives a
deadzone of 1.0 — the mount would never move — so `RESCALED_IN_V2` drops those
keys back to defaults on load. Bump the version and add to that set whenever a
saved key changes meaning.

## Synced from Scan360, 2026-08-07

`tracker_core.py` replaced wholesale, bringing:

* **Click-to-edges** (`estimate_object_box`). A click is one point but the lock
  needs a box, and a fixed square is the right size for exactly one object at
  one range — on anything bigger it locks a patch of the middle and the
  identity snapshot is a crop. Three independent readings (what the outline
  encloses, a colour flood, the smallest closed contour); takes the **largest**
  plausible one, because everything left after the plausibility test fails
  small, not large. Refuses rather than inventing. Panel switch: *Snap a click
  to the object's edges*.
* **Refusal reasons** (`st.note`). A search that will not take a plainly
  visible object now names the gate — which floor, by how much, or that two
  things score too alike. On screen it was otherwise indistinguishable from
  "cannot see it", so there was nothing to tune.
* **Immediate wide-scale escalation.** Widens the search scales the moment
  nothing clears all *three* floors, instead of waiting out
  `WIDE_SEARCH_AFTER_S`. Testing the combined score alone was not enough: a box
  at the wrong scale sweeps in background, keeps a high colour score, lands on
  the combined threshold and looks like a reason not to widen, while its shape
  score fails the real gate. The two scale sets are **merged**, not chosen
  between.
* **Velocity-aware re-lock drift.** The confirm-over-N-frames gate demanded the
  candidate stay within 1.5× its own size between frames; a fast crossing beats
  that every frame, so the counter reset forever and the re-lock never
  completed. It now also allows the distance the target could have travelled.

Ported into `tracker_gui.py` by hand, since the two panels have diverged past a
line diff: the click-to-edges path and its tick box, the `Click-lock: …` log
line, the note on the overlay, and the once-a-second `Not re-locking: …` log.

**Deliberately not ported:**

* `calibration.py` — measures degrees-per-pixel by dithering a turret that
  takes absolute angle commands. This mount has no angles to dither. The
  equivalent here is px/s per unit duty, which is a different measurement and a
  different module: `check_motion.py`, written 2026-08-09.
* `lock_lag_frames` / `pose_log` — Scan360 compensates capture latency by
  rewinding to the pose in force when the frame was exposed. There is no pose
  here to rewind to.
* `search_mode` / `scan_speed` — Scan360's autonomous sweep for a class. This
  rig is operator-picked by design.

## Known gaps and failures

* **The mount is too slow to follow a person at close range.** 155-189 px/s pan
  at full duty; someone walking a few metres away moves several hundred px/s
  across the frame. Gearing, not gains.
Closed, but worth knowing they were ever open:

* **Locking a cloud at full score** (the worst failure this design exists to
  prevent; watched live 2026-08-09 evening, then caught in an instrumented
  replay of clip 4: a sky patch 300 px from the drone scored 0.65-0.73,
  colour 0.73, passed every floor and the 3-frame confirmation). Root cause:
  a hexacopter's box is ~80 % sky, so the histogram and template honestly
  report that sky matches sky - no score floor can ever catch it. Fix:
  `contrast_frac` on every Appearance plus the flat `CONTRAST_VETO_MIN`
  veto on both re-acquisition paths. Clouds measure 0.000 median / 0.024
  p99; the true drone never fell below 0.075 even in a 1.6x-loose box. The
  veto disarms itself for featureless identities (pale mug on pale desk).
  It also killed the clip-2 edge-speck false lock (off 9 -> 0) and dropped
  clip 4's "locked" from a fake 44.5 % (mostly clouds) to an honest 11 % -
  the drone was out of frame for most of that clip.
* **A frozen re-lock bar** refused a plainly visible drone at `match
  0.53<0.60` for 12 s straight. The floors now decay with search time
  (RELAX_*): full bar for 2 s (jump recovery stays strict), sliding to
  0.50/0.28 by 12 s. The decay floor is guarded by the contrast veto above,
  which is what makes it safe - 0.48 used to lock cloud banks.

* **Losing a banking drone** ("same issue" report, 2026-08-09, twice). The
  score decays 0.51 -> 0.42 over ~8 frames as the drone banks, the lock drops,
  and search then demands re-lock-grade scores (0.60/0.35) of a pose the
  identity never memorised - an oracle scoring the *ground-truth box itself*
  cleared all three floors in only 14 of 168 lost frames, so no candidate
  source could fix it and lowering the floors measurably locks clouds
  (394 wrong-object frames at 0.48). The actual defect: `recent` stopped
  adapting below 0.5 similarity, freezing the pre-bank pose exactly when it
  was needed. `RECENT_SIM_MIN = 0.42` holds through the bank - clip 1 went
  73% -> 97.9% locked, wrong-object frames stayed 0, both rigs' suites green.
  The origin gates (the walk defence) are untouched.
* **A dark-blob-in-sky proposer is NOT the fix, measured.** Three detector
  families (black-hat, darkness-vs-ring, DoG) all hit 100% per-frame on the
  recorded flight at 2-4 ms, fed through the YOLO-era proposals slot - and
  changed nothing: locked% byte-identical to baseline, because the tracker's
  own search already had the drone as best candidate and the score floor was
  the binding constraint. Do not rebuild one before re-measuring; the modules
  and harness are described in the 2026-08-09 session, replayable against
  `test_out/drone_flight*.mp4` (gitignored, keep those clips).

* **A dead pan-left half-bridge**, found 2026-08-09 and the reason for the
  original "it will not track a person left and right" report. Pan right was
  healthy to 1023 while pan left worked once to 620, then stopped at every duty
  and did not recover after a 90 s cool-down — same session, same supply, same
  code. That asymmetry is what identifies it as hardware rather than tuning,
  and it is worth re-running `check_motion.py --sweep` before believing any
  tracking complaint on this rig.

* `refuse-identical-twin` used to fail. A look-alike blacklist entry
  re-centred onto any candidate that loosely matched it, so weak background
  matches walked the entry off the object it was guarding — traced moving from
  (420,217) to (356,99) over ~40 frames, after which the twin was uncovered and
  taken. Fixed in the shared core, so the fix is in Scan360 too. If the twin
  check ever regresses, look at `_is_distractor` first.
* The two re-lock floors and the look-alike switch are now on the panel
  (`relock_color_min`, `relock_struct_min`, `distractor_guard`). They ran at
  core defaults before, which meant the refusal message could say
  `colour 0.00<0.35` with no box to change the 0.35.
* There is no detection-age label because there are no detections: YOLO was
  removed from both rigs on 2026-08-07. Do not reintroduce it without asking —
  the operator's drag or click is the only way a target is acquired, and the
  grey proposal boxes it used to draw were ~200 ms stale and easy to misread
  as live.

## Gotchas

* **`OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` must be set before cv2 builds
  a VideoCapture** or opening the camera takes ~40 s. It is set at the top of
  `tracker_gui.py`; do not move the import above it.
* **MSMF, not DSHOW, for 1280x720@30.** DSHOW only offers YUY2 at that size
  (~5 fps over USB) and ignores MJPG requests.
* **CSRT costs ~25 ms a frame** and runs inside the control loop, dropping 30 Hz
  to ~20. The fps readout in the video window exists for this. Suspect it
  before touching gains when motion looks steppy. KCF is ~7-9 ms, MOSSE ~1 ms.
* **Two Claude consoles are often open on this repo.** On 2026-08-07 another
  session's commit swept up this directory's staged index mid-operation. Stage
  and commit in a single command, and never `git checkout` to undo your own
  edits — you will destroy the other console's work.

## State

First committed 2026-08-07 in `98ce37c0` on branch `simulator-gcs-gimbal-arty`
(14 files; it also carried Scan360's changes, because of the console collision
above — and that branch's history has since been rewritten, so do not expect
that hash to be an ancestor of HEAD).

`test_tracking.py` **13/13** and `test_integration.py` **3/3**, re-run after the
YOLO removal and the twin fix landed.

2026-08-09, first day on the real rig: duty floors measured by camera
(`check_motion.py`), a dead pan-left half-bridge found and repaired, and two
flights recorded to `test_out/drone_flight*.mp4`. Those clips are the tuning
ground truth for `RECENT_SIM_MIN` and the reason the re-lock floors were left
alone - re-derive from them, not from taste.
