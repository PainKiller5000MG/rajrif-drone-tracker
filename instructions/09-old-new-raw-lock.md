# 9. Old vs New vs Raw vs Lock

Four distinct acquisition/tracking mechanisms exist across this codebase.
This is what each one actually does, frame by frame, and why.

## ① Old system (retired 2026-08-07)

`Old/main_pre_lock_backup.py` — continuous re-detection, no memory, no
identity. Biggest box wins every frame.

**How it worked:**
1. A YOLO model (`Old/best.pt`, 6.25MB — an older/smaller model, not
   YOLOv11x) ran on a dedicated background thread.
2. Every loop iteration, it ran inference filtered at confidence `0.7`.
3. Of everything above that floor, it kept only the **largest box area** —
   no other criterion.
4. That box directly drove the mount, every frame, completely overwriting
   whatever it believed a moment ago.
5. Stale-out after `0.5`s of no detection.

```python
# Old/main_pre_lock_backup.py · detection_loop()
results = self.model(frame, conf=0.7, verbose=False)
best_box, max_area = None, 0
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best_box = (x1, y1, x2, y2)
```

**Why it was removed:** a sky patch scored 0.65–0.73 confidence and passed
as the biggest box — a hexacopter's own bounding box is ~80% sky, so
nothing about "biggest box above a confidence floor" can tell a drone from
a cloud that resembles one. Also lost drones mid-bank, since a fresh
re-detection every frame has no memory to fall back on when the
current-frame detection is momentarily weak.

## ② New system — `tracker_core.py`

Operator-selected identity lock. Memorizes appearance, refuses mismatches.
No detection model involved at all.

**How it works:**
1. Operator drags/clicks a box → `LockTracker.select(frame, box)`.
2. `select()` builds an `Identity` — a color-histogram-and-structure
   snapshot of that exact patch, stored as an immutable `anchor` view. A
   correlation tracker (CSRT or KCF) is initialized on the same box.
3. Every frame, `update()` runs the correlation tracker forward and scores
   the result against the identity's `anchor`/`recent`/`gallery` views.
4. Score ≥ `track_thresh` (0.42) → stays `lock`, identity slowly adapts —
   gated against the untouched `anchor`, so gradual drift can't walk the
   identity onto a different object.
5. Score drops → `coast` (predicts from velocity) → `search` (needs
   `relock_thresh` 0.60, sustained over `relock_frames` 3, to re-acquire).
6. A **contrast-mass veto** rejects anything too visually flat to be real
   (this is what stops sky from ever re-acquiring). A **distractor guard**
   remembers look-alikes and refuses to re-lock onto them.

```python
# tracker_core.py · LockTracker.select() — L661
def select(self, frame, box):
    self.identity = Identity(frame, box)
    self.cv_tracker = create_cv_tracker(self.cfg["tracker"])
    self.cv_tracker.init(frame, tuple(int(round(v)) for v in box))
    self.state = "lock"
```

**What it doesn't know:** no concept of "drone" — locks onto a mug, a
person, a car exactly as readily. Nothing here can auto-start.

## ③ Raw YOLO mode — `detect_and_track.py --mode raw`

The trained model, none of the persistence. Every frame independently
re-decided.

**How it works:**
1. `model.predict(frame, conf=args.conf_threshold, ...)` — `conf=0.60`
   default, so YOLO itself discards weak detections before returning
   results.
2. Boxes covering more than `--max-box-area-frac` (10% of frame) are
   additionally rejected.
3. Largest remaining box wins — same rule as Old, stronger model
   (YOLOv11x, mAP50 0.905), plus the two filters Old never had.
4. Aim point = box center, fed straight into the PD controller. Stale-out
   coasts at 0.4× command scale for 0.5s if nothing currently qualifies.
5. **Nothing remembered between frames** beyond that stale window.

```python
# detect_and_track.py — RAW branch
results = model.predict(frame, conf=args.conf_threshold, ...)
best_box, best_area = None, 0
for result in results:
    for box in result.boxes:
        ...
        if (area / (frame_w * frame_h)) > args.max_box_area_frac:
            continue
        if area > best_area:
            best_area = area
            best_box = (x1, y1, w, h)
```

**Explicit, known risk** (from the module's own docstring): "deliberately
close to the pre-2026-08-07 `Old/main_pre_lock_backup.py` design... offered
here because it was explicitly asked for, not because the failure mode has
been fixed."

## ④ Lock mode — `detect_and_track.py --mode lock` (default)

YOLO originates the lock; the New system's identity tracker owns
everything after.

**How it works:**
1. A dedicated detector thread (`AutoAcquireDetector` in
   `yolo_autoacquire.py`) polls at `poll_interval_s=0.12`s (~8Hz),
   decoupled from the camera/control loop so slow inference never blocks
   mount commands.
2. Each cycle: same two filters as Raw (confidence ≥0.60, box ≤10%),
   largest surviving box is the candidate.
3. Compared to the **previous** cycle's candidate by IoU. Overlap ≥0.35 →
   streak increments. Below that → streak **decays by one**, not resets to
   zero — a single rough cycle (real drone motion measured IoU as low as
   0.0–0.31) doesn't erase prior progress, but a genuinely different
   object still can't accumulate.
4. A wall-clock staleness guard discards the reference if ≥8× the poll
   interval has passed since the last successful cycle.
5. `streak_count` reaches `confirm_frames` (4, ≈1s) → box published. Main
   loop, only while `tracker.state == "idle"`, calls
   `tracker.select(frame, box)` — **the exact same call a manual drag
   makes.**
6. From that instant, this *is* the New system — same `LockTracker`, same
   everything. YOLO's job ends at handoff; it never overrides an existing
   lock.

```python
# yolo_autoacquire.py · _process_frame()
if too_stale:
    self._streak_count = 1
elif _boxes_consistent(self._streak_box, best_box, self.iou_min):
    self._streak_count += 1
else:
    self._streak_count = max(1, self._streak_count - 1)
...
if self._streak_count >= self.confirm_frames:
    self._confirmed_box = best_box   # published for the main loop to pick up
```

```python
# detect_and_track.py — LOCK branch
if mode == "lock":
    detector.submit_frame(frame)
    if tracker.state == "idle":
        auto_box = detector.get_confirmed_box()
        if auto_box is not None:
            tracker.select(frame, auto_box)   # same call a manual drag makes
    st = tracker.update(frame, (), now=now)
```

**Residual gap:** the model is single-class (`drone` only) — no explicit
"not a drone" signal exists, so a bird/kite/plane that's small, confident,
and consistent for a full second could still pass every filter. The
confidence/size/persistence gates substantially reduce this (validated
against a real webcam-face false positive during testing, which this
pipeline reliably rejects), but don't eliminate it. Manual clear /
panic-stop is the real backstop.

## Comparison table

| | Old | Raw | New | Lock |
|---|---|---|---|---|
| Detection model | Old/best.pt (weak) | YOLOv11x, 0.905 mAP50 | none — manual select | YOLOv11x, 0.905 mAP50 |
| Confidence floor | 0.70 | 0.60 | — | 0.60 |
| Box-size filter | none | ≤10% of frame | — | ≤10% of frame |
| Acquisition | auto, every frame | auto, every frame | manual click/drag | auto, ~1s confirm |
| Persistence/memory | none | none | full identity lock | full identity lock |
| Survives occlusion | no | no | yes (coast) | yes (coast) |
| Refuses wrong object | no | no | yes | yes |
| Auto-fire eligible | predates fire control | never, by design | manual-lock only | yes, gated |
| Known failure mode | cloud-locking, lost banking drones | same risk, filters reduce it | needs a human to start | same as New, once locked |

## Accuracy verdict

**Per-frame detection accuracy** — Raw and Lock are tied: identical
YOLOv11x model, identical 0.905 mAP50. Old is weaker. New doesn't detect
at all.

**Tracking accuracy over time** — Lock wins clearly. It's the only system
pairing the strongest detector with the identity-lock design purpose-built
to fix Old's exact documented failures.

**Overall: Lock mode is the most accurate**, and is the recommended
default for any real deployment. Raw mode is a deliberate,
explicitly-requested trade of that stability for lower acquisition
latency — not because the underlying risk was ever resolved.
