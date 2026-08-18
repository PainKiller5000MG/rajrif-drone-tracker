"""Optional YOLO auto-acquire feed for tracker_gui.py.

This exists only to let the operator skip the manual drag/click and have the
tracker take the first *plausible, sustained* YOLO detection as its lock
target. It never bypasses tracker_core's identity/state machine: it only
ever proposes a box for tracker_gui.py to hand to LockTracker.select(), the
exact same call a manual drag makes. See 20RajRif/CLAUDE.md's "Do not
reintroduce it without asking" note - this was asked for and approved for a
specific demo, and is off by default.

Deliberately NOT here: any persistence beyond a single acquisition, any use
of the tracker_core.update() proposals slot, anything hardware-facing. Once
a box is handed off, tracker_core owns the lock exactly like it does after a
manual selection.

A single low-confidence, oversized frame is not enough to trust: the model is
single-class ("drone"), so a false positive is still labelled "drone" with no
second class to rule it out. Two independent filters plus a short run of
consistent detections are what stand in for that missing class check:

  * confidence threshold
  * max box-area-fraction (a real drone detection measured <=2.6% of frame
    in testing; a false positive on a face/background measured 35-60%)
  * N consecutive detector cycles with a similar box before it is trusted

This mirrors, and deliberately does not repeat, the old pre-2026-08-07
Old/main_pre_lock_backup.py detection_loop: same idea of a decoupled
detector thread handing off through a lock-guarded shared frame, but that
thread re-centred on every single frame's biggest box with no consistency
check at all - documented in CLAUDE.md as exactly the design that locked
clouds and lost banking drones.
"""

import threading
import time


def _iou(a, b):
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _boxes_consistent(a, b, iou_min=0.35):
    """Are two (x, y, w, h) boxes close enough to be 'the same detection'?

    A drone a few frames apart at ~5Hz polling moves modestly relative to
    its own size, so a moderate IoU bar (not near-1.0) is enough to reject
    "different object entirely" while tolerating real motion.
    """
    return _iou(a, b) >= iou_min


class AutoAcquireDetector:
    """Runs YOLO on its own thread, off the camera/control loop's cadence.

    submit_frame() is called every video frame (cheap: just a reference
    swap under a lock). The detector thread itself only wakes up every
    poll_interval_s, so a slow forward pass can never block the caller or
    eat into the mount's 5Hz keepalive cadence.
    """

    def __init__(self, weights_path, conf_thresh=0.01, max_box_area_frac=1.0,
                 confirm_frames=4, log=None, poll_interval_s=0.12, iou_min=0.35):
        self.weights_path = weights_path
        self.conf_thresh = conf_thresh
        self.max_box_area_frac = max_box_area_frac
        self.confirm_frames = max(1, confirm_frames)
        self.poll_interval_s = poll_interval_s
        self.iou_min = iou_min
        self.log = log or (lambda _m: None)

        self._frame_lock = threading.Lock()
        self._latest_frame = None

        self._result_lock = threading.Lock()
        self._confirmed_box = None

        self._streak_box = None
        self._streak_count = 0
        self._streak_last_t = None
        # If real wall-clock time between successful detections exceeds
        # this, the previous streak_box is too stale to compare against
        # (the gap could mean the object moved far, or - just as likely on
        # a loaded machine - inference itself briefly fell behind, so the
        # "next cycle" is not actually ~poll_interval_s after the last one).
        # A generous multiple of the nominal cadence, not a tight bound.
        self._max_streak_gap_s = max(1.0, self.poll_interval_s * 8)

        self._stop_flag = threading.Event()
        self._thread = None

        self.ready = False
        self._model = None
        self._load_error = None
        self._try_load_model()

    def _try_load_model(self):
        import os

        abs_path = os.path.abspath(self.weights_path) if self.weights_path else "(empty path)"
        self.log("Auto-acquire: weights path = %s" % abs_path)

        if not self.weights_path or not os.path.exists(self.weights_path):
            self._load_error = "file not found"
            self.ready = False
            self.log("Auto-acquire: weights file does not exist at %s - "
                      "check the path (folder layout must match the README), "
                      "auto-acquire is disabled, manual lock still works." % abs_path)
            return

        self.log("Auto-acquire: weights file found (%d bytes), loading YOLO..." %
                  os.path.getsize(self.weights_path))
        try:
            import time as _time
            t0 = _time.time()
            from ultralytics import YOLO
            self._model = YOLO(self.weights_path)
            self.ready = True
            self.log("Auto-acquire: model loaded OK in %.1fs, ready to run." %
                      (_time.time() - t0))
        except Exception as exc:
            self._load_error = str(exc)
            self.ready = False
            self.log("Auto-acquire: could not load YOLO weights (%s) - "
                      "auto-acquire is disabled, manual lock still works." %
                      self._load_error)

    def submit_frame(self, frame):
        if not self.ready:
            return
        with self._frame_lock:
            self._latest_frame = frame

    def get_confirmed_box(self):
        with self._result_lock:
            box, self._confirmed_box = self._confirmed_box, None
            return box

    def start(self):
        if not self.ready:
            self.log("Auto-acquire: start() called but detector is not ready "
                      "(model never loaded) - nothing will run.")
            return
        if self._thread is not None:
            self.log("Auto-acquire: start() called but a thread is already running.")
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.log("Auto-acquire: detector thread started (polling every %.2fs)." %
                  self.poll_interval_s)

    def stop(self):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
            self.log("Auto-acquire: detector thread stopped.")

    def _process_frame(self, frame, cycle, heartbeat_every):
        """One detection cycle's worth of work: inference, filtering, and
        streak bookkeeping. Pulled out of _run() so it can be driven
        directly (no thread/sleep timing) in tests."""
        try:
            results = self._model.predict(
                frame, conf=self.conf_thresh, verbose=False)
        except Exception as exc:
            self.log("Auto-acquire: inference error (%s), skipping frame" % exc)
            return

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_w * frame_h
        best_box = None
        best_area = 0
        raw_count = 0
        rejected_oversize = 0

        for result in results:
            for box in result.boxes:
                raw_count += 1
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                w, h = x2 - x1, y2 - y1
                area = w * h
                area_frac = (area / frame_area) if frame_area else 0.0
                if area_frac > self.max_box_area_frac:
                    rejected_oversize += 1
                    continue
                if area > best_area:
                    best_area = area
                    best_box = (x1, y1, w, h)

        if best_box is None:
            if raw_count:
                self.log("Auto-acquire: cycle %d - %d raw detection(s) above "
                          "conf %.2f, all rejected as oversized (max %.0f%% of "
                          "frame, %d over)" %
                          (cycle, raw_count, self.conf_thresh,
                           self.max_box_area_frac * 100, rejected_oversize))
            elif cycle % heartbeat_every == 0:
                self.log("Auto-acquire: watching, no detection above "
                          "conf %.2f yet (cycle %d)" % (self.conf_thresh, cycle))
            # A single missed cycle (drone briefly not detected) shouldn't
            # throw away several cycles of progress - decay by one instead
            # of hard-resetting. Measured on real footage: a fast-moving
            # drone at this polling rate occasionally drops a cycle even
            # when it's genuinely still there. Enough consecutive misses
            # (>= confirm_frames worth) still fully clears it, so a stale
            # box can't confirm arbitrarily long after the object is gone.
            self._streak_count = max(0, self._streak_count - 1)
            if self._streak_count == 0:
                self._streak_box = None
                self._streak_last_t = None
            return

        best_area_frac = best_area / frame_area if frame_area else 0.0
        now = time.time()
        too_stale = (self._streak_last_t is not None and
                     (now - self._streak_last_t) > self._max_streak_gap_s)
        if too_stale:
            # Too much real time passed since the last successful cycle to
            # meaningfully compare boxes at all - could be the object moved
            # far, or (measured in practice on a loaded machine) inference
            # itself briefly fell behind, so "consecutive" cycles were not
            # actually close in wall-clock time. Start clean rather than
            # comparing against a comparison that's no longer meaningful.
            self._streak_count = 1
        elif self._streak_box is not None and _boxes_consistent(self._streak_box, best_box, self.iou_min):
            self._streak_count += 1
        else:
            # Fast real motion between polls can legitimately drop IoU
            # below the bar even for the same object (measured 0.0-0.94
            # across real flight footage depending on system load) - decay
            # by one rather than snapping to 1, so a single rough cycle
            # doesn't erase an otherwise-consistent run. A genuinely
            # different object still can't accumulate, since consecutive
            # cycles keep failing the check.
            self._streak_count = max(1, self._streak_count - 1)
        self._streak_box = best_box
        self._streak_last_t = now

        self.log("Auto-acquire: candidate box %dx%d (%.1f%% of frame) - "
                  "streak %d/%d" %
                  (best_box[2], best_box[3], best_area_frac * 100,
                   self._streak_count, self.confirm_frames))

        if self._streak_count >= self.confirm_frames:
            with self._result_lock:
                self._confirmed_box = best_box
            self._streak_box = None
            self._streak_count = 0
            self.log("Auto-acquire: CONFIRMED - handing box to tracker.select()")

    def _run(self):
        cycle = 0
        warned_no_frame = False
        heartbeat_every = max(1, int(5.0 / self.poll_interval_s))  # ~every 5s

        while not self._stop_flag.is_set():
            time.sleep(self.poll_interval_s)
            cycle += 1

            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                if not warned_no_frame:
                    self.log("Auto-acquire: no frame received yet from the "
                              "camera loop - waiting...")
                    warned_no_frame = True
                continue
            warned_no_frame = False

            self._process_frame(frame, cycle, heartbeat_every)
