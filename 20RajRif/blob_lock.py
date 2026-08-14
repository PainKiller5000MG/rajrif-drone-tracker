"""Totally-blob lock: follow the dark contrasting blob, no appearance profile.

The operator's other lock (`tracker_core.LockTracker`) memorises what the
object LOOKS like and refuses anything that does not match - which is what
makes it safe, and also what makes it slow to re-take a target whose look has
changed. This one is the mode the operator asked for by name: "make it a
black blob instead of a profile - easier to catch if lost". It knows nothing
about the target's appearance. Its whole identity is *the dark blob against
the sky nearest to where the target should be*, and when lost it simply takes
the strongest such blob that holds still relative to itself for a couple of
frames.

That trade is deliberate and the operator owns it: blob mode WILL follow a
bird if the bird is the best dark blob where it is looking. In exchange a
re-catch needs no score, no colour, no pose - if the eye can see a black dot,
so can this. The trigger is unaffected either way: manual, behind ARM.

The detector is the measured winner of the 2026-08-09 detector race
(multi-scale DoG, drone found in 100% of frames of a real flight at ~2 ms;
clouds cannot clear the response threshold, and dark-on-dark texture such as
foliage is rejected by a ring-brightness test - a real object *on sky* has
bright surroundings). Stateless per frame, so a panning mount cannot smear a
background model - there is none.

Interface-compatible with LockTracker as far as tracker_gui.py consumes it:
select / update / clear / locked / state, and a TrackState with the fields
the overlay and the control law read.
"""

import math
import time

import cv2
import numpy as np

from tracker_core import TrackState, clip_box

# ---- detector ----------------------------------------------------------

_DS_W, _DS_H = 320, 180  # 4x downscale from 1280x720
_THR = 9.0               # min DoG response (gray levels of local contrast)
_MIN_AREA = 4            # min component area at downscale (px)
_MAX_COMPS = 40          # cap on components examined per frame
_RING_CONTRAST = 20.0    # surroundings must be this much brighter than blob
_RING_MIN = 90.0         # surroundings must be at least this bright (sky test)
_INFLATE = 1.35          # grow box: DoG core = dark body, target = whole airframe


def _boxes_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw < bx or bx + bw < ax or ay + ah < by or by + bh < ay)


def find_blobs(frame_bgr, max_props=8):
    """Dark contrasting blobs on a bright background. Best first.

    -> list of (x, y, w, h, strength) floats, len <= max_props.
    Stateless: nothing carried between calls.
    """
    H, W = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (_DS_W, _DS_H), interpolation=cv2.INTER_AREA)
    sx = W / float(_DS_W)
    sy = H / float(_DS_H)

    f = small.astype(np.float32)
    b1 = cv2.GaussianBlur(f, (0, 0), 1.5)
    b2 = cv2.GaussianBlur(f, (0, 0), 4.5)
    b3 = cv2.GaussianBlur(f, (0, 0), 10.0)
    dog = np.maximum(b2 - b1, b3 - b2)   # merged scales; positive on dark blobs

    mask = (dog > _THR).astype(np.uint8)
    n, labels, stats, _cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return []

    ii = cv2.integral(small)             # O(1) ring means

    def rect_mean(x0, y0, x1, y1):
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(_DS_W, x1); y1 = min(_DS_H, y1)
        a = (x1 - x0) * (y1 - y0)
        if a <= 0:
            return 0.0, 0
        s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
        return s / float(a), a

    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    order = order[:_MAX_COMPS]

    cands = []
    for lb in order:
        x, y, w, h, area = stats[lb]
        if area < _MIN_AREA:
            continue
        roi = dog[y:y + h, x:x + w]
        lroi = labels[y:y + h, x:x + w]
        resp = float(roi[lroi == lb].max())

        # ring test: an object ON SKY has bright surroundings; texture whose
        # surroundings are dark too (foliage) is not an object on sky
        inner_mean, _ = rect_mean(x, y, x + w, y + h)
        pad = max(3, w // 2, h // 2)
        out_mean, out_a = rect_mean(x - pad, y - pad, x + w + pad, y + h + pad)
        in_a = (min(_DS_W, x + w) - max(0, x)) * (min(_DS_H, y + h) - max(0, y))
        ring_a = out_a - in_a
        if ring_a <= 0:
            continue
        ring_mean = (out_mean * out_a - inner_mean * (w * h)) / ring_a
        if ring_mean < _RING_MIN:
            continue
        if ring_mean - inner_mean < _RING_CONTRAST:
            continue

        cands.append((resp * (ring_mean - inner_mean), x, y, w, h))

    if not cands:
        return []
    cands.sort(reverse=True)

    # One object must come back as ONE box. A drone's thin arms respond
    # weaker than its body, so they surface as separate small components and
    # a raw component box lands on a part of the object. Cluster the
    # SKY-VALIDATED components whose inflated boxes touch - arms sit within
    # half a body of their body. Clustering after the ring test matters: a
    # morphological close on the raw mask was tried first and merged a
    # low-flying drone with the treeline through any thin dark bridge
    # (597 wrong-object frames on the recorded flight).
    merged = []
    for score, x, y, w, h in cands:
        gx, gy = 0.4 * w + 2, 0.4 * h + 2
        box = [x - gx, y - gy, x + w + gx, y + h + gy, x, y, x + w, y + h, score]
        absorbed = False
        for m in merged:
            if not (box[2] < m[0] or m[2] < box[0]
                    or box[3] < m[1] or m[3] < box[1]):
                m[0] = min(m[0], box[0]); m[1] = min(m[1], box[1])
                m[2] = max(m[2], box[2]); m[3] = max(m[3], box[3])
                m[4] = min(m[4], x); m[5] = min(m[5], y)
                m[6] = max(m[6], x + w); m[7] = max(m[7], y + h)
                m[8] = max(m[8], score)
                absorbed = True
                break
        if not absorbed:
            merged.append(box)
    merged.sort(key=lambda m: -m[8])

    out = []
    for m in merged:
        x, y, x2, y2, score = m[4], m[5], m[6], m[7], m[8]
        w, h = x2 - x, y2 - y
        fw = w * sx * _INFLATE
        fh = h * sy * _INFLATE
        fx = (x + w / 2.0) * sx - fw / 2.0
        fy = (y + h / 2.0) * sy - fh / 2.0
        box = (fx, fy, fw, fh)
        if any(_boxes_overlap(box, b[:4]) for b in out):
            continue
        out.append((fx, fy, fw, fh, float(score)))
        if len(out) >= max_props:
            break
    return out


# One import-time warmup call: OpenCV's lazy first-call allocations otherwise
# land on the first real frame (~11-18 ms observed).
find_blobs(np.zeros((720, 1280, 3), np.uint8))


# ---- the lock ----------------------------------------------------------

def _center(b):
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)


class BlobTracker:
    """Follow the dark blob nearest the prediction; catch the best when lost.

    States mirror LockTracker's so the overlay and control law read the same
    things: lock (blob in the gate), coast (briefly no blob - aim at the
    prediction, creep), search (take the strongest blob that stays put,
    relative to itself, for `relock_frames` consecutive frames).
    """

    def __init__(self, cfg=None, log=None):
        cfg = cfg or {}
        self.log = log or (lambda _m: None)
        self.coast_s = float(cfg.get("coast_s", 0.5))
        self.confirm_frames = max(1, int(cfg.get("relock_frames", 3) or 3))
        self.aim_alpha = float(cfg.get("aim_smooth", 0.45))
        self.reset()

    def reset(self):
        self.state = "idle"
        self.locked = False
        self.box = None
        self.center = None
        self.aim = None
        self.vel = (0.0, 0.0)
        self.size0 = None
        self.lost_since = None
        self._t_last = None
        self._catch_box = None
        self._catch_hits = 0
        self.note = ""

    def clear(self):
        self.reset()

    def select(self, frame, box):
        """Lock the blob under the operator's box, else the box itself."""
        box = tuple(float(v) for v in box)
        bx, by = _center(box)
        best = None
        for b in find_blobs(frame):
            cx, cy = _center(b)
            if box[0] <= cx <= box[0] + box[2] and box[1] <= cy <= box[1] + box[3]:
                if best is None or b[4] > best[4]:
                    best = b
        take = best[:4] if best is not None else box
        fh, fw = frame.shape[:2]
        take = clip_box(take, fw, fh)
        self.reset()
        self.box = take
        self.center = _center(take)
        self.aim = self.center
        self.size0 = (take[2], take[3])
        self.state = "lock"
        self.locked = True
        self._t_last = time.time()
        self.log("BLOB LOCKED %dx%d at (%d, %d)%s"
                 % (take[2], take[3], self.center[0], self.center[1],
                    "" if best is not None else " (no blob under the box - using the box)"))
        return True

    # gate: how far from the prediction the blob may be and still be ours
    def _gate(self, dt):
        base = 2.5 * max(self.size0) if self.size0 else 150.0
        return max(120.0, base) + math.hypot(*self.vel) * dt

    def update(self, frame, proposals=(), now=None):
        now = time.time() if now is None else now
        dt = 1 / 30.0 if self._t_last is None else max(1e-3, min(now - self._t_last, 0.25))
        self._t_last = now
        fh, fw = frame.shape[:2]

        st = TrackState()
        if self.state == "idle":
            return st

        blobs = find_blobs(frame)
        pred = (self.center[0] + self.vel[0] * dt,
                self.center[1] + self.vel[1] * dt)

        if self.state in ("lock", "coast"):
            best, best_d = None, None
            gate = self._gate(dt)
            for b in blobs:
                cx, cy = _center(b)
                d = math.hypot(cx - pred[0], cy - pred[1])
                if d <= gate and (best is None or d < best_d):
                    best, best_d = b, d
            if best is not None:
                nb = clip_box(best[:4], fw, fh)
                nc = _center(nb)
                self.vel = (0.7 * self.vel[0] + 0.3 * (nc[0] - self.center[0]) / dt,
                            0.7 * self.vel[1] + 0.3 * (nc[1] - self.center[1]) / dt)
                self.center = nc
                a = self.aim_alpha
                self.aim = (a * nc[0] + (1 - a) * self.aim[0],
                            a * nc[1] + (1 - a) * self.aim[1]) if self.aim else nc
                # The drawn box must sit stable on the FULL object. The
                # detector re-measures from scratch every frame, so its raw
                # width breathes as arms cross the response threshold - drawn
                # directly, that reads as flicker. Grow fast (an arm appearing
                # is real, cover it now), shrink slowly (an arm dropping out
                # for a frame is threshold noise, keep the boundary).
                gw = 0.5 if nb[2] > self.size0[0] else 0.06
                gh = 0.5 if nb[3] > self.size0[1] else 0.06
                self.size0 = (self.size0[0] + gw * (nb[2] - self.size0[0]),
                              self.size0[1] + gh * (nb[3] - self.size0[1]))
                self.box = clip_box((nc[0] - self.size0[0] / 2.0,
                                     nc[1] - self.size0[1] / 2.0,
                                     self.size0[0], self.size0[1]), fw, fh)
                if self.state != "lock":
                    self.log("Blob re-caught after %.1f s."
                             % (now - self.lost_since if self.lost_since else 0.0))
                self.state = "lock"
                self.lost_since = None
                self.note = ""
            else:
                if self.lost_since is None:
                    self.lost_since = now
                self.center = (min(max(pred[0], 0.0), fw - 1.0),
                               min(max(pred[1], 0.0), fh - 1.0))
                self.vel = (self.vel[0] * 0.9, self.vel[1] * 0.9)
                self.aim = self.center
                if now - self.lost_since <= self.coast_s:
                    self.state = "coast"
                    self.note = "no blob in the gate - aiming on prediction"
                else:
                    self.state = "search"
                    self._catch_box = None
                    self._catch_hits = 0

        elif self.state == "search":
            # take the STRONGEST blob anywhere, once it stays put relative to
            # itself (plus its own motion) for confirm_frames frames - the
            # only guard blob mode keeps, so noise cannot yank the mount
            if blobs:
                b = blobs[0]
                bc = _center(b)
                if self._catch_box is not None:
                    pc = _center(self._catch_box)
                    allow = 1.5 * max(b[2], b[3]) + math.hypot(*self.vel) * dt + 40.0
                    if math.hypot(bc[0] - pc[0], bc[1] - pc[1]) <= allow:
                        self._catch_hits += 1
                    else:
                        self._catch_hits = 1
                else:
                    self._catch_hits = 1
                self._catch_box = b[:4]
                if self._catch_hits >= self.confirm_frames:
                    nb = clip_box(b[:4], fw, fh)
                    self.box = nb
                    self.center = _center(nb)
                    self.aim = self.center
                    self.vel = (0.0, 0.0)
                    self.size0 = (nb[2], nb[3])
                    self.state = "lock"
                    self.log("Caught the strongest blob after %.1f s lost."
                             % (now - self.lost_since if self.lost_since else 0.0))
                    self.lost_since = None
                    self.note = ""
                else:
                    self.note = "catching blob (%d of %d frames)" % (
                        self._catch_hits, self.confirm_frames)
            else:
                self._catch_box = None
                self._catch_hits = 0
                self.note = "no dark blob against the sky anywhere"

        st.state = self.state
        st.box = self.box if self.state != "search" else None
        st.center = self.center
        st.aim = self.aim if self.state in ("lock", "coast") else None
        st.visible = self.state == "lock"
        # score readout: blob mode has no identity score; show the detector's
        # view instead so the header stays meaningful
        st.score = 1.0 if self.state == "lock" else 0.0
        st.color = 0.0
        st.struct = 0.0
        st.lost_s = 0.0 if self.lost_since is None else (now - self.lost_since)
        st.search_box = (0.0, 0.0, float(fw), float(fh)) if self.state == "search" else None
        st.note = self.note
        return st
