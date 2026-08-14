"""Headless checks for the lock-on tracker.

Builds synthetic footage with the failure modes the mount actually sees -
camera shake, a sudden jump, a full disappearance while a look-alike walks in,
then the real target coming back - and asserts the lock ends up on the *same*
object.

    py -3.11 test_tracking.py            run the checks
    py -3.11 test_tracking.py --save     also write the clips for eyeballing
"""

import math
import os
import sys

import cv2
import numpy as np

from tracker_core import LockTracker

W, H = 640, 360
SIZE = 64
RNG = np.random.RandomState(7)
BG = None


def _background():
    global BG
    if BG is None:
        bg = RNG.randint(40, 120, (H, W, 3)).astype(np.uint8)
        bg = cv2.GaussianBlur(bg, (0, 0), 6)
        for i in range(14):
            x, y = RNG.randint(0, W - 60), RNG.randint(0, H - 60)
            cv2.rectangle(bg, (x, y), (x + RNG.randint(20, 60), y + RNG.randint(20, 60)),
                          tuple(int(v) for v in RNG.randint(30, 150, 3)), -1)
        BG = cv2.GaussianBlur(bg, (0, 0), 1.5)
    return BG.copy()


def make_object(base_bgr, seed):
    """A textured patch: same silhouette, distinct colour + speckle."""
    rng = np.random.RandomState(seed)
    patch = np.zeros((SIZE, SIZE, 3), np.uint8)
    patch[:, :] = base_bgr
    for _ in range(18):
        x, y = rng.randint(4, SIZE - 12, 2)
        cv2.rectangle(patch, (x, y), (x + rng.randint(4, 10), y + rng.randint(4, 10)),
                      tuple(int(v) for v in rng.randint(0, 255, 3)), -1)
    cv2.rectangle(patch, (6, 6), (SIZE - 6, SIZE - 6), (255, 255, 255), 2)
    return patch


def paste(img, patch, cx, cy):
    h, w = patch.shape[:2]
    x, y = int(cx - w / 2), int(cy - h / 2)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    img[y0:y1, x0:x1] = patch[y0 - y:y1 - y, x0 - x:x1 - x]


def render(objects, shake=(0, 0), noise=4.0):
    """objects: list of (patch, cx, cy). shake moves the whole scene."""
    img = _background()
    for patch, cx, cy in objects:
        paste(img, patch, cx, cy)
    dx, dy = shake
    if dx or dy:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    if noise:
        img = np.clip(img.astype(np.float32) + RNG.normal(0, noise, img.shape),
                      0, 255).astype(np.uint8)
    return img


class Recorder:
    def __init__(self, name, enabled):
        self.frames = [] if enabled else None
        self.name = name

    def add(self, frame, st):
        if self.frames is None:
            return
        f = frame.copy()
        col = {"lock": (0, 255, 0), "coast": (0, 210, 255),
               "search": (0, 120, 255)}.get(st.state, (160, 160, 160))
        if st.box is not None and st.state != "idle":
            x, y, w, h = [int(v) for v in st.box]
            cv2.rectangle(f, (x, y), (x + w, y + h), col, 2)
        cv2.putText(f, "%s %.2f" % (st.state, st.score), (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        self.frames.append(f)

    def save(self, outdir):
        if not self.frames:
            return
        path = os.path.join(outdir, self.name + ".mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
        for f in self.frames:
            vw.write(f)
        vw.release()
        print("   wrote %s" % path)


def shake_at(i, amp=3.0):
    return (amp * math.sin(i * 2.3), amp * 0.7 * math.cos(i * 3.1))


# ----------------------------------------------------------------- checks

def scenario_hold(rec):
    """Shake + a sudden jump must not break the lock."""
    target = make_object((40, 40, 220), 11)
    tr = LockTracker({"relock_frames": 3})
    before_jump = []
    after_jump = []
    for i in range(120):
        if i < 45:
            cx, cy = 160 + i * 2.0, 180 + 25 * math.sin(i / 9.0)
        elif i == 45:
            cx, cy = 380, 120          # teleport, ~150 px in one frame
        else:
            cx, cy = 380 + (i - 45) * 1.2, 120 + 20 * math.sin(i / 7.0)
        frame = render([(target, cx, cy)], shake=shake_at(i))
        if i == 0:
            tr.select(frame, (cx - SIZE / 2, cy - SIZE / 2, SIZE, SIZE))
        st = tr.update(frame, now=i / 30.0)
        err = math.hypot(st.center[0] - cx, st.center[1] - cy)
        if 5 < i < 45:
            before_jump.append(err)
        elif i > 45:
            after_jump.append(err)
        rec.add(frame, st)
    shaky = max(before_jump)
    recovery = sum(1 for e in after_jump if e > 60)     # frames off the target
    settled = max(after_jump[-30:])
    print("   hold: state=%s  worst error under shake=%.1f px, "
          "re-acquired %.2f s after the jump, settled error=%.1f px"
          % (st.state, shaky, recovery / 30.0, settled))
    assert st.state == "lock", "lost the target during shake/jump: %s" % st.state
    assert shaky < 30, "shake alone pulled it %.1f px off" % shaky
    assert recovery <= 12, "took %.2f s to re-acquire after the jump" % (recovery / 30.0)
    assert settled < 30, "settled %.1f px off the target" % settled
    return True


def scenario_reid(rec):
    """Target vanishes, a different object shows up, target returns.

    The lock must come back on the original object and never sit on the decoy.
    """
    target = make_object((40, 40, 220), 11)      # red-ish
    decoy = make_object((60, 200, 60), 23)       # green-ish, same silhouette
    tr = LockTracker({"relock_frames": 3})
    locked_on_decoy = 0
    decoy_pos = (170.0, 90.0)
    st = None
    for i in range(200):
        objs = []
        tx = ty = None
        if i < 60:                                # target present
            tx, ty = 200 + i * 1.5, 200
        elif i < 130:                             # gone; decoy walks in
            objs.append((decoy, decoy_pos[0], decoy_pos[1]))
        else:                                     # target back, elsewhere
            tx, ty = 430 - (i - 130) * 1.0, 250
            objs.append((decoy, decoy_pos[0], decoy_pos[1]))
        if tx is not None:
            objs.append((target, tx, ty))
        frame = render(objs, shake=shake_at(i, 2.0))

        if i == 0:
            tr.select(frame, (tx - SIZE / 2, ty - SIZE / 2, SIZE, SIZE))
        st = tr.update(frame, now=i / 30.0)
        rec.add(frame, st)

        if st.state == "lock" and 60 <= i < 130:
            d = math.hypot(st.center[0] - decoy_pos[0], st.center[1] - decoy_pos[1])
            if d < SIZE:
                locked_on_decoy += 1

    d_target = math.hypot(st.center[0] - tx, st.center[1] - ty)
    d_decoy = math.hypot(st.center[0] - decoy_pos[0], st.center[1] - decoy_pos[1])
    print("   re-id: state=%s  dist to target=%.1f px  dist to decoy=%.1f px  "
          "frames locked on decoy=%d" % (st.state, d_target, d_decoy, locked_on_decoy))
    assert locked_on_decoy == 0, "locked onto the decoy while the target was gone"
    assert st.state == "lock", "never re-locked the original target (%s)" % st.state
    assert d_target < 45, "re-locked %.1f px from the target" % d_target
    assert d_decoy > SIZE, "ended up on the decoy"
    return True


def scenario_twins(rec):
    """Two identical objects: after a loss the tracker must refuse to guess."""
    twin = make_object((40, 40, 220), 11)
    tr = LockTracker({"relock_frames": 3})
    a = (180.0, 150.0)
    b = (450.0, 250.0)
    st = None
    wrong = 0
    for i in range(140):
        objs = [(twin, b[0], b[1])]
        if i < 40:
            objs.append((twin, a[0], a[1]))
        frame = render(objs, shake=shake_at(i, 2.0))
        if i == 0:
            tr.select(frame, (a[0] - SIZE / 2, a[1] - SIZE / 2, SIZE, SIZE))
        st = tr.update(frame, now=i / 30.0)
        rec.add(frame, st)
        if i > 60 and st.state == "lock":
            if math.hypot(st.center[0] - b[0], st.center[1] - b[1]) < SIZE:
                wrong += 1
    print("   twins: state=%s  frames locked on the twin=%d" % (st.state, wrong))
    assert wrong == 0, "swapped onto an identical-looking object"
    return True


def scenario_occlusion(rec):
    """A bar sweeps in front of the target, which also changes size."""
    target = make_object((200, 80, 40), 31)
    tr = LockTracker({"relock_frames": 3})
    st = None
    for i in range(150):
        scale = 1.0 + 0.45 * math.sin(i / 30.0)
        patch = cv2.resize(target, (int(SIZE * scale), int(SIZE * scale)))
        cx, cy = 150 + i * 2.0, 180
        frame = render([(patch, cx, cy)], shake=shake_at(i, 2.5))
        if 60 <= i < 80:                       # occluder passes over it
            x = int(cx - 40)
            cv2.rectangle(frame, (x, 0), (x + 90, H), (20, 20, 20), -1)
        if i == 0:
            tr.select(frame, (cx - SIZE / 2, cy - SIZE / 2, SIZE, SIZE))
        st = tr.update(frame, now=i / 30.0)
        rec.add(frame, st)
    err = math.hypot(st.center[0] - cx, st.center[1] - cy)
    print("   occlusion: state=%s  centre error=%.1f px" % (st.state, err))
    assert st.state == "lock", "did not recover after the occluder (%s)" % st.state
    assert err < 45, "recovered %.1f px off the target" % err
    return True


def scenario_speed(_rec):
    """Loop cost at 720p - the control loop has to stay near camera rate."""
    import time as _t
    global W, H, BG
    W, H, BG = 1280, 720, None
    try:
        target = make_object((40, 40, 220), 11)
        tr = LockTracker({})
        frame = render([(target, 400, 300)])
        tr.select(frame, (400 - SIZE / 2, 300 - SIZE / 2, SIZE, SIZE))
        n = 60
        frames = [render([(target, 400 + i * 3, 300)], shake=shake_at(i))
                  for i in range(n)]
        t0 = _t.time()
        for i, f in enumerate(frames):
            tr.update(f, now=i / 30.0)
        per = (_t.time() - t0) / n * 1000.0
        # searching is the expensive path - measure it too
        tr.state = "search"
        tr.lost_since = 2.0
        t0 = _t.time()
        for i, f in enumerate(frames[:20]):
            tr._update_search(f, tr.center, 0.03, 10.0 + i / 30.0, 1280, 720, ())
        per_search = (_t.time() - t0) / 20 * 1000.0
        print("   speed: %.1f ms/frame locked, %.1f ms/frame searching "
              "at 1280x720" % (per, per_search))
        assert per < 25.0, "too slow to track at 30 fps: %.1f ms" % per
        assert per_search < 60.0, "search too slow: %.1f ms" % per_search
    finally:
        W, H, BG = 640, 360, None
    return True


def scenario_says_why(rec):
    """A refused re-lock has to name the gate that refused it.

    The symptom that motivated this: an object is plainly visible, boxed by
    YOLO, and the mount sits in `search` doing nothing. On screen that is
    indistinguishable from "cannot see it", so there is nothing to tune. The
    state now carries the reason - which threshold, and by how much.
    """
    target = make_object((40, 40, 220), 11)
    other = make_object((60, 220, 90), 41)      # a plainly different object

    def objs(i):
        if i < 12:
            return [(target, 320.0, 180.0)]
        return [(other, 320.0, 180.0)]          # swapped: must NOT be taken

    tr = LockTracker({"relock_frames": 3})
    tr.select(render(objs(0)), (320 - 32, 180 - 32, 64, 64))

    notes = []
    for i in range(1, 90):
        frame = render(objs(i))
        st = tr.update(frame, [], now=i / 30.0)
        rec.add(frame, st)
        if st.state in ("search", "coast") and st.note:
            notes.append(st.note)
    assert notes, "refused the re-lock but never said why"
    assert any("too weak" in n or "look-alike" in n or "score alike" in n
               or "nothing in the search area" in n for n in notes), \
        "reason text is not one of the known gates: %s" % notes[:3]
    assert tr.state != "lock", "locked onto a completely different object"
    print("   refusal reason: %r" % notes[-1])
    return True


def scenario_relock_while_moving(rec):
    """The confirm-over-N-frames gate must not be defeated by the target
    still being in motion.

    It required the candidate to stay within 1.5x its own size between
    frames; a target crossing the frame at speed moves further than that, so
    the counter reset every frame and the re-lock never completed. The
    allowance now includes the distance the target could have travelled.
    """
    patch = make_object((40, 40, 220), 11)
    VPX = 45.0                                  # px/frame - a fast crossing
    span = W - 2.0 * SIZE

    def pos(i):
        k = i * VPX
        leg = int(k // span)
        off = k - leg * span
        return SIZE + (off if leg % 2 == 0 else span - off), 180.0

    def objs(i):
        if 14 <= i < 24:                        # brief hide, forces the drop
            return []
        return [(patch, pos(i)[0], 180.0)]

    tr = LockTracker({"relock_frames": 3})
    x0, _ = pos(0)
    tr.select(render(objs(0)), (x0 - 32, 180 - 32, 64, 64))

    dropped = relocked = None
    for i in range(1, 200):
        gt = (pos(i)[0] - 32, 180 - 32, 64.0, 64.0)
        frame = render(objs(i))
        st = tr.update(frame, [gt] if objs(i) else [], now=i / 30.0)
        rec.add(frame, st)
        if st.state != "lock" and dropped is None:
            dropped = i
        if dropped is not None and st.state == "lock":
            relocked = i
            break
    assert dropped is not None, "the hide never dropped the lock - test is void"
    assert relocked is not None, \
        "target moving %.0f px/frame was never re-locked" % VPX
    took = relocked - dropped
    assert took <= 25, "took %d frames to re-lock a moving target" % took
    print("   moving re-lock: dropped at frame %d, re-locked %d frames later "
          "while crossing at %.0f px/frame" % (dropped, took, VPX))
    return True


def scenario_relock_after_size_change(rec):
    """Re-lock an object that has moved closer or further away.

    The search tries a narrow set of scales first, because an object that
    merely jumped is the same size a frame later and extra scales only add
    rival peaks. The wide set used to be gated behind WIDE_SEARCH_AFTER_S, so
    a *brief* loss - which is what the target moving toward or away from the
    camera produces - only ever looked for it between 0.65x and 1.55x.
    Anything past that was left showing an orange search box next to a plainly
    visible object.

    It now widens the moment nothing clears all three floors. Testing the
    combined score alone was not enough: a box at the wrong scale sweeps in
    background, keeps a high colour score, lands exactly on the combined
    threshold and so looked like a reason not to widen, while its shape score
    failed the actual gate.
    """
    patch = make_object((40, 40, 220), 11)

    def relock_frames(ratio):
        target = cv2.resize(patch, (max(10, int(SIZE * ratio)),) * 2)

        def objs(i):
            if i < 40:                       # drifting toward the new size
                r = 1.0 + (ratio - 1.0) * (i / 40.0)
                s = max(10, int(SIZE * r))
                return [(cv2.resize(patch, (s, s)), 320.0, 180.0)]
            if i < 52:                       # brief loss - under a second
                return []
            return [(target, 320.0, 180.0)]

        tr = LockTracker({"relock_frames": 3})
        tr.select(render(objs(0)), (320 - 32, 180 - 32, 64, 64))
        dropped = None
        for i in range(1, 160):
            frame = render(objs(i))
            st = tr.update(frame, [], now=i / 30.0)
            rec.add(frame, st)
            if st.state != "lock" and dropped is None:
                dropped = i
            if dropped is not None and st.state == "lock":
                return i - dropped
        return None

    got = {r: relock_frames(r) for r in (0.45, 2.2)}
    for ratio, frames in got.items():
        assert frames is not None, \
            "object came back at %.0f%% of its locked size after a brief " \
            "loss and was never re-acquired" % (ratio * 100)
        assert frames <= 45, \
            "took %d frames to re-lock a %.0f%% size change" \
            % (frames, ratio * 100)
    print("   size change: re-locked at %s"
          % ", ".join("%.0f%% in %d frames" % (r * 100, n)
                      for r, n in sorted(got.items())))
    return True


def check_click_snaps_to_object_edges():
    """A click must lock the object, not a square of its middle.

    Clicking used to drop a fixed `click_box` square wherever the cursor was
    unless a detection happened to contain it. On anything bigger than that
    square - a drink can at arm's length - the lock got a patch of the
    object's middle, so the identity snapshot was a crop and every later
    re-lock was scored against it.
    """
    from tracker_core import estimate_object_box, iou

    # A plain object on a plain background: the drink-can case.
    bg = np.zeros((H, W, 3), np.uint8)
    bg[:, :] = (200, 170, 120)
    bg = np.clip(bg.astype(np.float32) + RNG.normal(0, 3.0, bg.shape),
                 0, 255).astype(np.uint8)

    def can(size):
        """A green can with a pale label band across its middle.

        The band matters: a click in the centre of the can lands *on* it, and
        a reading that follows colour returns the band - full width, a third
        of the height. That was 1 click in 3 coming back smaller than the
        object on the real rig.
        """
        p = np.zeros((size, size, 3), np.uint8)
        p[:, :] = (60, 120, 60)
        p[int(size * 0.34):int(size * 0.66), :] = (230, 230, 225)
        return p

    fixed_scores, edge_scores, found, undersize = [], [], 0, 0
    for size in (30, 64, 120, 170):
        img = bg.copy()
        paste(img, can(size), 320, 180)
        truth = (320 - size / 2.0, 180 - size / 2.0, float(size), float(size))
        fixed = (320 - 35.0, 180 - 35.0, 70.0, 70.0)
        fixed_scores.append(iou(fixed, truth))
        got = estimate_object_box(img, 320, 180, hint=70.0)
        if got is None:
            edge_scores.append(iou(fixed, truth))    # falls back, no worse
            continue
        found += 1
        edge_scores.append(iou(got, truth))
        assert got[2] >= 12 and got[3] >= 12, "returned a speck: %s" % (got,)
        if got[2] * got[3] < 0.75 * size * size:
            undersize += 1

    mean_edge = sum(edge_scores) / len(edge_scores)
    mean_fixed = sum(fixed_scores) / len(fixed_scores)
    assert found >= 3, "found edges on only %d of 4 objects" % found
    assert mean_edge > mean_fixed + 0.2, \
        "edge snap (%.2f) is no better than the fixed square (%.2f)" \
        % (mean_edge, mean_fixed)
    # The regression that matters: a box inside the object, not around it.
    assert undersize == 0, \
        "%d of 4 clicks returned under 3/4 of the can - the label band is " \
        "being locked instead of the can" % undersize

    # It must refuse rather than invent: a click on flat background has no
    # object under it, and a confidently wrong box is worse than none.
    blank = estimate_object_box(bg, 60, 60, hint=70.0)
    assert blank is None or (blank[2] < W * 0.55 and blank[3] < H * 0.55), \
        "flooded the whole frame from a click on empty background: %s" % (blank,)

    print("   click snap: mean IoU %.2f vs %.2f for the fixed square "
          "(edges found on %d/4)" % (mean_edge, mean_fixed, found))
    return True


def check_gui_cfg_complete():
    """Every knob the tracker reads must survive the trip through the GUI.

    The click-to-edges switch is read out of `cfg` in the tracker loop, so a
    key that never gets built into the cfg silently disables the feature
    rather than failing.
    """
    import ast

    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "tracker_gui.py"), encoding="utf-8").read()
    for key in ("click_snap_edges", "click_box"):
        assert '"%s": self.' % key in src, \
            "%s is never built into the cfg the tracker thread receives" % key
    ast.parse(src)                       # and the panel still imports cleanly
    return True


def check_blob_mode():
    """The totally-blob lock: follow a dark blob, catch it again after a loss.

    No appearance profile by design - the operator's chosen trade is instant
    re-catching over never-wrong. This only asserts the mechanics: it follows
    a moving dark blob against a bright sky, coasts through a disappearance,
    and re-catches the strongest blob within a few frames of it returning.
    """
    from blob_lock import BlobTracker, find_blobs

    W2, H2 = 640, 360
    sky = np.full((H2, W2, 3), 190, np.uint8)

    def scene(cx, present=True):
        img = sky.copy()
        img[:] = np.clip(img.astype(np.int16) +
                         RNG.randint(-4, 5, img.shape), 0, 255).astype(np.uint8)
        if present:
            cv2.circle(img, (int(cx), 170), 11, (25, 25, 25), -1)
        return img

    # the detector must see the blob at all
    assert any(b[0] <= 320 <= b[0] + b[2] for b in find_blobs(scene(320))), \
        "detector cannot see a plain dark blob on sky"

    tr = BlobTracker({"coast_s": 0.3, "relock_frames": 3})
    tr.select(scene(100), (100 - 20.0, 170 - 20.0, 40.0, 40.0))
    relocked_at = None
    for i in range(1, 120):
        cx = 100 + i * 4                       # steady crossing
        present = not (40 <= i < 60)           # hidden for 20 frames
        st = tr.update(scene(cx, present), (), now=i / 30.0)
        if i < 40:
            assert st.state == "lock", "lost a plainly visible blob at f%d" % i
            assert abs(st.center[0] - cx) < 30, "blob lock drifted off"
        if i >= 60 and st.state == "lock" and relocked_at is None:
            relocked_at = i
    assert relocked_at is not None, "never re-caught the returned blob"
    assert relocked_at - 60 <= 8, \
        "took %d frames to re-catch (should be ~relock_frames)" % (relocked_at - 60)
    print("   blob mode: followed at 4 px/frame, re-caught %d frames after return"
          % (relocked_at - 60))
    return True


def check_mount():
    """DC duty mapping: stiction floor, pulsing, watchdog re-sends."""
    from mount import EspDcMount

    class FakeLink:
        def __init__(self):
            self.lines = []

        def write_line(self, t):
            self.lines.append(t)

        def drain(self, log_lines=True):
            return []

    link = FakeLink()
    m = EspDcMount(link, min_duty=300, max_duty=1000, pulse_below=0.3)
    assert m._duty(0.0, 300, 1000) == 0, "zero demand must not drive"
    full = m._duty(1.0, 300, 1000)
    assert full == 1000, "full demand should be max duty, got %s" % full
    half = m._duty(0.65, 300, 1000)
    assert 300 < half < 1000, "mid demand out of range: %s" % half
    assert m._duty(-1.0, 300, 1000) == -1000, "sign must be preserved"
    # a demand under pulse_below alternates between the floor and nothing
    seen = {m._duty(0.05, 300, 1000) for _ in range(1)}
    for _ in range(40):
        m._t0 -= 0.01
        seen.add(m._duty(0.05, 300, 1000))
    assert seen == {0, 300}, "small demand should pulse the floor, saw %s" % seen

    # per-direction floors: one direction may need more to break loose
    asym = EspDcMount(link, min_duty=300, max_duty=1000, min_duty_neg=520,
                      min_duty_tilt=280, min_duty_tilt_neg=610, pulse_below=0.3)
    pos = asym._duty(0.31, 300, 1000, 520)
    neg = asym._duty(-0.31, 300, 1000, 520)
    assert 300 <= pos <= 320, "positive floor ignored: %d" % pos
    assert -540 <= neg <= -520, "negative floor ignored: %d" % neg
    asym.drive(0.31, -0.31, 0.03)          # just above the pulse threshold
    pan_d, tilt_d = [int(v) for v in link.lines[-1].split(",")[1:]]
    assert 300 <= pan_d <= 330, "pan should sit just above its floor: %d" % pan_d
    assert -640 <= tilt_d <= -610, \
        "tilt down should sit just above its own floor: %d" % tilt_d

    m.drive(0.0, 0.0, 0.03)
    n = len(link.lines)
    m.drive(0.0, 0.0, 0.03)
    assert len(link.lines) == n, "identical command re-sent too eagerly"
    m.last_send_time -= 1.0
    m.drive(0.0, 0.0, 0.03)
    assert len(link.lines) > n, "watchdog keep-alive not sent"
    print("   mount: duty floor/pulse/keep-alive ok (%s)" % link.lines[-1])
    return True


def check_axis_cmd():
    """Deadzone hysteresis: no hunting once the target is centred."""
    from tracker_gui import _axis_cmd
    prev, in_dz = [0.0, 0.0], [False, False]
    assert _axis_cmd(0.30, prev, 0, in_dz, 0.03, 1.6, 0.1, 0.05, 1.0) > 0
    assert _axis_cmd(-0.30, prev, 0, in_dz, 0.03, 1.6, 0.1, 0.05, 1.0) < 0
    assert _axis_cmd(0.02, prev, 0, in_dz, 0.03, 1.6, 0.1, 0.05, 1.0) == 0.0
    # inside the deadzone: stays quiet until well outside it again
    assert _axis_cmd(0.07, prev, 0, in_dz, 0.03, 1.6, 0.1, 0.05, 1.0) == 0.0
    assert _axis_cmd(0.12, prev, 0, in_dz, 0.03, 1.6, 0.1, 0.05, 1.0) != 0.0
    assert abs(_axis_cmd(9.0, prev, 0, in_dz, 0.03, 1.6, 0.1, 0.05, 1.0)) <= 1.0
    assert _axis_cmd(0.30, prev, 0, in_dz, 0.03, 1.6, 0.1, 0.05, -1.0) < 0, \
        "direction flip ignored"
    print("   aiming: PD limits, deadzone hysteresis and direction flip ok")
    return True


def check_firmware():
    """Both firmware builds must parse and keep the protocol they claim.

    The two have to stay interchangeable: whichever is on the board, the GUI
    talks to it the same way.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    py_path = os.path.join(here, "Old", "For_ESP32", "ESP32_MAIN.py")
    src = open(py_path, encoding="utf-8").read()
    compile(src, py_path, "exec")
    for token in ('"M"', '"S"', '"F"', "LEGACY_DIRS", "CMD_TIMEOUT_MS",
                  "START_FIRE", "line.split"):
        assert token in src, "MicroPython firmware lost %s" % token

    ino_path = os.path.join(here, "Old", "For_ESP32", "ESP32_MAIN",
                            "ESP32_MAIN.ino")
    ino = open(ino_path, encoding="utf-8").read()
    for token in ('"M"', '"S"', '"F"', '"P"', '"V"', "START_FIRE",
                  "CMD_TIMEOUT_MS", "DOWN RIGHT"):
        assert token in ino, "Arduino firmware lost %s" % token
    assert ino.count("{") == ino.count("}"), "unbalanced braces in the sketch"
    print("   firmware: MicroPython + Arduino builds both keep M/S/F/P/V, "
          "the legacy words and the watchdog")
    return True


def main():
    save = "--save" in sys.argv
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_out")
    if save:
        os.makedirs(outdir, exist_ok=True)

    checks = [("hold-through-shake-and-jump", scenario_hold),
              ("re-identify-after-disappearing", scenario_reid),
              ("refuse-identical-twin", scenario_twins),
              ("survive-occlusion-and-scale", scenario_occlusion),
              ("say-why-it-will-not-relock", scenario_says_why),
              ("relock-while-still-moving", scenario_relock_while_moving),
              ("relock-after-size-change", scenario_relock_after_size_change),
              ("click-snaps-to-object-edges",
               lambda _r: check_click_snaps_to_object_edges()),
              ("gui-passes-every-knob", lambda _r: check_gui_cfg_complete()),
              ("black-blob-mode", lambda _r: check_blob_mode()),
              ("loop-speed-at-720p", scenario_speed),
              ("dc-mount-drive", lambda _r: check_mount()),
              ("aiming-controller", lambda _r: check_axis_cmd()),
              ("esp32-firmware", lambda _r: check_firmware())]
    failed = 0
    for name, fn in checks:
        rec = Recorder(name, save)
        print("-> %s" % name)
        try:
            fn(rec)
            print("   PASS")
        except AssertionError as exc:
            failed += 1
            print("   FAIL: %s" % exc)
        if save:
            rec.save(outdir)
    print("\n%d/%d checks passed" % (len(checks) - failed, len(checks)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
