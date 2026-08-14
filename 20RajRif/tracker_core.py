"""Lock-on tracking with identity.

The job: the operator drags a box round *one* object - any object, no class
list - and the mount stays on that object until told otherwise. Vibration,
motion blur, a sudden jump across the frame and a full disappearance must all
end with the *same* object locked, never the one standing next to it.

How it holds:

frame-to-frame    An OpenCV correlation tracker (CSRT by default) follows the
                  box. On its own it is happy to slide onto the background or
                  onto whatever passes in front, so it is never trusted blind.

identity          At lock time we snapshot the object's appearance: an HSV
                  hue/saturation histogram (colour, survives blur and scale)
                  and a normalised grey template (structure, survives colour
                  shift). Every candidate box is scored against that snapshot.
                  A small gallery of later high-scoring views is kept so a
                  slow turn or lighting change is absorbed - but a view only
                  joins it if it still resembles the operator's *original*
                  view, and the lock is only followed at all while it does.
                  Scoring against the gallery alone lets the identity walk
                  (see ORIGIN_* below); the first view has to be a gate, not
                  merely a member.

vibration         Frame-to-frame camera shake is measured by phase
                  correlation on a thumbnail and fed into the motion
                  prediction, so a shaking mount does not read as the target
                  jumping. Output is a smoothed aim point, so the motors are
                  not asked to chase the shake.

jump / occlusion  A tracker box that lands far from the prediction *and*
                  scores badly is rejected rather than followed. Brief losses
                  coast on constant velocity; longer ones drop to a search
                  that grows out from the last known position, multi-scale
                  template matching plus (optionally) class-agnostic YOLO
                  boxes as extra candidates.

re-lock           A search candidate must beat a higher threshold than
                  ordinary tracking, clear both the colour and structure
                  floors on their own, be a plausible size, hold up for
                  several consecutive frames, and - the important one - have
                  no rival elsewhere in the frame scoring nearly as well. If
                  two objects look alike the tracker waits instead of
                  guessing, so it cannot silently swap targets.
"""

import math
import time

import cv2
import numpy as np

TEMPL_PX = 48          # working size for the normalised template
SEARCH_TEMPL_PX = 40   # template size used inside the search (bounds cost)
GALLERY_MAX = 6
GALLERY_MIN_GAP_S = 1.5
# How much a view must still look like the ORIGINAL selection - the operator's
# own drag - to be followed, remembered, or folded into the gallery.
#
# Without these the identity ratchets. `score()` reports the best match over
# the *gallery*, and every view the tracker accepts can join it, so each
# absorbed view lowers the bar for the next: the correlation tracker slides a
# little onto the clutter next door, that view is accepted at track_thresh
# (0.42) and remembered, and now a view slightly further out clears the bar
# too. Nothing ever fails a gate, so none of the re-lock defences - the 0.60
# floor, the rival guard, the look-alike memory - is ever consulted; they
# defend a target the lock has already left. Measured on the real class
# before the gates went in: an object scoring 0.09 against the operator's
# selection scored 0.80 after 24 accepted intermediate views, while still
# scoring 0.09 against the selection itself.
#
# The first view is therefore a gate, not just gallery member 0. The floors
# are deliberately loose - a lighting change or a slow turn must still be
# absorbed - they only have to be tighter than "a different object entirely".
ORIGIN_KEEP_MIN = 0.28      # keep following the correlation tracker at all
ORIGIN_RECENT_MIN = 0.30    # allowed to become the adapting 'recent' view
ORIGIN_GALLERY_MIN = 0.55   # allowed to join the gallery for good
# How similar to the views we already hold a view must be before `recent`
# adapts to it. This is NOT a walk defence - the origin gates above are - it
# only sets how far into an appearance change the adapting view may follow.
# 0.5 was too timid: a drone banking against the sky slides 0.51->0.42 over a
# few frames, `recent` froze at the pre-bank pose the moment it was needed,
# and the lock dropped into a search that then demanded re-lock-grade scores
# of a target it had never actually lost sight of. Measured on two recorded
# flights: 0.42 holds through the bank (73% -> 98% of frames locked, still
# zero wrong-object frames); the identical-twin, backdrop and re-id suites
# all still pass, because the origin gates are what defend those.
RECENT_SIM_MIN = 0.42
MAX_SHIFT_FRAC = 0.08  # the most of the frame camera shake can plausibly move
# Scales the search tries around the size the target was last seen at.
# A target that jumps is the same size a frame later, so the quick search
# stays narrow - extra scales there only add rival peaks and slow the
# recovery. One that has been gone for a second or more has usually gone
# somewhere, and typically further away: at the old 0.65 floor anything
# coming back at half its locked size was never re-acquired at all. So the
# search widens the longer it has been looking.
SEARCH_SCALES = (0.65, 0.8, 1.0, 1.25, 1.55)
SEARCH_SCALES_WIDE = (0.35, 0.45, 0.6, 0.75, 0.9, 1.1, 1.35, 1.7, 2.1)
WIDE_SEARCH_AFTER_S = 1.0
# One re-lock bar cannot be right for every second of a search. At the moment
# of loss the object still looks like itself, so the full bar is correct and
# protects the fast-jump recovery. Ten seconds in, the common case is the
# opposite: the object is plainly visible but CHANGED - banked, re-lit,
# smaller - and a frozen bar refuses it forever (watched live: a drone at
# colour 0.88 held at "match 0.53<0.60" for 12 s while parked in clear view).
# So the combined and colour floors decay with search time, from the
# configured values down to floors that measurably still refuse clouds -
# 0.48 locked cloud banks for 394 frames on the recorded flight, 0.50 never
# did. Nothing else relaxes: the rival margin, the look-alike memory, the
# static-scenery check, the confirm-over-frames gate and the shape floor all
# stay at full strength, and they are what actually prevent a wrong lock.
RELAX_AFTER_S = 2.0     # full bar until here - jump recovery must stay strict
RELAX_FULL_S = 12.0     # fully relaxed from here on
RELAX_SCORE_FLOOR = 0.50
RELAX_COLOR_FLOOR = 0.28
# The shape floor decays too (to a floor verified safe even held there
# permanently: zero wrong-object frames across all recorded flights). The
# gallery only holds poses seen while locked, so an object returning at a
# NEW aspect - a drone nose-on that was selected side-on - fails shape
# against every stored view and a frozen floor refuses it forever: watched
# live at "colour 0.96 / shape 0.27<0.34" with the drone filling the frame.
RELAX_STRUCT_FLOOR = 0.24
# The contrast-mass veto on re-acquisition. A re-lock candidate must contain
# SOMETHING - a measurable fraction of pixels differing from its own border
# (see Appearance.contrast_frac) - or it is surroundings, not object.
# Watched live: a cloud patch cleared score 0.65-0.73, colour 0.73, shape
# 0.59 and the 3-frame confirmation against a drone identity, because that
# identity was 80 % sky and sky honestly matches sky. Measured over 300
# random sky patches against two flights: clouds sit at 0.000 median / 0.024
# p99 while the true drone never fell below 0.075 even in a box 1.6x too
# large, so 0.06 separates them with ~3x margin either way. A flat floor,
# NOT a fraction of the selection's own contrast: a tight box on a close
# drone reads 0.94 and any ratio of that vetoes the same drone seen far
# away. The veto disarms itself when the identity is featureless (a pale
# mug on a pale desk cannot demand its candidates be contrasty).
CONTRAST_PX = 18.0
CONTRAST_VETO_MIN = 0.06
# Look-alike scans run on each of the first N locked frames (then every
# `distractor_scan_every`), so even a seconds-long lock leaves a blacklist
# behind for the search that follows it.
EARLY_DISTRACTOR_FRAMES = 3
# A re-lock candidate whose patch matches the SAME coordinates in the frame
# captured at search entry this closely is a piece of scene that was already
# there after the object left - static background, not the object coming
# back. The turret holds still while searching (by design), so coordinates
# are comparable; the small alignment window below absorbs camera shake.
STATIC_BG_NCC = 0.88
STATIC_BG_PAD = 12


# ---------------------------------------------------------------- helpers

def clip_box(box, frame_w, frame_h, min_size=8):
    x, y, w, h = box
    w = max(min_size, w)
    h = max(min_size, h)
    x = max(0.0, min(float(x), frame_w - min_size))
    y = max(0.0, min(float(y), frame_h - min_size))
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)
    return (x, y, w, h)


def box_center(box):
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)


def patch_of(frame, box):
    x, y, w, h = [int(round(v)) for v in box]
    x = max(0, x)
    y = max(0, y)
    x2 = min(frame.shape[1], x + max(1, w))
    y2 = min(frame.shape[0], y + max(1, h))
    if x2 - x < 4 or y2 - y < 4:
        return None
    return frame[y:y2, x:x2]


def touches_edge(box, frame_w, frame_h, margin=2.0):
    """True if the box is up against a frame border - i.e. probably cropped.

    A target halfway out of shot reads as a small, oddly-shaped object. Both
    its size and its appearance are lies about the real thing.
    """
    return (box[0] <= margin or box[1] <= margin
            or box[0] + box[2] >= frame_w - margin
            or box[1] + box[3] >= frame_h - margin)


CLICK_ROI_SCALE = 4.0    # how far around a click to go looking for the edges


def estimate_object_box(frame, cx, cy, hint=70.0, min_px=12, max_frac=0.55):
    """Find the edges of the object under (cx, cy). -> (x, y, w, h) or None.

    A click is one point, but the lock needs a box, and a fixed-size square
    is the wrong size for everything except one object at one range: too big
    and the identity snapshot is mostly background, too small and it is a
    detail of the object rather than the object.

    Three independent readings of "what is under the cursor", because none of
    them works everywhere:

      enclosed  what the object's OUTLINE contains, found by flooding the
                background inward and keeping what it cannot reach. Ignores
                internal colour, so a can with a label band stays one object.
                Needs the outline to be closed after the morphological close.
      flood     a region of similar colour grown from the click. Good on a
                solid object against plain sky, and the fallback when the
                outline is too soft to close.
      contour   the smallest closed Canny contour enclosing the click.

    If all three are implausible the caller keeps its fixed square, which is
    honest rather than confidently wrong.
    """
    fh, fw = frame.shape[:2]
    cx, cy = int(round(cx)), int(round(cy))
    if not (0 <= cx < fw and 0 <= cy < fh):
        return None

    half = max(min_px * 2.0, hint * CLICK_ROI_SCALE / 2.0)
    rx0 = max(0, int(cx - half))
    ry0 = max(0, int(cy - half))
    rx1 = min(fw, int(cx + half))
    ry1 = min(fh, int(cy + half))
    roi = frame[ry0:ry1, rx0:rx1]
    if roi.shape[0] < min_px or roi.shape[1] < min_px:
        return None
    px, py = cx - rx0, cy - ry0

    max_w, max_h = max_frac * fw, max_frac * fh

    def plausible(box):
        if box is None:
            return False
        x, y, w, h = box
        if w < min_px or h < min_px or w > max_w or h > max_h:
            return False
        # it has to actually be under the cursor, not merely nearby
        if not (x - 1 <= cx <= x + w + 1 and y - 1 <= cy <= y + h + 1):
            return False
        # a box that fills the whole search window is a flood that escaped
        return not (w >= (rx1 - rx0) * 0.95 and h >= (ry1 - ry0) * 0.95)

    cands = [_enclosed_box(roi, px, py, rx0, ry0),
             _flood_box(roi, px, py, rx0, ry0),
             _edge_box(roi, px, py, rx0, ry0)]
    cands = [b for b in cands if plausible(b)]
    if not cands:
        return None
    # The LARGEST plausible reading, not the smallest. Every failure mode
    # left after plausibility is an undersize one - a colour region inside
    # the object rather than the object - because oversize is what
    # plausible() already rejects. Measured over 32 synthetic cans, drones
    # and textured blocks on plain and cluttered backgrounds: smallest-of-two
    # returned under three quarters of the object 24 times and scored mean
    # IoU 0.51; largest-of-three scores 0.82 and undersizes 4 times, all of
    # them a low-contrast outline against clutter.
    cands.sort(key=lambda b: -b[2] * b[3])
    return cands[0]


def _flood_box(roi, px, py, ox, oy):
    """Grow a region of similar colour out from the click."""
    h, w = roi.shape[:2]
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    # Tolerance from the local spread, so a flat object and a noisy one both
    # get a sensible one instead of a hand-picked constant.
    patch = blur[max(0, py - 6):py + 7, max(0, px - 6):px + 7]
    if patch.size == 0:
        return None
    tol = float(np.clip(patch.reshape(-1, 3).std(axis=0).mean() * 2.5, 6.0, 40.0))
    mask = np.zeros((h + 2, w + 2), np.uint8)
    try:
        cv2.floodFill(blur.copy(), mask, (px, py), 255,
                      (tol,) * 3, (tol,) * 3,
                      4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    except cv2.error:
        return None
    inner = mask[1:-1, 1:-1]
    if cv2.countNonZero(inner) < 12:
        return None
    x, y, bw, bh = cv2.boundingRect(inner)
    return (float(x + ox), float(y + oy), float(bw), float(bh))


def _enclosed_box(roi, px, py, ox, oy):
    """The region the object's OUTLINE encloses, whatever colours are inside.

    The other two readings follow colour, so on anything with internal
    structure - a drink can with a label band, a drone with a bright top -
    a click in the middle returns that internal region and the box comes back
    smaller than the object. This one ignores what is inside: it floods the
    *background* inward from the edge of the search window through everything
    that is not an edge, and whatever the flood cannot reach is inside the
    outline.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # One flood from a padded corner reaches all background-connected free
    # space, so the outline only has to be closed, not centred or convex.
    pad = cv2.copyMakeBorder(edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    mask = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)
    try:
        cv2.floodFill(pad, mask, (0, 0), 128)
    except cv2.error:
        return None
    inner = pad[1:-1, 1:-1]
    # Everything the background flood could NOT reach: the outline and all
    # the free space behind it. Taking only the free space would split the
    # object whenever an internal edge runs right across it - a label band on
    # a can does exactly that, and the click, being in the band, then got the
    # band. Keeping the edge pixels in the mask holds the pieces together.
    enclosed = (inner != 128).astype(np.uint8) * 255
    if cv2.countNonZero(enclosed) < 12:
        return None                       # outline leaked: nothing is enclosed

    n, lab, stats, _ = cv2.connectedComponentsWithStats(enclosed, 8)
    which = lab[py, px] if 0 <= py < lab.shape[0] and 0 <= px < lab.shape[1] else 0
    if which == 0:
        # the click landed in background free space - take the nearest
        # enclosed component rather than giving up
        y0, y1 = max(0, py - 6), min(lab.shape[0], py + 7)
        x0, x1 = max(0, px - 6), min(lab.shape[1], px + 7)
        near = lab[y0:y1, x0:x1]
        near = near[near > 0]
        if near.size == 0:
            return None
        which = int(np.bincount(near).argmax())
    if which <= 0 or which >= n:
        return None
    x, y, bw, bh, _area = stats[which]
    # the enclosed region stops just inside the outline; give back the couple
    # of pixels the edge band itself occupies
    return (float(x + ox - 2), float(y + oy - 2), float(bw + 4), float(bh + 4))


def _edge_box(roi, px, py, ox, oy):
    """Smallest closed edge contour that encloses the click."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(gray))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, lo, hi)
    # join the gaps, or an outline with one break encloses nothing
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        if cv2.contourArea(c) < 40:
            continue
        if cv2.pointPolygonTest(c, (float(px), float(py)), False) < 0:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if best is None or bw * bh < best[2] * best[3]:
            best = (x, y, bw, bh)
    if best is None:
        return None
    x, y, bw, bh = best
    return (float(x + ox), float(y + oy), float(bw), float(bh))


def snap_selection(frame, box):
    """Shrink a loose drag to the object inside it. -> (box, snapped?).

    Operators drag generously - a box 2-3x the object, with the desk in it.
    The identity is snapshotted from exactly what is in the box, so a loose
    drag makes an identity that is mostly BACKGROUND - and the background
    does not leave when the object does. Measured on the reported scene: a
    3x drag round a mug held 'lock' on the empty desk for 90 of 90 frames
    after the mug was removed, at match 0.97+, because 97 % of what it was
    matching really was still there.

    Only ever shrinks: the edge estimate must lie inside the drag (a little
    slack) and be meaningfully smaller, else the operator's box stands.
    """
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    slack_x, slack_y = 0.1 * w, 0.1 * h
    best = None
    # The edge readings are noise-sensitive - an outline that closes at one
    # working scale can fail to close at another - so try a few hint sizes
    # and keep the largest estimate that actually shrinks the drag.
    for hint in (max(w, h) / 2.0, max(w, h) / 3.0, min(w, h) / 2.0):
        est = estimate_object_box(frame, cx, cy, hint=hint)
        if est is None:
            continue
        ex, ey, ew, eh = est
        inside = (ex >= x - slack_x and ey >= y - slack_y
                  and ex + ew <= x + w + slack_x and ey + eh <= y + h + slack_y)
        if not inside or ew * eh > 0.6 * w * h:
            continue
        if best is None or ew * eh > best[2] * best[3]:
            best = est
    if best is None:
        return box, False
    return best, True


def iou(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def create_cv_tracker(name):
    """CSRT / KCF / MOSSE across the OpenCV 4.5 -> 4.10 API shuffle."""
    name = (name or "CSRT").upper()
    candidates = {
        "CSRT": ("TrackerCSRT_create",),
        "KCF": ("TrackerKCF_create",),
        "MOSSE": ("TrackerMOSSE_create", "legacy.TrackerMOSSE_create"),
        "MIL": ("TrackerMIL_create",),
    }.get(name, ("TrackerCSRT_create",))
    for attr in candidates:
        for root in (cv2, getattr(cv2, "legacy", None)):
            if root is None:
                continue
            fn = getattr(root, attr.split(".")[-1], None)
            if fn is not None:
                return fn()
    return cv2.TrackerCSRT_create()


# ------------------------------------------------------------ appearance

class Appearance:
    """Colour histogram + normalised grey template for one view of a target."""

    __slots__ = ("hist", "templ", "gray", "size", "weak_color", "contrast_frac")

    def __init__(self, patch):
        h, w = patch.shape[:2]
        self.size = (w, h)

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        # Ignore washed-out / near-black pixels: their hue is noise and would
        # make two unrelated objects look similar.
        mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))
        # A view with almost no saturated pixels (a white mug, a grey drone
        # against sky) has a colour histogram that says only "this is pale" -
        # and every pale patch in the scene says the same. Remember that, so
        # compare() can stop letting a meaningless colour score carry the
        # verdict: measured on the reported desk, the EMPTY spot behind a
        # removed white mug scored colour 0.83-0.99 against it, which at the
        # normal 45 % colour weight kept the combined score above the
        # keep-following bar with the object not in the scene at all.
        self.weak_color = cv2.countNonZero(mask) < 0.12 * mask.size
        hist = cv2.calcHist([hsv], [0, 1], None if self.weak_color else mask,
                            [24, 24], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        self.hist = hist

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # How much of the patch actually differs from its own border. The
        # border is whatever surrounds the object - for a flying target,
        # sky - so this is the fraction of the box occupied by *object*.
        # A thin-armed drone is only ~20 % object; the other 80 % of its
        # histogram and template describe sky, and any smooth sky patch
        # therefore scores deceptively well against them. What sky cannot
        # fake is this number: a cloud is smooth against its own border.
        if gray.shape[0] >= 8 and gray.shape[1] >= 8:
            border = np.concatenate([
                gray[:2].ravel(), gray[-2:].ravel(),
                gray[:, :2].ravel(), gray[:, -2:].ravel()]).astype(np.float32)
            self.contrast_frac = float(
                np.mean(np.abs(gray.astype(np.float32) - border.mean())
                        > CONTRAST_PX))
        else:
            self.contrast_frac = 1.0     # too small to judge - never veto
        self.templ = cv2.resize(gray, (TEMPL_PX, TEMPL_PX)).astype(np.float32)
        k = SEARCH_TEMPL_PX / float(max(w, h))
        self.gray = cv2.resize(
            gray, (max(6, int(round(w * k))), max(6, int(round(h * k))))
        ).astype(np.float32)

    def compare(self, other):
        """-> (combined, colour, structure), each 0..1.

        The colour weight belongs to this view (the identity side): if this
        view is washed out its colour score cannot tell one pale thing from
        another, so shape decides. Blur tolerates the shift - a 15 px motion
        blur still grey-correlates at 0.93 (measured), so shape-weighted
        keep-following does not drop a shaken lock.
        """
        c = cv2.compareHist(self.hist, other.hist, cv2.HISTCMP_CORREL)
        c = max(0.0, min(1.0, float(c)))
        s = float(cv2.matchTemplate(self.templ, other.templ,
                                    cv2.TM_CCOEFF_NORMED)[0][0])
        s = max(0.0, min(1.0, s))
        w = 0.15 if self.weak_color else 0.45
        return w * c + (1.0 - w) * s, c, s


class Identity:
    """Everything that says 'this is the object the operator picked'."""

    def __init__(self, frame, box):
        patch = patch_of(frame, box)
        if patch is None:
            raise ValueError("selection too small")
        first = Appearance(patch)
        self.anchor = first          # the original view, never replaced
        self.gallery = [first]
        self.recent = first          # slowly-adapting view, for tracking only
        self.size0 = (float(box[2]), float(box[3]))
        self.aspect0 = float(box[2]) / max(1.0, float(box[3]))
        self._last_gallery_t = time.time()

    def score(self, frame, box):
        patch = patch_of(frame, box)
        if patch is None:
            return None
        cand = Appearance(patch)
        best = (0.0, 0.0, 0.0)
        for view in self.gallery:
            r = view.compare(cand)
            if r[0] > best[0]:
                best = r
        rec = self.recent.compare(cand)
        # Against the operator's original view alone, with no help from
        # anything the tracker has taught itself since. This is the only
        # reading that cannot drift, so it is the one the gates use.
        org = self.anchor.compare(cand)
        return {
            "cand": cand,
            "anchor": best[0], "anchor_color": best[1], "anchor_struct": best[2],
            "recent": rec[0],
            "origin": org[0], "origin_color": org[1], "origin_struct": org[2],
            "combined": max(best[0], 0.5 * (best[0] + rec[0])),
        }

    def adapt(self, sc):
        """Fold a confirmed view in - never on a weak match, or we drift.

        Both gates are against the original view, not the gallery. Gating on
        the gallery is what let the identity walk: a view need only resemble
        the last thing absorbed, and a chain of such steps arrives anywhere.
        """
        if sc["anchor"] < RECENT_SIM_MIN or sc["origin"] < ORIGIN_RECENT_MIN:
            return
        self.recent = sc["cand"]
        now = time.time()
        if sc["anchor"] >= 0.72 and sc["origin"] >= ORIGIN_GALLERY_MIN \
                and (now - self._last_gallery_t) >= GALLERY_MIN_GAP_S:
            self.gallery.append(sc["cand"])
            if len(self.gallery) > GALLERY_MAX:
                # keep the original view at index 0, retire the oldest after it
                del self.gallery[1]
            self._last_gallery_t = now

    def size_plausible(self, box, tol=2.6):
        w, h = float(box[2]), float(box[3])
        if w < 6 or h < 6:
            return False
        sw = w / self.size0[0]
        sh = h / self.size0[1]
        if not (1.0 / tol <= sw <= tol and 1.0 / tol <= sh <= tol):
            return False
        aspect = w / max(1.0, h)
        return 1.0 / 2.2 <= (aspect / self.aspect0) <= 2.2


# --------------------------------------------------------------- tracker

DEFAULTS = {
    "tracker": "CSRT",
    "track_thresh": 0.42,       # keep-following score
    "relock_thresh": 0.60,      # stricter score needed to re-acquire
    "relock_color_min": 0.35,
    "relock_struct_min": 0.34,
    "relock_frames": 3,         # consecutive agreeing candidates
    "relock_margin": 0.08,      # gap to the runner-up, else it is ambiguous
    "coast_s": 0.5,             # keep aiming on prediction this long
    "give_up_s": 0.0,           # 0 = search forever
    "max_jump_frac": 0.22,      # of frame diagonal, per frame
    "stabilise": True,          # vibration compensation
    "aim_smooth": 0.45,
    "distractor_guard": True,   # remember look-alikes seen next to the target
    "distractor_scan_every": 10,
    "distractor_ttl": 10.0,
}


class TrackState:
    __slots__ = ("state", "box", "center", "aim", "visible", "score",
                 "color", "struct", "lost_s", "search_box", "note")

    def __init__(self):
        self.state = "idle"
        self.box = None
        self.center = None
        self.aim = None
        self.visible = False
        self.score = 0.0
        self.color = 0.0
        self.struct = 0.0
        self.lost_s = 0.0
        self.search_box = None
        self.note = ""


class LockTracker:
    def __init__(self, cfg=None, log=None):
        self.cfg = dict(DEFAULTS)
        if cfg:
            self.cfg.update({k: v for k, v in cfg.items() if k in DEFAULTS})
        self.log = log or (lambda _m: None)
        self.reset()

    # ---- lifecycle ----
    def reset(self):
        self.state = "idle"
        self.identity = None
        self.cv_tracker = None
        self.box = None
        self.last_size = None
        self.center = None
        self.aim = None
        self.vel = (0.0, 0.0)
        self.score = 0.0
        self.color = 0.0
        self.struct = 0.0
        self.lost_since = None
        self.relock_hits = 0
        self.relock_last = None
        self.distractors = []
        self.relock_reason = ""
        # What the loss location looks like WITHOUT the object - snapshotted
        # on the first missed frame. An object that was carried away leaves
        # its background behind, and on a pale scene that background can
        # out-score the re-lock floors (measured 0.73-0.94 against a white
        # mug's identity). A candidate near the loss point that looks more
        # like this snapshot than like the target IS the background.
        self.bg_view = None
        self.bg_center = None
        # The whole frame as it looked when the search began - i.e. with the
        # object already gone. Anything that matches its own coordinates in
        # this frame is scenery.
        self.loss_frame = None
        self._prev_small = None
        self._t_last = time.time()
        self._miss_since = None
        self._frame_i = 0
        self._size_fix = 0

    def clear(self):
        self.reset()

    def select(self, frame, box):
        """Lock onto the operator's box. box = (x, y, w, h) in pixels."""
        h, w = frame.shape[:2]
        box = clip_box(box, w, h, min_size=12)
        if box[2] < 12 or box[3] < 12:
            self.log("Selection too small - drag a bigger box.")
            return False
        try:
            self.identity = Identity(frame, box)
        except ValueError:
            return False
        self.cv_tracker = create_cv_tracker(self.cfg["tracker"])
        self.cv_tracker.init(frame, tuple(int(round(v)) for v in box))
        self.box = box
        self.last_size = (box[2], box[3])
        self.center = box_center(box)
        self.aim = self.center
        self.vel = (0.0, 0.0)
        self.state = "lock"
        self.lost_since = None
        self.relock_hits = 0
        self.relock_last = None
        self.relock_reason = ""
        self.bg_view = None
        self.bg_center = None
        self.loss_frame = None
        self._miss_since = None
        self._t_last = time.time()
        self.log("LOCKED on %dx%d target at (%d, %d)"
                 % (box[2], box[3], self.center[0], self.center[1]))
        return True

    @property
    def locked(self):
        return self.state != "idle"

    # ---- per-frame ----
    def update(self, frame, proposals=(), now=None):
        now = time.time() if now is None else now
        dt = max(1e-3, min(now - self._t_last, 0.5))
        self._t_last = now
        fh, fw = frame.shape[:2]

        shift = self._global_shift(frame) if self.cfg["stabilise"] else (0.0, 0.0)

        st = TrackState()
        if self.state == "idle":
            st.state = "idle"
            return st

        # where the target should be now, shake included
        pred = (self.center[0] + self.vel[0] * dt + shift[0],
                self.center[1] + self.vel[1] * dt + shift[1])
        # An object that leaves the frame must not drag the prediction off
        # with it: the search region is built round this point, and a point
        # outside the frame collapses it to nothing - so the search never
        # looks at the edge the object will come back in through. Park it on
        # the border instead.
        edge = 0.5 * max(self.identity.size0)
        pred = (min(max(pred[0], -edge), fw + edge),
                min(max(pred[1], -edge), fh + edge))

        if self.state in ("lock", "coast"):
            self._update_locked(frame, pred, shift, dt, now, fw, fh)
        if self.state in ("coast", "search"):
            # Look for it from the first frame it goes missing rather than
            # waiting out the coast timer - a target that jumps clear of the
            # correlation tracker is back under the crosshair in a few frames
            # instead of the best part of a second. Coast vs search only
            # decides what the mount does meanwhile.
            self._update_search(frame, pred, dt, now, fw, fh, proposals,
                                drifted=(self.state == "coast"))

        st.state = self.state
        st.box = self.box
        st.center = self.center
        st.aim = self.aim
        st.visible = self.state == "lock"
        st.score = self.score
        st.color = self.color
        st.struct = self.struct
        st.lost_s = 0.0 if self.lost_since is None else (now - self.lost_since)
        st.search_box = self._search_region(fw, fh, st.lost_s) \
            if self.state in ("search", "coast") else None
        # Why the re-lock has not happened yet. Without this the operator sees
        # a box on the object and a tracker that will not take it, with no way
        # to tell a score just under the floor from a look-alike veto.
        st.note = self.relock_reason if self.state in ("search", "coast") else ""
        return st

    # ---- internals ----
    def _global_shift(self, frame):
        """Camera shake between frames, in full-resolution pixels."""
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        small = small.astype(np.float32)
        prev, self._prev_small = self._prev_small, small
        if prev is None:
            return (0.0, 0.0)
        try:
            (dx, dy), response = cv2.phaseCorrelate(prev, small)
        except cv2.error:
            return (0.0, 0.0)
        if response < 0.12:      # no agreement - probably a scene change
            return (0.0, 0.0)
        dx *= frame.shape[1] / 160.0
        dy *= frame.shape[0] / 90.0
        # A big bright object jumping across the frame can dominate the
        # correlation and be reported as camera motion - which would drag the
        # prediction (and the mount) toward whatever moved. Real shake, and
        # even a fast slew, stays small between frames at 30 fps, so anything
        # this large is the scene lying to us: ignore it.
        if abs(dx) > MAX_SHIFT_FRAC * frame.shape[1] or \
                abs(dy) > MAX_SHIFT_FRAC * frame.shape[0]:
            return (0.0, 0.0)
        return (dx, dy)

    def _accept(self, frame, box, shift, dt, now, learn_size=True):
        self.box = box
        # remember the size we last saw it at, so a search after a loss looks
        # for the object as it is now, not as it was when first selected -
        # but never learn a size off a box that is hanging over the edge of
        # the frame. That box is cropped, not smaller, and believing it sent
        # the whole search hunting for a half-size object it never found.
        if learn_size:
            self.last_size = (box[2], box[3])
        new_center = box_center(box)
        # velocity relative to the scene, not to the shaking camera
        vx = (new_center[0] - self.center[0] - shift[0]) / dt
        vy = (new_center[1] - self.center[1] - shift[1]) / dt
        lim = frame.shape[1] * 1.5
        vx = max(-lim, min(lim, vx))
        vy = max(-lim, min(lim, vy))
        a = 0.35
        self.vel = ((1 - a) * self.vel[0] + a * vx,
                    (1 - a) * self.vel[1] + a * vy)
        self.center = new_center
        s = self.cfg["aim_smooth"]
        self.aim = (self.aim[0] + s * (new_center[0] - self.aim[0]),
                    self.aim[1] + s * (new_center[1] - self.aim[1]))
        self.state = "lock"
        self.lost_since = None
        self._miss_since = None
        self.relock_hits = 0
        self.bg_view = None
        self.bg_center = None
        self.loss_frame = None

    def _looks_like_background(self, cand_app, cand_box, id_score):
        """The score against the loss-site background, if it wins. -> float|None.

        Only meaningful near where the object vanished - the snapshot says
        nothing about the rest of the frame. The margin keeps it honest: a
        tie goes to the identity, so this only ever refuses a candidate the
        background CLEARLY explains better than the target does.
        """
        if self.bg_view is None or self.bg_center is None:
            return None
        cc = box_center(cand_box)
        if math.hypot(cc[0] - self.bg_center[0], cc[1] - self.bg_center[1]) \
                > 1.5 * max(self.identity.size0):
            return None
        bg = self.bg_view.compare(cand_app)[0]
        return bg if bg > id_score + 0.05 else None

    def _is_static_background(self, frame_gray, box):
        """Has this exact spot simply not changed since the object left?

        The strongest possible evidence against a candidate: if its patch
        matches the same coordinates in the search-entry frame, it is a
        piece of scene that was already there WITHOUT the object - however
        well it happens to score against the identity. On the reported pale
        desk, background candidates cleared every appearance floor (0.73-
        0.94); no threshold fixes that, but "has it changed" does. A small
        alignment window absorbs camera shake.
        """
        if self.loss_frame is None:
            return False
        fh, fw = self.loss_frame.shape[:2]
        x, y, w, h = [int(round(v)) for v in box]
        x = max(0, x)
        y = max(0, y)
        w = min(w, fw - x)
        h = min(h, fh - y)
        if w < 8 or h < 8:
            return False
        patch = frame_gray[y:y + h, x:x + w].astype(np.float32)
        pad = STATIC_BG_PAD
        wx0, wy0 = max(0, x - pad), max(0, y - pad)
        window = self.loss_frame[wy0:min(fh, y + h + pad),
                                 wx0:min(fw, x + w + pad)].astype(np.float32)
        if window.shape[0] < h or window.shape[1] < w:
            return False
        try:
            res = cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            return False
        return float(res.max()) >= STATIC_BG_NCC

    def _coast(self, frame, pred, now):
        if self._miss_since is None and self.box is not None:
            # First missed frame: remember what this spot looks like now,
            # IF the target is actually gone from it. A "miss" also happens
            # while the object is still standing there - motion blur, a
            # passing hand, the correlation tracker simply blinking - and a
            # snapshot taken then contains the OBJECT, after which the
            # static-scenery test refuses the object itself as "unchanged
            # scenery" and the search stays orange forever on a target in
            # plain view (reproduced: tracker blinked for 20 frames over a
            # static object, never re-locked). So look first: if the lost
            # box still resembles the target on either reading, this is a
            # blink, and the background guards must stay unarmed - the
            # ordinary floors and the look-alike memory still apply.
            sc_miss = self.identity.score(frame, self.box)
            target_gone = sc_miss is None or (
                sc_miss["origin"] < self.cfg["track_thresh"]
                and sc_miss["anchor_struct"] < self.cfg["relock_struct_min"])
            if target_gone:
                patch = patch_of(frame, self.box)
                if patch is not None:
                    self.bg_view = Appearance(patch)
                    self.bg_center = box_center(self.box)
                # The whole frame too, for the static-scenery test - and
                # ONLY now, never refreshed later in the episode: a later
                # frame may have the occluder gone and the object back, and
                # a snapshot containing the object would make the test
                # refuse the object itself. Search starts looking from the
                # very first missed frame, so its guard has to exist from
                # the very first missed frame as well - captured at search
                # entry (0.5 s in), the measured pale-desk false re-lock
                # was already done at 0.1 s.
                self.loss_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.center = pred
        s = self.cfg["aim_smooth"] * 0.5
        self.aim = (self.aim[0] + s * (pred[0] - self.aim[0]),
                    self.aim[1] + s * (pred[1] - self.aim[1]))
        if self.box is not None:
            self.box = (pred[0] - self.box[2] / 2.0,
                        pred[1] - self.box[3] / 2.0, self.box[2], self.box[3])
        # a coasting target's velocity must decay or it flies off the frame
        self.vel = (self.vel[0] * 0.88, self.vel[1] * 0.88)
        if self._miss_since is None:
            self._miss_since = now
        if self.lost_since is None:
            self.lost_since = now
        self.state = "coast"
        if (now - self._miss_since) >= self.cfg["coast_s"]:
            self.state = "search"
            self.relock_hits = 0
            self.relock_last = None
            self.log("Target lost - searching for the same object...")

    def _update_locked(self, frame, pred, shift, dt, now, fw, fh):
        ok, raw = False, None
        try:
            ok, raw = self.cv_tracker.update(frame)
        except cv2.error:
            ok = False
        if ok and raw is not None:
            box = clip_box(tuple(float(v) for v in raw), fw, fh)
            sc = self.identity.score(frame, box)
            # ---- scale wander: a resize must prove itself ----
            # CSRT re-estimates the box scale every frame, and on a plain
            # object the estimate wanders: observed on the rig going to a
            # half-size patch and later to ~3x, swallowing the jar and the
            # clutter. Each wander was accepted (still above track_thresh),
            # LEARNED (last_size, adapt), and a 3x box is mostly background
            # - at which point removing the object changes only a third of
            # the pixels and the removal guards, which all key off a loss,
            # never fire. So a box that changed size by more than 10 % is
            # compared against a same-centre box at the PREVIOUS size, and
            # the better identity match wins: a resize that only swept in
            # background loses that comparison, a genuine approach/recede
            # wins it. If the correlation tracker keeps insisting on a size
            # the evidence keeps refusing, it is re-initialised at the
            # corrected box - its internal state is what ballooned.
            # The trigger is deliberately tight (2 %): last_size follows
            # every accepted box, so a looser gate lets a per-frame creep
            # ratchet under it - 6 %/frame slipped straight through a 10 %
            # gate and tripled the box anyway. A real approach changes
            # apparent size ~2-3 %/frame at most, and a genuine resize wins
            # its proof, so proving often costs ~1 ms and no correctness.
            if sc is not None and self.last_size is not None:
                lw, lh = self.last_size
                if abs(box[2] - lw) > 0.02 * lw or abs(box[3] - lh) > 0.02 * lh:
                    c0 = box_center(box)
                    alt = clip_box((c0[0] - lw / 2.0, c0[1] - lh / 2.0, lw, lh),
                                   fw, fh)
                    sc_alt = self.identity.score(frame, alt)
                    if sc_alt is not None and \
                            sc_alt["combined"] >= sc["combined"]:
                        box, sc = alt, sc_alt
                        self._size_fix += 1
                        if self._size_fix >= 3:
                            self.cv_tracker = create_cv_tracker(
                                self.cfg["tracker"])
                            self.cv_tracker.init(
                                frame, tuple(int(round(v)) for v in box))
                            self._size_fix = 0
                    else:
                        self._size_fix = 0
                else:
                    self._size_fix = 0
            if sc is not None and self.identity.size_plausible(box):
                cx, cy = box_center(box)
                jump = math.hypot(cx - pred[0], cy - pred[1])
                diag = math.hypot(fw, fh)
                # allow a bigger jump the longer we have been coasting
                budget = diag * self.cfg["max_jump_frac"] * (1.0 + 4.0 * dt)
                if self.state == "coast":
                    budget *= 1.8
                # Two gates, and the second one is the one that stops a swap:
                # the box must still look like what the operator picked. A
                # correlation tracker sliding onto the clutter next door
                # clears track_thresh comfortably once the identity has
                # absorbed a few of its own slides, and this path has no
                # rival guard and no look-alike memory to catch that - so it
                # has to fail here and drop into `search`, where they live.
                good = (sc["combined"] >= self.cfg["track_thresh"]
                        and sc["origin"] >= ORIGIN_KEEP_MIN)
                if good and self.state == "coast":
                    # While coasting nothing visible is being followed, so a
                    # box the correlation tracker offers is a RE-acquisition
                    # and must clear re-lock-grade floors, not the keep-lock
                    # bar. Measured on the rig's pale-mug scene: with only
                    # the keep-lock bar, CSRT drifted onto a white box and
                    # re-captured the lock at 0.48 combined / 0.26 shape -
                    # numbers the search path would have refused out of
                    # hand. Colour alone (0.75) carried the combined score,
                    # which is exactly what the separate floors exist for.
                    # And it must be WHERE THE PREDICTION IS: coasting means
                    # aiming at a dead-reckoned guess, and a box that
                    # teleports an object-width away (130 px, measured) is
                    # not "kept following it through the blur" - it is a
                    # fresh acquisition and belongs to the search path,
                    # which confirms over frames and consults the rival and
                    # look-alike guards.
                    near = math.hypot(cx - pred[0], cy - pred[1]) <= \
                        max(self.identity.size0) + math.hypot(*self.vel) * dt
                    good = (near
                            and sc["combined"] >= self.cfg["relock_thresh"]
                            and sc["anchor_color"] >= self.cfg["relock_color_min"]
                            and sc["anchor_struct"] >= self.cfg["relock_struct_min"]
                            and sc["cand"].contrast_frac >= self._contrast_floor()
                            and self._looks_like_background(
                                sc["cand"], box, sc["combined"]) is None
                            and not (self.cfg["distractor_guard"]
                                     and self._is_distractor(box, now)))
                # an excellent match is allowed to teleport: that is the
                # object genuinely jumping, not the tracker sliding away
                if good and (jump <= budget or sc["anchor"] >= 0.68):
                    self.score = sc["combined"]
                    self.color = sc["anchor_color"]
                    self.struct = sc["anchor_struct"]
                    # A box against the border is a cropped view: keep
                    # following it, but do not learn size or appearance from
                    # it or the identity drifts to half an object.
                    cropped = touches_edge(box, fw, fh)
                    if not cropped:
                        self.identity.adapt(sc)
                    self._accept(frame, box, shift, dt, now,
                                 learn_size=not cropped)
                    self._frame_i += 1
                    # Scan for look-alikes EVERY frame right after the lock,
                    # then drop to the cheap schedule. A look-alike needs two
                    # sightings to count, so with only the periodic scan a
                    # lock shorter than 2 x distractor_scan_every frames
                    # never blacklisted anything - and a short lock is
                    # exactly when the operator moves the object and the
                    # search needs the blacklist most. Measured: a pale box
                    # scoring 0.73 against a pale mug was refused after a
                    # 30-frame lock (3 scans) and locked after a 15-frame
                    # one (1 scan). Now it has its 2 hits by frame 2.
                    if self.cfg["distractor_guard"] and \
                            (self._frame_i <= EARLY_DISTRACTOR_FRAMES
                             or self._frame_i % max(1, self.cfg["distractor_scan_every"]) == 0):
                        self._scan_distractors(frame, fw, fh, now)
                    return
        self._coast(frame, pred, now)

    # ---- look-alike memory ----
    # Appearance alone cannot separate two identical objects. What can is
    # history: anything that was standing somewhere else *while* we had the
    # real target is, by definition, not the target. Those places are
    # remembered, so a search after a loss will not settle on one of them.
    def _scan_distractors(self, frame, fw, fh, now):
        self._expire_distractors(now)
        cands = self._candidates(frame, (0.0, 0.0, float(fw), float(fh)), (), fw, fh)
        thresh = self.cfg["relock_thresh"] * 0.8
        for c in cands:
            if c["score"] < thresh:
                continue
            if self.box is not None and iou(c["box"], self.box) > 0.10:
                continue
            # Exclusion zone round our own target: a template peak sitting
            # half on it is a sidelobe of the target, not a rival, and
            # blacklisting it would later block the real re-lock. "Half on
            # it" is an overlap property, so test overlap with the target
            # box inflated by half - not a big centre-distance radius. The
            # old 1.6 x size radius also swallowed a genuinely different
            # pale object sitting 1.3 object-widths away, so it was never
            # blacklisted and a later search took it (measured: scans saw
            # it at 0.59-0.63 on every frame and skipped it every time).
            if self.box is not None:
                tx, ty, tw, th = self.box
                infl = (tx - tw * 0.25, ty - th * 0.25, tw * 1.5, th * 1.5)
                if iou(c["box"], infl) > 0.0:
                    continue
            self._remember_distractor(c["box"], now)

    def _remember_distractor(self, box, now):
        for d in self.distractors:
            if self._same_place(box, d["box"]):
                d["box"] = box
                d["last_seen"] = now
                d["hits"] += 1
                return
        if len(self.distractors) < 12:
            self.distractors.append({"box": box, "last_seen": now, "hits": 1})

    def _same_place(self, a, b, loose=False):
        # Recording uses the tight test so two nearby look-alikes stay
        # separate entries. Rejection uses the loose one: a template peak
        # that merely clips a known look-alike still scores well on colour,
        # and must not be allowed through as "something new".
        if iou(a, b) > (0.10 if loose else 0.35):
            return True
        ca, cb = box_center(a), box_center(b)
        reach = (1.3 if loose else 0.7) * max(self.identity.size0)
        return math.hypot(ca[0] - cb[0], ca[1] - cb[1]) < reach

    def _expire_distractors(self, now):
        ttl = self.cfg["distractor_ttl"]
        self.distractors = [d for d in self.distractors
                            if (now - d["last_seen"]) <= ttl]

    def _is_distractor(self, box, now):
        for d in self.distractors:
            if d["hits"] < 2:
                continue          # a single sighting could have been noise
            if self._same_place(box, d["box"], loose=True):
                # Refresh the sighting but NEVER move the entry. Moving it
                # ("follow a moving look-alike") let the entry walk: each
                # refused peak dragged it a little, a chain of them carried
                # it from one pale rival onto another 130 px away, and the
                # first rival - still exactly where it was recorded - then
                # sat outside the loose reach and completed a re-lock.
                # An anchor that follows what it rejects is not an anchor;
                # same lesson as the identity walk. A look-alike that
                # genuinely moves far lays outside the reach and must then
                # beat the full re-lock gates instead - that is the lesser
                # risk.
                d["last_seen"] = now
                return True
        return False

    def _contrast_floor(self):
        """0.0 disarms the veto - an identity with no contrast of its own
        (pale object on a pale surface) cannot demand contrast of others."""
        a = self.identity.anchor.contrast_frac
        return CONTRAST_VETO_MIN if a >= 3.0 * CONTRAST_VETO_MIN else 0.0

    def _search_region(self, fw, fh, lost_s):
        w0, h0 = self.identity.size0
        grow = 2.2 + 2.2 * min(lost_s, 6.0)
        half_w = min(fw, w0 * grow)
        half_h = min(fh, h0 * grow)
        cx, cy = self.center
        x = max(0.0, cx - half_w)
        y = max(0.0, cy - half_h)
        x2 = min(float(fw), cx + half_w)
        y2 = min(float(fh), cy + half_h)
        return (x, y, x2 - x, y2 - y)

    def _update_search(self, frame, pred, dt, now, fw, fh, proposals,
                       drifted=False):
        if self.cfg["give_up_s"] > 0 and self.lost_since is not None \
                and (now - self.lost_since) > self.cfg["give_up_s"]:
            self.log("Gave up on the target after %.0f s." % self.cfg["give_up_s"])
            self.reset()
            return

        # keep drifting the last known position with the (decaying) velocity so
        # the search box follows an object that left the frame moving
        if not drifted:      # on a coast frame _coast() has already done this
            self.center = pred
            self.vel = (self.vel[0] * 0.9, self.vel[1] * 0.9)

        lost_s = 0.0 if self.lost_since is None else (now - self.lost_since)
        region = self._search_region(fw, fh, lost_s)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
            if self.loss_frame is not None else None
        n_static = [0]

        def look(wide):
            found = self._candidates(frame, region, proposals, fw, fh, lost_s,
                                     wide=wide)
            n = len(found)
            if self.cfg["distractor_guard"]:
                self._expire_distractors(now)
                kept = [c for c in found if not self._is_distractor(c["box"], now)]
                if len(kept) != len(found):
                    self.relock_last = None if not kept else self.relock_last
                found = kept
            if gray is not None:
                # Scenery that has not changed since the search began cannot
                # be the object coming back, however well it scores.
                kept = [c for c in found
                        if not self._is_static_background(gray, c["box"])]
                n_static[0] += len(found) - len(kept)
                if len(kept) != len(found):
                    self.relock_last = None if not kept else self.relock_last
                found = kept
            return found, n

        # Narrow scales first: an object that merely jumped is the same size a
        # frame later, and the extra scales only add rival peaks and slow that
        # recovery down. But if nothing here clears the bar, widen NOW rather
        # than waiting out WIDE_SEARCH_AFTER_S - an object that went further
        # away comes back smaller immediately, and at the old 0.65x floor the
        # search was looking for a size it no longer is. Escalating within the
        # frame keeps the fast jump recovery and stops losing the scale case.
        # The time-decayed floors (see RELAX_* above). Clamped so a user who
        # configures a bar BELOW the relax floor keeps their lower bar.
        relax = min(1.0, max(0.0, (lost_s - RELAX_AFTER_S)
                             / max(1e-3, RELAX_FULL_S - RELAX_AFTER_S)))
        eff_thresh = self.cfg["relock_thresh"] - relax * max(
            0.0, self.cfg["relock_thresh"]
            - max(self.cfg["track_thresh"] + 0.06, RELAX_SCORE_FLOOR))
        eff_color = self.cfg["relock_color_min"] - relax * max(
            0.0, self.cfg["relock_color_min"] - RELAX_COLOR_FLOOR)
        eff_struct = self.cfg["relock_struct_min"] - relax * max(
            0.0, self.cfg["relock_struct_min"] - RELAX_STRUCT_FLOOR)

        def clears_the_bar(c):
            return (c["score"] >= eff_thresh
                    and c["color"] >= eff_color
                    and c["struct"] >= eff_struct)

        cands, n_found = look(wide=False)
        # Widen when nothing clears ALL THREE floors, not merely the combined
        # score. A box at the wrong scale sweeps in background: its colour
        # stays high and drags the combined score onto the threshold while its
        # shape is ruined. Testing the score alone let that box look like a
        # good enough reason not to widen, and the scale case never recovered.
        if not any(clears_the_bar(c) for c in cands):
            wide_cands, wide_n = look(wide=True)
            # Merge rather than choose. The gate is three separate floors
            # (combined, colour, shape) and the two scale sets do not win on
            # the same one: a too-large box still scores well on colour while
            # its shape is ruined by the background it swept in. Picking a set
            # by combined score alone throws away the candidate that would
            # actually have passed.
            seen = list(cands)
            for c in wide_cands:
                if not any(iou(c["box"], k["box"]) > 0.6 for k in seen):
                    seen.append(c)
            cands, n_found = seen, max(n_found, wide_n)
        # The contrast-mass veto (see CONTRAST_VETO_MIN). Filters the LIST,
        # not just the winner, so a cloud outscoring the real target does not
        # cost the frame - the target underneath is still considered.
        cfloor = self._contrast_floor()
        if cands and cfloor > 0.0:
            kept = [c for c in cands if c["appearance"].contrast_frac >= cfloor]
            if not kept:
                self.relock_hits = 0
                self.relock_last = None
                self.relock_reason = (
                    "all %d candidates are featureless sky or background "
                    "(contrast < %.2f)" % (len(cands), cfloor))
                return
            cands = kept
        if not cands:
            self.relock_hits = 0
            self.relock_last = None
            if not n_found:
                self.relock_reason = "nothing in the search area looks like it"
            elif n_static[0]:
                self.relock_reason = (
                    "%d candidates are scenery unchanged since the loss"
                    % n_static[0])
            else:
                self.relock_reason = (
                    "all %d candidates are known look-alikes" % n_found)
            return

        cands.sort(key=lambda c: -c["score"])
        best = self._refine_candidate(frame, cands[0], fw, fh)
        # A rival is another OBJECT, not another peak on the same one. The
        # multi-scale search puts several boxes on one target - a big one and
        # a small one on the same patch overlap too little for IoU to call
        # them the same, and the ambiguity guard then vetoed every re-lock
        # for as long as the target was in view. Anything centred within
        # about an object's width of the best candidate is that same object.
        reach = 0.75 * max(best["box"][2], best["box"][3],
                           *self.identity.size0)
        bc = box_center(best["box"])
        rival = None
        for c in cands[1:]:
            cc = box_center(c["box"])
            if iou(c["box"], best["box"]) >= 0.3 or \
                    math.hypot(cc[0] - bc[0], cc[1] - bc[1]) < reach:
                continue
            rival = c
            break

        self.score = best["score"]
        self.color = best["color"]
        self.struct = best["struct"]

        # Entry to confirmation needs the full threshold; SUSTAINING it may
        # ride 0.05 under. The score of a real candidate wobbles about that
        # much frame to frame from sensor noise alone, and a hard floor with
        # a hard reset made confirmation a coin-flip for anything hovering
        # near it - measured on the pale-mug scene: 0.58-0.63 around a 0.60
        # floor, hits resetting to zero on every dip, re-lock sometimes
        # never completing. The colour and shape floors stay hard.
        need = eff_thresh - (0.05 if self.relock_hits > 0 else 0.0)
        if best["score"] < need \
                or best["color"] < eff_color \
                or best["struct"] < eff_struct:
            self.relock_hits = 0
            self.relock_last = None
            short = []
            if best["score"] < need:
                short.append("match %.2f<%.2f" % (best["score"], need))
            if best["color"] < eff_color:
                short.append("colour %.2f<%.2f" % (best["color"], eff_color))
            if best["struct"] < eff_struct:
                short.append("shape %.2f<%.2f" % (best["struct"], eff_struct))
            self.relock_reason = "best candidate too weak: " + ", ".join(short)
            return
        if rival is not None and (best["score"] - rival["score"]) < self.cfg["relock_margin"]:
            # two things look alike - refuse to guess which one is ours
            self.relock_hits = 0
            self.relock_last = None
            self.relock_reason = (
                "two objects score alike (%.2f vs %.2f, need %.2f apart)"
                % (best["score"], rival["score"], self.cfg["relock_margin"]))
            return
        bg = self._looks_like_background(best["appearance"], best["box"],
                                         best["score"])
        if bg is not None:
            # It matches what this spot looked like AFTER the object left
            # better than it matches the object. That is the desk, not the
            # target - on a pale scene it can clear every floor above.
            self.relock_hits = 0
            self.relock_last = None
            self.relock_reason = (
                "best candidate matches the empty background better "
                "(bg %.2f > match %.2f)" % (bg, best["score"]))
            return

        # The candidate must also stay put across frames. "Put" has to be
        # generous enough for an object that is still moving: the whole point
        # of a re-lock is usually that the thing got away, and at 30 fps a fast
        # target crosses well over its own width between frames. Allow the
        # distance it could plausibly have travelled since the last look.
        if self.relock_last is not None:
            drift = math.hypot(best["box"][0] - self.relock_last[0],
                               best["box"][1] - self.relock_last[1])
            allow = max(self.identity.size0) * 1.5 + \
                math.hypot(*self.vel) * dt
            if drift > allow:
                self.relock_hits = 0
        self.relock_last = best["box"]
        self.relock_hits += 1
        if self.relock_hits < self.cfg["relock_frames"]:
            self.relock_reason = "confirming (%d of %d frames, match %.2f)" % (
                self.relock_hits, self.cfg["relock_frames"], best["score"])

        if self.relock_hits >= self.cfg["relock_frames"]:
            box = clip_box(best["box"], fw, fh)
            self.cv_tracker = create_cv_tracker(self.cfg["tracker"])
            self.cv_tracker.init(frame, tuple(int(round(v)) for v in box))
            self.box = box
            self.center = box_center(box)
            self.aim = self.center
            self.vel = (0.0, 0.0)
            self.state = "lock"
            self.lost_since = None
            self._miss_since = None
            self.relock_hits = 0
            self.relock_reason = ""
            self.bg_view = None
            self.bg_center = None
            self.loss_frame = None
            self.identity.recent = best["appearance"]
            self.log("Re-locked the same target (score %.2f) after %.1f s."
                     % (best["score"], lost_s))

    def _refine_candidate(self, frame, cand, fw, fh):
        """Re-align the winning box to the anchor template. -> cand (maybe new).

        The multi-scale search works on a ~40 px thumbnail, so its peaks are
        quantised to 2-3 real pixels - and on a plain object whose identity
        lives in thin features (a rim, a line of print) that misalignment
        alone costs 0.1-0.2 of template score. Measured on the pale-mug
        scene: the mug back in its own spot scored 0.55 against its own
        identity purely because every candidate framed it a few pixels off,
        and 0.55 sits under the 0.60 floor - so the re-lock the search
        existed for never happened. One full-resolution match in a small
        window around the winner fixes the framing; the result is kept only
        if it actually scores better.
        """
        x, y, w, h = [int(round(v)) for v in cand["box"]]
        tw = max(8, min(int(round(w)), fw))
        th = max(8, min(int(round(h)), fh))
        base = self.identity.anchor.templ            # 48x48 float32
        try:
            tmpl = cv2.resize(base, (tw, th))
            pad = 10
            wx0, wy0 = max(0, x - pad), max(0, y - pad)
            wx1, wy1 = min(fw, x + tw + pad), min(fh, y + th + pad)
            if wx1 - wx0 < tw or wy1 - wy0 < th:
                return cand
            window = cv2.cvtColor(frame[wy0:wy1, wx0:wx1],
                                  cv2.COLOR_BGR2GRAY).astype(np.float32)
            _mn, _mx, _ml, ml = cv2.minMaxLoc(
                cv2.matchTemplate(window, tmpl, cv2.TM_CCOEFF_NORMED))
        except cv2.error:
            return cand
        box = (float(wx0 + ml[0]), float(wy0 + ml[1]), float(tw), float(th))
        sc = self.identity.score(frame, box)
        if sc is None or sc["anchor"] <= cand["score"]:
            return cand
        return {"box": box, "score": sc["anchor"],
                "color": sc["anchor_color"], "struct": sc["anchor_struct"],
                "appearance": sc["cand"]}

    def _candidates(self, frame, region, proposals, fw, fh, lost_s=0.0,
                    wide=False):
        out = []
        boxes = list(self._template_boxes(frame, region, lost_s, wide))
        for p in proposals or ():
            box = clip_box(tuple(float(v) for v in p), fw, fh)
            if self.identity.size_plausible(box, tol=3.2):
                boxes.append(box)
        # drop near-duplicates before the (costlier) identity scoring
        keep = []
        for b in boxes:
            if not any(iou(b, k) > 0.6 for k in keep):
                keep.append(b)
        for box in keep[:24]:
            sc = self.identity.score(frame, box)
            if sc is None:
                continue
            out.append({"box": box, "score": sc["anchor"],
                        "color": sc["anchor_color"], "struct": sc["anchor_struct"],
                        "appearance": sc["cand"]})
        return out

    def _template_boxes(self, frame, region, lost_s=0.0, wide=False):
        """Multi-scale template match of the original view inside `region`."""
        rx, ry, rw, rh = [int(round(v)) for v in region]
        if rw < 12 or rh < 12:
            return
        sub = frame[ry:ry + rh, rx:rx + rw]
        if sub.size == 0:
            return
        gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        w0, h0 = self.identity.size0
        k = SEARCH_TEMPL_PX / float(max(w0, h0))
        sw = max(8, int(round(rw * k)))
        sh = max(8, int(round(rh * k)))
        small = cv2.resize(gray, (sw, sh)).astype(np.float32)

        base = self.identity.anchor.gray
        # follow the size the target drifted to while it was visible
        grew = max(0.4, min(2.5, self.last_size[0] / max(1.0, w0))) \
            if self.last_size else 1.0
        scales = SEARCH_SCALES_WIDE \
            if (wide or lost_s >= WIDE_SEARCH_AFTER_S) else SEARCH_SCALES
        for scale in [s * grew for s in scales]:
            tw = int(round(base.shape[1] * scale))
            th = int(round(base.shape[0] * scale))
            if tw < 6 or th < 6 or tw >= sw or th >= sh:
                continue
            tmpl = cv2.resize(base, (tw, th))
            res = cv2.matchTemplate(small, tmpl, cv2.TM_CCOEFF_NORMED)
            for _ in range(3):        # top peaks, with suppression between
                _mn, mx, _ml, ml = cv2.minMaxLoc(res)
                if mx < 0.2:
                    break
                px, py = ml
                yield (rx + px / k, ry + py / k, tw / k, th / k)
                x0 = max(0, px - tw // 2)
                y0 = max(0, py - th // 2)
                res[y0:py + th // 2 + 1, x0:px + tw // 2 + 1] = -1.0
