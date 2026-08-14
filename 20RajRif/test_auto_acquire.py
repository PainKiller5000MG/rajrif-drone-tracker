"""Headless checks for yolo_autoacquire.AutoAcquireDetector's confirmation
logic - the part that stands in for a missing "not a drone" class, since the
model is single-class and any false positive is still labelled "drone".

No real ultralytics/torch/weights needed: a fake model object stands in for
YOLO, so these run in any environment, including CI, per the same "there is
no other safety net" expectation as test_tracking.py / test_integration.py.

    py -3.11 test_auto_acquire.py
"""

import sys
import threading
import time

import numpy as np

import yolo_autoacquire as ya


class FakeBox:
    def __init__(self, xyxy):
        self.xyxy = [np.array(xyxy, dtype=float)]


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def make_detector(model, confirm_frames=3, max_box_area_frac=0.10,
                   poll_interval_s=0.01, iou_min=0.35):
    det = ya.AutoAcquireDetector.__new__(ya.AutoAcquireDetector)
    det.weights_path = "fake"
    det.conf_thresh = 0.8
    det.max_box_area_frac = max_box_area_frac
    det.confirm_frames = confirm_frames
    det.poll_interval_s = poll_interval_s
    det.iou_min = iou_min
    det.log = lambda _m: None
    det._frame_lock = threading.Lock()
    det._latest_frame = None
    det._result_lock = threading.Lock()
    det._confirmed_box = None
    det._streak_box = None
    det._streak_count = 0
    det._streak_last_t = None
    det._max_streak_gap_s = max(1.0, poll_interval_s * 8)
    det._stop_flag = threading.Event()
    det._thread = None
    det.ready = True
    det._model = model
    det._load_error = None
    return det


FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


def run_detector_for(det, seconds):
    det.submit_frame(FRAME)
    det.start()
    time.sleep(seconds)
    box = det.get_confirmed_box()
    det.stop()
    return box


def check_consistent_detection_confirms(_rec=None):
    class SteadyModel:
        def predict(self, frame, conf, verbose):
            return [FakeResult([FakeBox((100, 100, 250, 240))])]

    det = make_detector(SteadyModel(), confirm_frames=3)
    box = run_detector_for(det, 0.3)
    assert box is not None, "a steady, plausible detection never confirmed"
    assert box == (100.0, 100.0, 150.0, 140.0), "unexpected box: %r" % (box,)


def check_confirmed_box_clears_on_read(_rec=None):
    class SteadyModel:
        def predict(self, frame, conf, verbose):
            return [FakeResult([FakeBox((100, 100, 250, 240))])]

    det = make_detector(SteadyModel(), confirm_frames=2)
    det.submit_frame(FRAME)
    det.start()
    time.sleep(0.3)
    first = det.get_confirmed_box()
    second = det.get_confirmed_box()
    det.stop()
    assert first is not None, "expected a confirmed box on first read"
    assert second is None, "get_confirmed_box() must clear itself once read"


def check_oversized_box_never_confirms(_rec=None):
    # ~53% of a 1280x720 frame - the same order of magnitude as the
    # frame-spanning false positives observed on a webcam face (35-60%),
    # versus real drone detections measured at <=2.6% of frame.
    class FaceModel:
        def predict(self, frame, conf, verbose):
            return [FakeResult([FakeBox((0, 0, 700, 700))])]

    det = make_detector(FaceModel(), confirm_frames=3, max_box_area_frac=0.10)
    box = run_detector_for(det, 0.3)
    assert box is None, "oversized box should have been filtered out, got %r" % (box,)


def check_inconsistent_detections_never_confirm(_rec=None):
    class JumpyModel:
        def __init__(self):
            self.n = 0

        def predict(self, frame, conf, verbose):
            self.n += 1
            if self.n % 2 == 0:
                return [FakeResult([FakeBox((50, 50, 150, 140))])]
            return [FakeResult([FakeBox((900, 500, 1000, 590))])]

    det = make_detector(JumpyModel(), confirm_frames=3)
    box = run_detector_for(det, 0.3)
    assert box is None, "alternating unrelated boxes should never build a streak, got %r" % (box,)


def check_occasional_miss_still_confirms(_rec=None):
    """A real drone at real polling rates occasionally drops a single cycle
    (measured on real footage) even when genuinely still there - this must
    NOT block confirmation forever, only slow it down. Soft-decrement (-1
    per miss) instead of a hard reset to 0 is what makes this possible."""
    class FlickerModel:
        def __init__(self):
            self.n = 0

        def predict(self, frame, conf, verbose):
            self.n += 1
            if self.n % 3 == 0:
                return [FakeResult([])]
            return [FakeResult([FakeBox((100, 100, 250, 240))])]

    det = make_detector(FlickerModel(), confirm_frames=4)
    box = run_detector_for(det, 0.5)
    assert box is not None, "an occasional single miss should not permanently block confirmation"


def check_sustained_absence_fully_resets(_rec=None):
    """Unlike a single dropped cycle, many consecutive missed cycles must
    still fully clear the streak - otherwise a stale box could confirm
    arbitrarily long after the object actually left frame.

    Drives _process_frame() directly, one deterministic cycle at a time,
    rather than racing a real thread against sleep() - avoids the timing
    flakiness of trying to time-box phases against a background thread."""
    class ScriptedModel:
        def __init__(self, present_flags):
            self.present_flags = present_flags
            self.n = -1

        def predict(self, frame, conf, verbose):
            self.n += 1
            if self.present_flags[self.n]:
                return [FakeResult([FakeBox((100, 100, 250, 240))])]
            return [FakeResult([])]

    # present, present, then 10 misses (>> confirm_frames=4), then present again
    flags = [True, True] + [False] * 10 + [True]
    det = make_detector(ScriptedModel(flags), confirm_frames=4, poll_interval_s=999)
    det._latest_frame = FRAME

    for cycle, _ in enumerate(flags, start=1):
        det._process_frame(FRAME, cycle, heartbeat_every=999)

    box = det.get_confirmed_box()
    assert box is None, ("streak must not survive a 10-cycle gap and confirm on "
                          "stale progress from before the gap, got %r" % (box,))
    assert det._streak_count == 1, ("the single post-gap detection should have "
                                     "started a fresh streak at 1, got %d" %
                                     det._streak_count)


def check_iou_helper(_rec=None):
    assert ya._iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert ya._iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0
    assert ya._boxes_consistent((0, 0, 10, 10), (0, 0, 10, 10))
    assert not ya._boxes_consistent((0, 0, 10, 10), (100, 100, 10, 10))


def check_bad_weights_path_disables_cleanly(_rec=None):
    det = ya.AutoAcquireDetector("/nonexistent/path/does-not-exist.pt",
                                  log=lambda _m: None)
    assert det.ready is False, "a missing weights file should leave ready=False"
    # must not raise even though it was never start()ed / has no thread
    det.stop()


def main():
    checks = [
        ("iou-helper", check_iou_helper),
        ("consistent-detection-confirms", check_consistent_detection_confirms),
        ("confirmed-box-clears-on-read", check_confirmed_box_clears_on_read),
        ("oversized-box-never-confirms", check_oversized_box_never_confirms),
        ("inconsistent-detections-never-confirm", check_inconsistent_detections_never_confirm),
        ("occasional-miss-still-confirms", check_occasional_miss_still_confirms),
        ("sustained-absence-fully-resets", check_sustained_absence_fully_resets),
        ("bad-weights-path-disables-cleanly", check_bad_weights_path_disables_cleanly),
    ]
    failed = 0
    for name, fn in checks:
        print("-> %s" % name)
        try:
            fn()
            print("   PASS")
        except AssertionError as exc:
            failed += 1
            print("   FAIL: %s" % exc)
    print("\n%d/%d checks passed" % (len(checks) - failed, len(checks)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
