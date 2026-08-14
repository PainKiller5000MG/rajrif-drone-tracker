"""Quick camera device enumerator - lists every index that actually opens
and what resolution/backend it reports, so you don't have to guess the
--source index for detect_and_track.py / detect_drone.py.

    python probe_cameras.py
"""
import os

# Without this, opening a camera through the MSMF backend can take ~40s per
# device (known issue, see 20RajRif/CLAUDE.md gotchas) - set before cv2
# builds any VideoCapture.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2

print("Probing camera indices 0-9 ...\n")
found_any = False
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            backend = cap.getBackendName()
            print("Index %d: OPENS, frame %dx%d, backend=%s" % (i, w, h, backend))
            found_any = True
        else:
            print("Index %d: opens but returns no frames (skip)" % i)
        cap.release()
    else:
        print("Index %d: does not open" % i)

if not found_any:
    print("\nNo working camera indices found at all. This means Windows/OpenCV "
          "cannot see any camera, not just the GoPro - check the Windows Camera "
          "app first, and confirm the GoPro is in USB Webcam mode.")
else:
    print("\nTry the --source value(s) above that opened successfully. The "
          "GoPro is usually the one with an unusual resolution/aspect ratio "
          "compared to a laptop's built-in webcam - unplug the GoPro and "
          "re-run this script to see which index disappears, to be certain.")
