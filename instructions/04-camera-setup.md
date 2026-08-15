# 4. Camera setup

Every script in this repo either prompts you to confirm a camera device
index or takes one via `--source`/a panel field — none of them guess.
Device indices are OS- and setup-dependent, and they shift whenever you
plug/unplug other cameras, so don't assume last time's number is still
right.

## Find the correct index

Use `drone_tracking_module/probe_cameras.py` — it lists every index that
actually opens, along with its resolution and backend:

```
cd drone_tracking_module
python probe_cameras.py
```

It sets `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` before opening
anything, avoiding a known ~40s-per-device hang on Windows (see
`11-troubleshooting.md`).

### If multiple indices open with the same resolution

This happens — a single physical camera can be enumerated under more than
one Windows backend (MSMF and DSHOW) at different indices. To find out
which index is actually your GoPro/capture card:

1. **Unplug it.**
2. Run `python probe_cameras.py` — note which indices show "OPENS."
3. **Plug it back in**, confirm it's powered on and in the correct mode
   (see below).
4. Run `probe_cameras.py` again — whichever index newly appears is the
   real one. If it's a webcam-mode device that maps to two indices at
   once (two backends), both will appear/disappear together.

### `[ERROR] obsensor_uvc_stream_channel... Camera index out of range`

This is harmless noise — OpenCV silently probing other camera backends
(like Orbbec depth-camera drivers) that have no device attached. Not a
bug, not something to fix.

## GoPro-specific setup

- Must be switched into **USB Webcam mode**, not just plugged in normally.
  Exact steps vary by model (most Hero 9–12: hold the Mode button while
  powering on, or select it from the GoPro's own settings menu).
- Some older models need **GoPro's own "GoPro Webcam" desktop utility**
  installed before Windows will see them as a camera at all.
- The USB-C cable must be **data-capable**, not charge-only.
- **Confirm in the Windows Camera app first** — if it doesn't show up
  there, OpenCV never will either, regardless of index. This isolates
  whether the problem is a camera/driver issue versus a script issue.

## Using the confirmed index

```
python detect_and_track.py --weights weights\drone_yolov11x.pt --source 1
```
or leave `--source` off and answer the interactive prompt.

Next: [Running tracker_gui.py](05-run-tracker-gui.md).
