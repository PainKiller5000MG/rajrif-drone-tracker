#!/usr/bin/env python3
"""
Standalone drone detection module — detection only.

Camera in, detections out, to screen and log file. No tracking across
frames, no mount/serial control, no pan/tilt output, no fire trigger.
This script is intentionally separate from, and does not import or modify,
any tracker/mount code elsewhere in this repo.

Usage:
    python detect_drone.py --weights /path/to/best.pt
    python detect_drone.py --weights /path/to/best.pt --source 0
    python detect_drone.py --weights /path/to/best.pt --source path/to/video.mp4 --no-display

If --source is omitted, you will be prompted to confirm the OpenCV
VideoCapture device index interactively (nothing is guessed).
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Standalone YOLOv11 drone detection (read-only).")
    p.add_argument("--weights", required=True, help="Path to trained YOLOv11 .pt weights file.")
    p.add_argument("--source", default=None,
                   help="Camera device index (e.g. 0, 1) or a video file path. "
                        "If omitted, you'll be prompted to confirm the device index.")
    p.add_argument("--conf-threshold", type=float, default=0.01,
                   help="Confidence threshold: gates which detections YOLO returns at all, "
                        "what gets drawn/logged, and the console alert (default: 0.01 - "
                        "effectively unfiltered; 0.0 itself is refused by Ultralytics). "
                        "WARNING: at this setting expect frequent false positives (see "
                        "instructions/11-troubleshooting.md).")
    p.add_argument("--max-box-area-frac", type=float, default=1.0,
                   help="Reject detections whose box covers more than this fraction of the "
                        "frame area (default: 1.0 - disabled). Set below 1.0 to guard "
                        "against frame-spanning false positives from noisy/low-res feeds.")
    p.add_argument("--half", dest="half", action="store_true", default=True,
                   help="Use FP16 inference (default: on). Requires CUDA.")
    p.add_argument("--no-half", dest="half", action="store_false",
                   help="Disable FP16 inference (use FP32).")
    p.add_argument("--device", default=None,
                   help="Torch device override, e.g. 'cuda:0' or 'cpu'. Auto-detected if omitted.")
    p.add_argument("--log-format", choices=["csv", "json"], default="csv",
                   help="Detection log file format (default: csv).")
    p.add_argument("--log-path", default=None,
                   help="Path to the detection log file. Default: detections_<timestamp>.<ext> "
                        "next to this script.")
    p.add_argument("--no-display", action="store_true",
                   help="Don't open a cv2.imshow window (for headless smoke tests). "
                        "Frames are still processed and logged.")
    p.add_argument("--frame-width", type=int, default=None, help="Optional capture width override.")
    p.add_argument("--frame-height", type=int, default=None, help="Optional capture height override.")
    return p.parse_args()


def confirm_device_index():
    """Interactively ask the user to confirm the VideoCapture source. Never guess."""
    print("\nNo --source was given.")
    print("OpenCV VideoCapture needs a device index (0, 1, 2, ...) for your GoPro's "
          "USB webcam / capture-card feed, or a video file path for testing.")
    print("Device indices are OS- and setup-dependent — plugging in/out other cameras "
          "changes them, so this is not something to guess at.")
    while True:
        raw = input("Enter the device index or video file path to use: ").strip()
        if raw == "":
            continue
        try:
            return int(raw)
        except ValueError:
            return raw


def check_weights_size(weights_path: str):
    """Warn (and require confirmation) if the weights look like a heavy YOLOv11 variant
    that may not fit comfortably in 6GB VRAM."""
    name = Path(weights_path).stem.lower()
    heavy_markers = ("11x", "yolo11x", "v11x", "-x", "_x")
    looks_heavy = any(m in name for m in heavy_markers)

    size_mb = None
    if os.path.exists(weights_path):
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        if size_mb > 80:
            looks_heavy = True

    if looks_heavy:
        print("\n" + "=" * 70)
        print("WARNING: These weights look like a YOLOv11x (extra-large) model")
        if size_mb is not None:
            print(f"  (file size: {size_mb:.1f} MB)")
        print("YOLOv11x is heavy for a 6GB VRAM GPU and may OOM or run very slowly")
        print("at FP16, especially at larger frame sizes.")
        print("Consider testing with YOLOv11m or YOLOv11s weights instead.")
        print("=" * 70)
        resp = input("Continue anyway with these weights? [y/N]: ").strip().lower()
        if resp != "y":
            print("Aborting. Re-run with lighter weights, or pass 'y' to proceed.")
            sys.exit(1)


def make_log_writer(log_format: str, log_path: str):
    if log_format == "csv":
        is_new = not os.path.exists(log_path)
        f = open(log_path, "a", newline="")
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp_utc", "class", "confidence", "x1", "y1", "x2", "y2"])

        def log_row(ts, cls_name, conf, box):
            writer.writerow([ts, cls_name, f"{conf:.4f}", *box])
            f.flush()

        return log_row, f

    else:  # json — one JSON object per line (JSON Lines)
        f = open(log_path, "a")

        def log_row(ts, cls_name, conf, box):
            record = {
                "timestamp_utc": ts,
                "class": cls_name,
                "confidence": round(float(conf), 4),
                "bbox": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

        return log_row, f


def main():
    args = parse_args()

    check_weights_size(args.weights)

    source = args.source
    if source is None:
        source = confirm_device_index()
    else:
        # allow numeric strings passed via --source to behave like a device index
        try:
            source = int(source)
        except ValueError:
            pass  # treat as a file path / URL

    if args.log_path is None:
        ext = "csv" if args.log_format == "csv" else "jsonl"
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(Path(__file__).parent / f"detections_{ts_tag}.{ext}")
    else:
        log_path = args.log_path

    print(f"Log file: {log_path}")

    # --- Load model ---
    from ultralytics import YOLO
    import torch
    import cv2
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    use_half = args.half and device.startswith("cuda")
    if args.half and not device.startswith("cuda"):
        print("NOTE: --half requested but no CUDA device available; running FP32 on CPU instead.")

    print(f"Loading weights: {args.weights}")
    print(f"Device: {device}  |  half precision: {use_half}")
    model = YOLO(args.weights)
    class_names = model.names  # {class_id: class_name}

    # --- Open capture ---
    cap = cv2.VideoCapture(source)
    if args.frame_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    if args.frame_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)

    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {source!r}")
        sys.exit(1)

    print(f"Video source opened: {source!r}")
    print(f"Confidence alert threshold: {args.conf_threshold}")
    print("Press 'q' in the video window to quit." if not args.no_display
          else "Running headless (--no-display). Press Ctrl+C to quit.")

    log_row, log_file = make_log_writer(args.log_format, log_path)

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream / failed to read frame.")
                break
            frame_count += 1

            results = model.predict(frame, conf=args.conf_threshold, half=use_half,
                                     device=device, verbose=False)
            result = results[0]

            frame_h, frame_w = frame.shape[:2]
            frame_area = frame_w * frame_h

            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = class_names.get(cls_id, str(cls_id))

                box_area_frac = ((x2 - x1) * (y2 - y1)) / frame_area if frame_area else 0
                if box_area_frac > args.max_box_area_frac:
                    continue

                ts = datetime.now(timezone.utc).isoformat()
                log_row(ts, cls_name, conf, (x1, y1, x2, y2))

                if not args.no_display:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if conf >= args.conf_threshold:
                    print(f"Drone detected, confidence {conf:.2f}")

            if not args.no_display:
                cv2.imshow("Drone Detection (read-only)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quit requested.")
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        log_file.close()
        print(f"Processed {frame_count} frames. Detections logged to: {log_path}")


if __name__ == "__main__":
    main()
