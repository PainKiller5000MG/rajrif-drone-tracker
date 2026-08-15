# 2. Fetch the model weights

The trained weights (`drone_yolov11x.pt`, 114,382,930 bytes / 109.1 MB) are
**not stored in git** — they exceed GitHub's 100MB hard file-size limit.
Only a placeholder `weights/README.md` ships with the repo in each module
folder. You must fetch the real file yourself, once, into **both**
locations that use it.

## Source

HuggingFace repo `doguilmak/Drone-Detection-YOLOv11x`, matching exactly the
training run described in `Drone-Detection-YOLOv11x/README.md` (YOLOv11x,
1,012 training images, mAP50 0.905, single class `drone`). This has been
verified three independent ways this session: architecture inspection
(56.9M params matching the source repo's own stated 56,828,179), matching
Ultralytics' own published YOLO11x file size (109–115MB), and folder-size
breakdown confirming nothing is missing or truncated. There is no
alternate, larger, or "more complete" version of this file anywhere — see
`11-troubleshooting.md` if this comes up again.

## Linux / macOS

```
mkdir -p drone_detection_module/weights drone_tracking_module/weights
curl -L -o drone_detection_module/weights/drone_yolov11x.pt \
  "https://huggingface.co/doguilmak/Drone-Detection-YOLOv11x/resolve/main/weight/best.pt"
cp drone_detection_module/weights/drone_yolov11x.pt drone_tracking_module/weights/
```

## Windows (PowerShell)

`curl` is aliased to `Invoke-WebRequest` by default in PowerShell, which
doesn't support the same flags — use `curl.exe` explicitly, or the native
PowerShell command.

```powershell
mkdir drone_detection_module\weights, drone_tracking_module\weights -Force

curl.exe -L -o drone_detection_module\weights\drone_yolov11x.pt `
  "https://huggingface.co/doguilmak/Drone-Detection-YOLOv11x/resolve/main/weight/best.pt"

Copy-Item drone_detection_module\weights\drone_yolov11x.pt drone_tracking_module\weights\
```

Or, if `curl.exe` isn't found:
```powershell
Invoke-WebRequest -Uri "https://huggingface.co/doguilmak/Drone-Detection-YOLOv11x/resolve/main/weight/best.pt" `
  -OutFile "drone_detection_module\weights\drone_yolov11x.pt"
```

## Verify the download

```
python3 -c "
import os
p = 'drone_detection_module/weights/drone_yolov11x.pt'
size = os.path.getsize(p)
print('%d bytes (%.1f MB)' % (size, size/1024/1024))
assert size > 100_000_000, 'file looks truncated - re-download'
"
```

Expected: `114382930 bytes (109.1 MB)`.

Next: [ESP32 firmware setup](03-esp32-firmware-setup.md) if you're setting
up real hardware, or skip to [running detect_and_track.py](07-run-detect-and-track.md)
for a software-only test.
