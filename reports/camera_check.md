# New external webcam check (WP0)

Date: 2026-10-23, 22:21 IST. Commit: b8006ab.

Device: "USB 2.0 Camera", VID:PID 0c49:636f, attached via usbipd (busid 4-10).
Appears as /dev/video0 (stream) + /dev/video1 (metadata), same as before.

Advertised formats (v4l2-ctl --list-formats-ext): MJPG only (compressed),
up to 4208x3120 stills and 3840x2160@30 / 2688x1512@30 / 2560x1440@30 /
1080p/720p/640x480 at 5–30 fps. This is a large upgrade over the old
integrated camera (which capped at 640x480 and timed out at 720p).

Measured through the existing capture layer (`src.capture.CameraCapture`,
MJPG, 30 fps requested, ~8 s per resolution, frames counted between
timestamps):

| Requested | Actual | Frames | FPS |
|---|---|---:|---:|
| 640x480 | 640x480 | 156 | **19.4** |
| 1280x720 | 1280x720 | 156 | **19.4** |

Conclusions:
- 720p MJPEG now works under WSL2 (old camera: timed out).
- ~19.4 FPS sustained (old camera: 15–18). Requesting 30 still yields ~19.4;
  budget the live pipeline for **~15 FPS worst case**.
- Config change vs old camera: none needed — 640x480 MJPG still negotiates.
  The app can stay at 640x480 for the demo; 720p is available if wanted.
- Not yet tested: 1080p+ under load, long-run stability, head-mount wobble.
  Camera was released after the test.
