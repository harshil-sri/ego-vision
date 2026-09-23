from dataclasses import dataclass
from typing import Union

import cv2


CameraSource = Union[int, str]


@dataclass
class CameraConfig:
    source: CameraSource = "/dev/video0"
    width: int = 640
    height: int = 480
    fps: int = 30
    pixel_format: str = "MJPG"


@dataclass
class FramePacket:
    frame_id: int
    timestamp_ns: int
    image: object


class CameraCapture:
    """Hardware-facing camera abstraction."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.cap: cv2.VideoCapture | None = None

        self.actual_width: int | None = None
        self.actual_height: int | None = None
        self.reported_fps: float | None = None

        self._frame_id = 0

    def open(self) -> None:
        source = self.config.source

        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera: {self.config.source}")

        fourcc = cv2.VideoWriter_fourcc(*self.config.pixel_format)

        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.reported_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self._frame_id = 0

    def read(self) -> FramePacket | None:
        if self.cap is None:
            raise RuntimeError("Camera is not open.")

        ok, frame = self.cap.read()

        if not ok:
            return None

        self._frame_id += 1

        return FramePacket(
            frame_id=self._frame_id,
            timestamp_ns=__import__("time").monotonic_ns(),
            image=frame,
        )

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()