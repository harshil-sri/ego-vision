from threading import Condition

from src.capture import FramePacket


class LatestFrameBuffer:
    """Thread-safe buffer that always keeps the newest frame."""

    def __init__(self) -> None:
        self._frame: FramePacket | None = None
        self._condition = Condition()
        self._dropped_frames = 0

    def put(self, packet: FramePacket) -> None:
        with self._condition:
            if self._frame is not None:
                self._dropped_frames += 1

            self._frame = packet
            self._condition.notify()

    def get(self, timeout: float | None = None) -> FramePacket | None:
        with self._condition:
            if self._frame is None:
                self._condition.wait(timeout)

            packet = self._frame
            self._frame = None

            return packet

    @property
    def dropped_frames(self) -> int:
        with self._condition:
            return self._dropped_frames