from threading import Event, Thread

from src.capture import CameraCapture, CameraConfig
from src.frame_buffer import LatestFrameBuffer


class CameraWorker:
    """Continuously captures frames in a background thread."""

    def __init__(
        self,
        config: CameraConfig,
        buffer: LatestFrameBuffer,
    ) -> None:
        self.config = config
        self.buffer = buffer

        self.camera = CameraCapture(config)
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Camera worker is already running.")

        # Camera initialization happens on the calling thread.
        self.camera.open()

        self._thread = Thread(
            target=self._run,
            name="camera-capture",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                packet = self.camera.read()

                if packet is None:
                    continue

                self.buffer.put(packet)

        except Exception as exc:
            self._error = exc

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self.camera.release()

    @property
    def error(self) -> Exception | None:
        return self._error