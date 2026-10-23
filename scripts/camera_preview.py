import argparse
import time

import cv2

from src.camera_worker import CameraWorker
from src.capture import CameraConfig
from src.frame_buffer import LatestFrameBuffer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview a camera stream."
    )

    parser.add_argument("--source", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()

    config = CameraConfig(
        source=args.source,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    buffer = LatestFrameBuffer()
    worker = CameraWorker(config, buffer)

    worker.start()

    try:
        while True:
            if worker.error is not None:
                raise worker.error

            packet = buffer.get(timeout=1.0)

            if packet is None:
                continue

            frame = packet.image

            cv2.putText(
                frame,
                f"Frame {packet.frame_id}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Ego Vision - Camera Preview", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            if cv2.getWindowProperty(
                "Ego Vision - Camera Preview",
                cv2.WND_PROP_VISIBLE,
            ) < 1:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        worker.stop()
        cv2.destroyAllWindows()

        print(f"Dropped frames: {buffer.dropped_frames}")


if __name__ == "__main__":
    main()