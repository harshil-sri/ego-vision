import argparse
import time

import cv2

from src.capture import CameraCapture, CameraConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a camera stream.")

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

    try:
        with CameraCapture(config) as camera:
            print(f"Camera: {config.source}")
            print(
                f"Requested: "
                f"{config.width}x{config.height} @ {config.fps} FPS"
            )
            print(
                f"Negotiated: "
                f"{camera.actual_width}x{camera.actual_height} "
                f"@ {camera.reported_fps:.2f} FPS"
            )
            print("Press 'q' or ESC to quit.")

            frame_count = 0
            fps_start = time.monotonic()

            while True:
                packet = camera.read()

                if packet is None:
                    print("Failed to read frame.")
                    break

                frame = packet.image
                frame_count += 1

                cv2.putText(
                    frame,
                    f"Frame {packet.frame_id}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                now = time.monotonic()
                elapsed = now - fps_start

                if elapsed >= 1.0:
                    measured_fps = frame_count / elapsed

                    print(
                        f"Measured FPS: {measured_fps:.2f} | "
                        f"Frame: {frame.shape[1]}x{frame.shape[0]}"
                    )

                    frame_count = 0
                    fps_start = now

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
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
