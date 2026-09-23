import argparse
import time

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a V4L2 camera.")
    parser.add_argument(
        "--source",
        default="/dev/video0",
        help="Camera device path or OpenCV source index",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {args.source}")

    # Request MJPEG at the target resolution/FPS.
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera: {args.source}")
    print(f"Requested: {args.width}x{args.height} @ {args.fps} FPS")
    print(f"Actual:    {actual_width}x{actual_height} @ {reported_fps:.2f} FPS")
    print("Press 'q' to quit.")

    frame_count = 0
    start_time = time.monotonic()
    fps_start = start_time

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("Failed to read frame.")
                break

            frame_count += 1

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

            cv2.imshow("Camera Test", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            # Also exit if the window was closed.
            if cv2.getWindowProperty("Camera Test", cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()