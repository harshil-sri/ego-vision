from src.capture import FramePacket
from src.frame_buffer import LatestFrameBuffer


def main() -> None:
    buffer = LatestFrameBuffer()

    buffer.put(
        FramePacket(
            frame_id=1,
            timestamp_ns=100,
            image="frame-1",
        )
    )

    buffer.put(
        FramePacket(
            frame_id=2,
            timestamp_ns=200,
            image="frame-2",
        )
    )

    packet = buffer.get()

    print(f"Latest frame: {packet.frame_id}")
    print(f"Dropped frames: {buffer.dropped_frames}")


if __name__ == "__main__":
    main()