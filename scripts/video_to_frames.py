
import cv2
import os
import argparse

def video_to_frames(video_path, output_dir, frame_interval=1):
    """
    Extract frames from a video and save them as images.

    Args:
        video_path: Path to input video.
        output_dir: Directory to save extracted frames.
        frame_interval: Extract every Nth frame.
    """

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(
                output_dir,
                f"{saved_count:06d}.jpg"
            )

            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"Extracted {saved_count} frames")
    print(f"Frames saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from a video"
    )

    parser.add_argument("video", help="Input video file")
    parser.add_argument(
        "-o", "--output",
        default="frames",
        help="Output directory (default: frames)"
    )
    parser.add_argument(
        "-i", "--interval",
        type=int,
        default=1,
        help="Extract every Nth frame (default: 1)"
    )

    args = parser.parse_args()

    if args.interval < 1:
        parser.error("Interval must be >= 1")

    video_to_frames(
        args.video,
        args.output,
        args.interval
    )
