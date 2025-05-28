import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path
import time

# Constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAMERATE = 30

def setup_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FRAMERATE)
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FRAMERATE)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Failed to start RealSense camera: {e}")
        return None, None, None

    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    return pipeline, align, depth_scale

def create_directories(base_dir: Path, mode: str):
    base_dir.mkdir(parents=True, exist_ok=True)
    if mode == "general":
        (base_dir / "rgb").mkdir(exist_ok=True)
        (base_dir / "depth").mkdir(exist_ok=True)
    elif mode == "intrinsic":
        (base_dir / "intrinsic").mkdir(exist_ok=True)
    elif mode == "extrinsic":
        (base_dir / "extrinsic").mkdir(exist_ok=True)

def capture_frames(cam_index: int, mode: str, num_frames: int):
    pipeline, align, depth_scale = setup_camera()
    if not pipeline:
        return

    base_dir = Path("itr3") / f"cam_{cam_index}"
    create_directories(base_dir, mode)

    try:
        print("Warming up camera... please hold still.")
        time.sleep(2)  # Wait for 2 seconds before capturing

        for i in range(1, num_frames + 1):
            print(f"Capturing frame {i}/{num_frames}...")
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                print("Failed to retrieve frames.")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

            if mode == "general":
                cv2.imwrite(str(base_dir / "rgb" / f"{i}.png"), color_image)
                np.save(str(base_dir / "depth" / f"{i}.npy"), depth_image)
            elif mode == "intrinsic":
                cv2.imwrite(str(base_dir / "intrinsic" / f"{i}.png"), color_image)
            elif mode == "extrinsic":
                filename = str(base_dir / "extrinsic" / f"{i}.png")
                cv2.imwrite(filename, color_image)
                print(f"Saved extrinsic image: {filename}")
                break  # Only 1 image needed for extrinsic mode

            time.sleep(2)

    finally:
        pipeline.stop()
        print("Camera stopped.")

if __name__ == "__main__":
    cam_index = int(input("Enter camera number (e.g., 0, 1, 2): "))
    mode = input("Enter mode (general, intrinsic, extrinsic): ").strip().lower()
    if mode not in ["general", "intrinsic", "extrinsic"]:
        print("Invalid mode.")
        exit()

    num_frames = 1 if mode == "extrinsic" else int(input("Enter number of frames to capture: "))

    print(f"\nStarting capture for cam_{cam_index} in mode '{mode}' for {num_frames} frame(s)...")
    capture_frames(cam_index, mode, num_frames)
    print("Done.")
