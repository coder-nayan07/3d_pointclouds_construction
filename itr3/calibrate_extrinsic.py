import cv2
import numpy as np
from pathlib import Path

CHESSBOARD_CORNERS = (9, 6)
SQUARE_SIZE_METERS = 0.0235

def get_camera_pose_relative_to_chessboard(cam_path):
    image_path = cam_path / "extrinsic" / "1.png"
    intrinsics_path = cam_path /  "intrinsics.npy"
    distortion_path = cam_path / "distortion.npy"

    if not all([f.exists() for f in [image_path, intrinsics_path, distortion_path]]):
        print(f"Missing required files for extrinsic calibration of {cam_path.name}:")
        if not image_path.exists(): print(f"  - {image_path} (Run capture_frames.py mode 3)")
        if not intrinsics_path.exists(): print(f"  - {intrinsics_path} (Run calibrate_intrinsics.py)")
        if not distortion_path.exists(): print(f"  - {distortion_path} (Run calibrate_intrinsics.py)")
        return None

    intrinsics_K = np.load(intrinsics_path)
    distortion_D = np.load(distortion_path)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((CHESSBOARD_CORNERS[0] * CHESSBOARD_CORNERS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_CORNERS[0], 0:CHESSBOARD_CORNERS[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_METERS

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load calibration image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_CORNERS, None)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners2, intrinsics_K, distortion_D)

        if ret_pnp:
            R_cam_from_world, _ = cv2.Rodrigues(rvec)
            R_world_from_cam = R_cam_from_world.T
            t_world_from_cam = -R_world_from_cam @ tvec

            T_world_from_camera = np.eye(4)
            T_world_from_camera[0:3, 0:3] = R_world_from_cam
            T_world_from_camera[0:3, 3] = t_world_from_cam.flatten()

            output_path = cam_path / "T_world_from_camera.npy"
            np.save(str(output_path), T_world_from_camera)
            print(f"Extrinsic calibration successful for {cam_path.name}")
            print(f"  Saved extrinsic matrix to {output_path}")
            return T_world_from_camera
        else:
            print(f"solvePnP failed for {cam_path.name}")
            return None
    else:
        print(f"Chessboard not found in image for {cam_path.name}")
        return None

if __name__ == "__main__":
    print("Ensure CHESSBOARD_CORNERS and SQUARE_SIZE_METERS are correct.")
    print("Looking for camera directories inside 'itr3/'...\n")

    root_dir = Path("itr3")
    cam_dirs = [d for d in root_dir.glob("cam_*") if d.is_dir()]

    if not cam_dirs:
        print("No camera directories (cam_*) found inside 'itr3/'. Exiting.")
        exit()

    print("Found camera directories:")
    for i, cam in enumerate(cam_dirs):
        print(f"  {i}: {cam.name}")

    choice = input("Enter index of camera to calibrate (e.g., 0), or 'all' to calibrate all: ").strip()

    selected_dirs = []
    if choice.lower() == 'all':
        selected_dirs = cam_dirs
    else:
        try:
            indices = [int(i.strip()) for i in choice.split(',')]
            for idx in indices:
                if 0 <= idx < len(cam_dirs):
                    selected_dirs.append(cam_dirs[idx])
                else:
                    print(f"Warning: Index {idx} out of range.")
        except ValueError:
            print("Invalid input. Please enter valid index numbers.")
            exit()

    for cam_path in selected_dirs:
        print(f"\n--- Calibrating Extrinsics for {cam_path.name} ---")
        get_camera_pose_relative_to_chessboard(cam_path)
