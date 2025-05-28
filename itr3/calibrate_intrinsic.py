import cv2
import numpy as np
from pathlib import Path
import glob

CHESSBOARD_CORNERS = (9, 6)
SQUARE_SIZE_METERS = 0.0235

def calibrate_camera_intrinsics(camera_index: int):
    image_folder = Path(f"itr3/cam_{camera_index}/intrinsic")
    if not image_folder.exists():
        print(f"Intrinsic image folder not found: {image_folder}")
        return None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((CHESSBOARD_CORNERS[0] * CHESSBOARD_CORNERS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_CORNERS[0], 0:CHESSBOARD_CORNERS[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_METERS

    objpoints = []
    imgpoints = []

    image_files = sorted(image_folder.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else float('inf'))
    if not image_files:
        print(f"No PNG images found in {image_folder}")
        return None, None

    print(f"Found {len(image_files)} images in {image_folder}")
    img_shape = None

    for fname in image_files:
        img = cv2.imread(str(fname))
        if img is None:
            print(f"Warning: Could not read image {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_CORNERS, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
        else:
            print(f"Chessboard not found in {fname}")

    if not objpoints or not imgpoints:
        print(f"Could not find valid chessboards for camera {camera_index}")
        return None, None

    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    if ret:
        print(f"\nCalibration successful for cam_{camera_index}")
        print(f"Camera Matrix (K):\n{mtx}")
        print(f"Distortion Coefficients (D):\n{dist}")

        out_dir = Path(f"itr3/cam_{camera_index}")
        np.save(out_dir / "intrinsics.npy", mtx)
        np.save(out_dir / "distortion.npy", dist)
        print(f"Saved intrinsics.npy and distortion.npy to {out_dir}")
        return mtx, dist
    else:
        print(f"Calibration failed for cam_{camera_index}")
        return None, None


if __name__ == "__main__":
    try:
        cam_index = int(input("Enter camera index to calibrate (e.g., 0, 1, 2): ").strip())
    except ValueError:
        print("Invalid input. Please enter a numeric camera index.")
        exit()

    calibrate_camera_intrinsics(cam_index)
