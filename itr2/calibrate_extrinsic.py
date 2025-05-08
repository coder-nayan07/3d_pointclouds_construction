import numpy as np
import cv2
import pyrealsense2 as rs
from pathlib import Path

# ----------------------------------------
# Configuration
# ----------------------------------------
cam_dir = Path(r'D:\codes\camera_calib_IIITA\itr2\cam_2')
camera_matrix = np.load(cam_dir / 'intrinsics.npy')
extrinsic_output_path = cam_dir / 'extrinsic_matrix.npy'

pattern_size = (4, 3)          # Number of inner corners (columns, rows)
square_size  = 0.0335          # Square size in meters
axis_length  = 0.05            # Axis length in meters for visualization

# Generate 3D points of the chessboard corners
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
for j in range(pattern_size[1]):
    for i in range(pattern_size[0]):
        objp[j*pattern_size[0] + i] = [i * square_size, (pattern_size[1]-1 - j) * square_size, 0]

# Subpixel corner refinement criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ----------------------------------------
# RealSense camera setup
# ----------------------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("[INFO] Press 's' to save the extrinsic matrix.")
print("[INFO] Press 'q' to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, pattern_size,
                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            success, rvec, tvec = cv2.solvePnP(objp, corners_refined, camera_matrix, None)

            if success:
                R, _ = cv2.Rodrigues(rvec)
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = R
                extrinsic[:3, 3] = tvec.flatten()

                # Print the extrinsic matrix
                print("\n[EXTRINSIC MATRIX]")
                print(extrinsic)

                # Show axes
                cv2.drawFrameAxes(color_image, camera_matrix, None, rvec, tvec, axis_length)
                cv2.putText(color_image, "Chessboard detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    np.save(str(extrinsic_output_path), extrinsic)
                    print(f"[SAVED] Extrinsic matrix saved to {extrinsic_output_path}")
                elif key == ord('q'):
                    print("[QUIT] Exiting...")
                    break
        else:
            cv2.putText(color_image, "Chessboard NOT detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show image
        cv2.imshow('RealSense Pose Estimation', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
