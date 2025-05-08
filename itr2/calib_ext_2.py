import numpy as np
import cv2
import pyrealsense2 as rs

# ----------------------------------------
# Load camera intrinsics (matrix and distortion) from files
# ----------------------------------------
camera_matrix = np.load(r'D:\codes\camera_calib_IIITA\itr2\cam_1\intrinsics.npy')
dist_coeffs   = np.load(r'D:\codes\camera_calib_IIITA\itr2\cam_1\distortion.npy')

# ----------------------------------------
# Chessboard pattern and real-world scale
# ----------------------------------------
pattern_size = (4, 3)        # (columns, rows) inner corners
square_size  = 0.0335        # meters per square

# Set this to True if the camera is on the opposite side of the chessboard
is_flipped = False  # 🔁 CHANGE this to False for cam_0 (reference view), True for cam_2 (opposite view)

# Generate object points in the world frame
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
for j in range(pattern_size[1]):
    for i in range(pattern_size[0]):
        x = i * square_size
        y = (pattern_size[1] - 1 - j) * square_size
        if is_flipped:
            x = (pattern_size[0] - 1) * square_size - x
            y = (pattern_size[1] - 1) * square_size - y
        objp[j * pattern_size[0] + i] = [x, y, 0]

# Criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ----------------------------------------
# Start RealSense camera stream
# ----------------------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline_profile = pipeline.start(config)

try:
    while True:
        # Get a frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)

        if found:
            # Refine corner accuracy
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Solve PnP for pose estimation
            success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
            if success:
                # Convert rotation vector to matrix
                R_matrix, _ = cv2.Rodrigues(rvec)

                # Construct 4x4 extrinsic matrix
                extrinsic = np.eye(4, dtype=np.float64)
                extrinsic[:3, :3] = R_matrix
                extrinsic[:3, 3]  = tvec.flatten()

                print("Extrinsic (4x4) matrix:\n", extrinsic)

                # Save to .npy
                save_path = r'D:\codes\camera_calib_IIITA\itr2\cam_1\extrinsic_matrix.npy'
                np.save(save_path, extrinsic)
                print(f"Saved to {save_path}")

                # Visualize pose
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

        # Show image
        cv2.imshow('RealSense Pose Estimation', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
