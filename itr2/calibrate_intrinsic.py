import pyrealsense2 as rs
import cv2
import numpy as np
import os

# Chessboard parameters
pattern_size = (4, 3)
square_size = 0.0335  # in meters

# Prepare object points
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

objpoints = []
imgpoints = []

# Output directories
output_dir = r"D:\codes\camera_calib_IIITA\itr2\cam_0"
image_save_dir = os.path.join(output_dir, "calibration_images")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(image_save_dir, exist_ok=True)

# Start RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Disable auto exposure and white balance
sensor = profile.get_device().query_sensors()[1]  # 0 = depth, 1 = color
sensor.set_option(rs.option.enable_auto_exposure, 0)
sensor.set_option(rs.option.enable_auto_white_balance, 0)
sensor.set_option(rs.option.exposure, 200)  # Adjust as needed
sensor.set_option(rs.option.white_balance, 4600)  # Adjust as needed

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
print("Press SPACE to capture corners, 'q' to quit and calibrate.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if found:
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(color_image, pattern_size, corners_subpix, found)
        else:
            corners_subpix = None

        cv2.imshow('RealSense Calibration', color_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if found:
                objpoints.append(objp.copy())
                imgpoints.append(corners_subpix)
                img_filename = os.path.join(image_save_dir, f"capture_{len(objpoints)}.png")
                cv2.imwrite(img_filename, color_image)
                print(f"Captured and saved corners set #{len(objpoints)} -> {img_filename}")
            else:
                print("Chessboard not detected.")
        elif key == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# Calibration
if len(objpoints) > 0:
    image_size = gray.shape[::-1]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print("\n=== Calibration Complete ===")
    print("RMS Error:", ret)
    print("Intrinsic Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs.ravel())

    # Save calibration results
    np.save(os.path.join(output_dir, "intrinsics.npy"), camera_matrix)
    np.save(os.path.join(output_dir, "distortion.npy"), dist_coeffs)
    np.save(os.path.join(output_dir, "image_size.npy"), np.array(image_size))
    print(f"Saved calibration data to: {output_dir}")
else:
    print("Not enough data for calibration.")
