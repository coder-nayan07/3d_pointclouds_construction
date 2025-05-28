import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import os
import time

# --- Configuration ---
# Chessboard parameters
# IMPORTANT: This is the number of *interior* corners, NOT the number of squares.
# Count the corners where 4 squares meet. For an 8x8 square board, use (7, 7).
# For a board with 10x7 squares, use (9, 6).
CHESSBOARD_SIZE = (4, 3)  # Example: (width, height) = (cols, rows) of interior corners
SQUARE_SIZE = 0.03      # Size of one square in meters (e.g., 25mm = 0.025m).
                         # This defines the scale of the World coordinate system.

# RealSense camera configuration
RS_RESOLUTION_X = 640
RS_RESOLUTION_Y = 480
RS_FPS = 30

# Calibration File (where INTRINSIC MATRIX K is stored)
# MAKE SURE this is the correct path to your .npy file containing the 3x3 camera matrix
# Note: This file is expected to contain ONLY the K matrix, not distortion coefficients.
INTRINSICS_K_FILE = r"D:\codes\camera_calib_IIITA\cam_0\camera_intrinsic_matrix.npy" # <-- !!! RENAME THIS TO YOUR FILE NAME !!!

# Output file for the LAST detected extrinsic matrix
EXTRINSIC_OUTPUT_FILE = r"D:\codes\camera_calib_IIITA\cam_0\camera_extrinsic_matrix.npy" # Using .npy for single matrix save

# --- Setup RealSense Pipeline ---
try:
    print("Setting up RealSense camera...")
    context = rs.context()
    devices = context.query_devices()
    if not devices:
        print("Error: No RealSense device detected. Is it plugged in?")
        sys.exit(1)
    print(f"Found {len(devices)} RealSense device(s).")

    pipeline = rs.pipeline()
    cfg = rs.config()

    # Enable the color stream (necessary for chessboard detection)
    cfg.enable_stream(rs.stream.color, RS_RESOLUTION_X, RS_RESOLUTION_Y, rs.format.bgr8, RS_FPS)

    # Start streaming
    profile = pipeline.start(cfg)
    print(f"Streaming started with color stream at {RS_RESOLUTION_X}x{RS_RESOLUTION_Y}@{RS_FPS}fps.")

    # Get image size from the stream profile (needed for solvePnP compatibility,
    # though not directly used beyond getting intrinsics in this script)
    color_stream = profile.get_stream(rs.stream.color)
    color_profile = rs.video_stream_profile(color_stream)
    img_size = (color_profile.width(), color_profile.height())


except Exception as e:
    print(f"Critical Error starting RealSense pipeline: {e}")
    print("Please ensure the RealSense SDK is installed and the camera is connected.")
    sys.exit(1)

# --- Get Camera Intrinsics (Load K from file, get D from SDK) ---
K_color = None
D_color = None

print("\nAttempting to load intrinsic matrix K from file...")
try:
    # Load the K matrix from the .npy file
    loaded_k = np.load(INTRINSICS_K_FILE)

    # Check if the loaded data is indeed a 3x3 matrix
    if loaded_k.shape == (3, 3):
        K_color = loaded_k.astype(np.float32) # Ensure correct dtype
        print(f"Successfully loaded intrinsic matrix K from {INTRINSICS_K_FILE}")
        print("--- Loaded Camera Intrinsic Matrix (K) ---")
        print("Intrinsic Matrix (K):\n", K_color)
        print("-" * 30)

        # Now, get distortion coefficients from the SDK as they are not in the K file
        print("Retrieving distortion coefficients D from RealSense SDK...")
        try:
             color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
             color_intrinsics = color_profile.get_intrinsics()
             D_color = np.array(color_intrinsics.coeffs, dtype=np.float32)

             # Ensure D_color has 5 coefficients, pad with zeros if less, truncate if more
             if len(D_color.shape) == 2:
                 D_color = D_color.flatten()
             if len(D_color) < 5:
                 D_color = np.pad(D_color, (0, 5 - len(D_color)), 'constant')
             if len(D_color) > 5:
                 D_color = D_color[:5]

             print("Successfully retrieved distortion coefficients D from SDK.")
             print("--- Distortion Coefficients (D) ---")
             print("Distortion Coefficients (D):\n", D_color)
             print("Distortion Model (from SDK):", color_intrinsics.model)
             print("-" * 30)

        except Exception as e:
            print(f"Error retrieving distortion coefficients D from SDK: {e}")
            # D is critical for solvePnP, so exit if we can't get it
            pipeline.stop()
            sys.exit(1)

    else:
        print(f"Error: File {INTRINSICS_K_FILE} does not contain a 3x3 matrix.")
        # K is also critical, so exit
        pipeline.stop()
        sys.exit(1)


except FileNotFoundError:
    print(f"Critical Error: Intrinsic matrix K file not found at {INTRINSICS_K_FILE}.")
    print("Please ensure the path is correct and the file exists.")
    pipeline.stop()
    sys.exit(1)

except Exception as e:
    print(f"Critical Error loading intrinsic matrix K from file: {e}")
    print("Please ensure the file is a valid NumPy .npy file containing a single 3x3 array.")
    pipeline.stop()
    sys.exit(1)


# Check if both K and D were successfully obtained
if K_color is None or D_color is None:
     print("Critical Error: Could not obtain both intrinsic matrix K and distortion coefficients D.")
     pipeline.stop()
     sys.exit(1)


# --- Prepare 3D points for the chessboard corners (World Coordinate System) ---
# These points define the World coordinate system relative to the chessboard.
#
# --- IMPORTANT ---
# We are defining the World Coordinate System (WCS) such that its origin (0,0,0)
# is located at the LEFTMOST and BOTTOM-MOST interior corner of the chessboard.
# The X-axis points along the bottom row of interior corners to the right.
# The Y-axis points along the leftmost column of interior corners upwards.
# The Z-axis points outwards from the chessboard surface.
# Units are in meters, scaled by SQUARE_SIZE.
#
print("\nPreparing 3D object points for chessboard (defines World system)...")
num_corners = CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1]
objp = np.zeros((num_corners, 3), np.float32)

# Create a grid of (x, y) coordinates corresponding to the grid indices
# x_grid runs from 0 to CHESSBOARD_SIZE[0] - 1 (cols)
# y_grid runs from 0 to CHESSBOARD_SIZE[1] - 1 (rows)
# Default mgrid creates grid points like (0,0), (1,0), ..., (cols-1, 0), (0,1), ... (cols-1, rows-1)
# after transpose and reshape, these map to the order returned by findChessboardCorners
grid_coords_flat = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

# The points in objp need to correspond to the order returned by findChessboardCorners.
# findChessboardCorners typically returns points row by row, from top-left to bottom-right.
# So, the first point in 'corners' corresponds to grid_coords_flat[0] (0,0), the second to grid_coords_flat[1] (1,0), etc.
#
# We need to assign the World Coordinates (X_world, Y_world, Z_world) to objp
# such that objp[i] contains the World coordinates of the corner whose image coordinates
# are in corners[i].
#
# Original grid indices (x_grid, y_grid): origin top-left (0,0), x right, y down.
# Desired World coords (X_world, Y_world): origin bottom-left, X right, Y up.
#
# X_world = x_grid * SQUARE_SIZE
# Y_world = (CHESSBOARD_SIZE[1] - 1 - y_grid) * SQUARE_SIZE  # Invert y_grid, scale
# Z_world = 0
#
# Apply this transformation to each pair of grid indices in grid_coords_flat
objp[:, 0] = grid_coords_flat[:, 0] * SQUARE_SIZE # Scale x_grid for World X
objp[:, 1] = (CHESSBOARD_SIZE[1] - 1 - grid_coords_flat[:, 1]) * SQUARE_SIZE # Invert and scale y_grid for World Y
objp[:, 2] = 0 # Z remains 0

print(f"Generated 3D object points for {num_corners} corners.")
print("World Origin (0,0,0) is at the leftmost, bottom-most interior corner.")
print("World X-axis: Right along the bottom row.")
print("World Y-axis: Up along the leftmost column.")
print("World Z-axis: Outward from the board.")
print(f"Example points:")

# Find indices in grid_coords_flat corresponding to significant points
# We need the flat index to look up in objp
origin_grid_idx = (0, CHESSBOARD_SIZE[1]-1) # bottom-left grid index
maxX_grid_idx = (CHESSBOARD_SIZE[0]-1, CHESSBOARD_SIZE[1]-1) # bottom-right grid index
maxY_grid_idx = (0, 0) # top-left grid index
topRight_grid_idx = (CHESSBOARD_SIZE[0]-1, 0) # top-right grid index

# Find the corresponding index in the flattened grid_coords_flat
# This requires reconstructing the original grid order before flatten/reshape
# For mgrid[0:cols, 0:rows].T.reshape(-1, 2), the order is (0,0)...(cols-1,0), (0,1)...(cols-1,1), etc.
# The flat index for grid (x, y) is y * cols + x
cols = CHESSBOARD_SIZE[0]
origin_flat_idx = origin_grid_idx[1] * cols + origin_grid_idx[0]
maxX_flat_idx = maxX_grid_idx[1] * cols + maxX_grid_idx[0]
maxY_flat_idx = maxY_grid_idx[1] * cols + maxY_grid_idx[0]
topRight_flat_idx = topRight_grid_idx[1] * cols + topRight_grid_idx[0]


print(f" Origin (grid {origin_grid_idx}): {objp[origin_flat_idx]}")
print(f" X-axis end (grid {maxX_grid_idx}): {objp[maxX_flat_idx]}")
print(f" Y-axis end (grid {maxY_grid_idx}): {objp[maxY_flat_idx]}")
print(f" Top-right corner (grid {topRight_grid_idx}): {objp[topRight_flat_idx]}")


print("Units are in meters based on SQUARE_SIZE.")
print("-" * 30)


# --- Main Loop: Find Pattern and Calculate Camera-to-World Pose ---
print("\nStarting extrinsic matrix estimation loop.")
print("Hold the chessboard pattern in front of the camera.")
print("The Camera-to-World matrix is calculated in real-time when the pattern is detected.")
print(f"The matrix from the LAST detected frame will be saved to {EXTRINSIC_OUTPUT_FILE} upon exiting.")
print("Press 'q' or 'Esc' to exit.")

# Variable to store the last successfully found Camera-to-World matrix
last_camera_to_world_matrix = None

# Flags for findChessboardCorners for robustness
chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS
# print(f"Using findChessboardCorners flags: {chessboard_flags}") # Optional: print flag value

try:
    while True:
        # Wait for a coherent set of frames (color only in this script)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # If no color frame, continue
        if not color_frame:
            continue

        # Convert the color frame to a NumPy array (OpenCV format)
        color_image = np.asanyarray(color_frame.get_data())
        # Convert color image to grayscale for corner detection
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners in the grayscale image using specified flags
        # 'corners' will be a list of 2D points in image coordinates, ordered
        # typically from top-left, row by row.
        ret, corners = cv2.findChessboardCorners(gray_image, CHESSBOARD_SIZE, None, chessboard_flags)

        # Create a copy to draw on
        display_image = color_image.copy()
        info_text = "Pattern Not Found. Adjust view."
        text_color = (0, 0, 255) # Red

        # If the pattern is found
        if ret:
            # Refine the corner locations for better accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

            # Draw the detected and refined corners on the display image
            cv2.drawChessboardCorners(display_image, CHESSBOARD_SIZE, corners2, ret)

            # --- Solve for the pose (Extrinsic Parameters) ---
            # solvePnP requires object points (3D, objp), image points (2D, corners2),
            # Camera Matrix (K_color), Distortion Coefficients (D_color)
            # It returns rvec (rotation vector) and tvec (translation vector).
            # These represent the transformation FROM the World coordinate system (defined by objp)
            # TO the Camera coordinate system.
            # So, R_wc and T_wc are returned.
            # E_world_to_camera = [ R_wc | T_wc ]
            #                       [ 0    |  1   ]
            success, rvec_wc, tvec_wc = cv2.solvePnP(objp, corners2, K_color, D_color, flags=cv2.SOLVEPNP_IPPE)
            # Note: IPPE is generally good for planar objects like chessboards when K is known.
            # You could also use cv2.SOLVEPNP_DLS or cv2.SOLVEPNP_EPNP if IPPE fails,
            # but IPPE is often preferred for planar targets.

            if success:
                info_text = "Pattern & Pose Found!"
                text_color = (0, 255, 0) # Green

                # Draw coordinate axes of the World frame on the image for visualization
                # drawFrameAxes takes rvec and tvec representing World -> Camera
                axis_length = SQUARE_SIZE * 3  # Length of the drawn axes in meters
                cv2.drawFrameAxes(display_image, K_color, D_color, rvec_wc, tvec_wc, axis_length)

                # --- Calculate the Camera to World Extrinsic Matrix ---
                # We have R_wc (rotation World -> Camera) and T_wc (translation World -> Camera) from solvePnP
                # E_world_to_camera = [ R_wc | T_wc ]
                #                       [ 0    |  1   ]
                # We want E_camera_to_world = [ R_cw | T_cw ] which is the inverse
                #                               [ 0    |  1   ]
                # The inverse transformation (Camera -> World) is:
                # R_cw = R_wc.T
                # T_cw = -R_cw @ T_wc  (Position of Camera origin in World coordinates)

                # Convert rotation vector (World to Camera) to 3x3 matrix
                R_wc, _ = cv2.Rodrigues(rvec_wc)

                # Calculate rotation matrix Camera to World (transpose of R_wc)
                R_cw = R_wc.T

                # Calculate translation vector Camera to World
                # T_cw is the vector from the World origin to the Camera origin, expressed in World coordinates.
                T_cw_vec = -R_cw @ tvec_wc.flatten() # Use @ for matrix multiplication, flatten tvec_wc

                # Construct the 4x4 homogeneous Camera to World matrix
                current_camera_to_world_matrix = np.eye(4)
                current_camera_to_world_matrix[:3, :3] = R_cw             # Set the rotation part (Camera -> World)
                current_camera_to_world_matrix[:3, 3] = T_cw_vec          # Set the translation part (Camera origin in World coords)

                # --- Update the last successfully calculated matrix ---
                last_camera_to_world_matrix = current_camera_to_world_matrix

                # --- Optionally print the matrix for every frame (can be noisy) ---
                # print("\n--- Camera to World Extrinsic Matrix Found! ---")
                # print("Matrix (E_camera_to_world):\n", current_camera_to_world_matrix)
                # print("-" * 60) # Add separator if printing every frame


            else:
                 # solvePnP failed for the detected pattern
                 info_text = "Pattern Found, PnP Failed."
                 text_color = (0, 165, 255) # Orange
                 last_camera_to_world_matrix = None # Clear the matrix if PnP fails

        else:
            # Pattern not found
            last_camera_to_world_matrix = None # Clear the matrix if pattern is lost


        # Put info text on the image
        cv2.putText(display_image, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

        # Add text indicating if a matrix is available to save
        if last_camera_to_world_matrix is not None:
             cv2.putText(display_image, f"Last pose available. Exiting will save.", (10, 60),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        else:
             cv2.putText(display_image, f"No pose available to save.", (10, 60),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)


        # Display the resulting frame
        cv2.imshow('Camera View with Estimated Extrinsic Matrix (Press Q or Esc to exit)', display_image)

        # Handle user input
        key = cv2.waitKey(1) # Wait 1ms for a key press

        # Check for 'q' or 'Esc' key press to exit
        if key & 0xFF == ord('q') or key == 27:  # 27 is Esc key
            print("Exiting loop...")
            break
        # Optional: Add a short delay if pattern is not found to reduce CPU usage
        # if not ret:
        #    time.sleep(0.05) # Small delay


finally:
    # --- Cleanup ---
    print("\nStopping RealSense pipeline...")
    pipeline.stop()

    print("Closing OpenCV windows...")
    cv2.destroyAllWindows()
    print("Cleanup complete.")

# --- Save the LAST successfully estimated matrix ---
print(f"\n--- Saving Last Estimated Camera-to-World Extrinsic Matrix ---")
if last_camera_to_world_matrix is not None:
    try:
        np.save(EXTRINSIC_OUTPUT_FILE, last_camera_to_world_matrix)
        print(f"Matrix saved successfully to {EXTRINSIC_OUTPUT_FILE}")
        print("Matrix:\n", last_camera_to_world_matrix)
        print("-" * 60)
    except Exception as e:
        print(f"Error saving last extrinsic matrix to {EXTRINSIC_OUTPUT_FILE}: {e}")
        print("-" * 60)
else:
    print(f"No valid extrinsic matrix was found during the run. Nothing saved to {EXTRINSIC_OUTPUT_FILE}.")
    print("-" * 60)


# --- Final Output Summary ---
print("\nScript finished.")