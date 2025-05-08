import numpy as np
import cv2
import open3d as o3d
import os # To check if files exist

# --- Configuration ---
# Paths to your saved ALIGNED images
rgb_image_path = r"D:\codes\camera_calib_IIITA\itr2\cam_2\rgb\1.png"   # Path to your ALIGNED RGB image
depth_image_path = r"D:\codes\camera_calib_IIITA\itr2\cam_2\depth\1.png" # Path to your corresponding ALIGNED Depth image

# Path to your PRE-SAVED intrinsic matrix .npy file
# THIS FILE SHOULD CONTAIN THE 3x3 INTRINSIC MATRIX K for the COLOR CAMERA
# at the resolution of your saved images (e.g., 640x480).
INTRINSIC_MATRIX_FILE = r"D:\codes\camera_calib_IIITA\itr2\cam_2\intrinsics.npy" # <-- !!! REPLACE WITH YOUR ACTUAL FILE PATH !!!

# Depth Scale (Multiplier to convert raw depth unit to meters)
depth_scale = 0.001 # For Intel RealSense Z16 depth format, 1 unit = 0.001 meters

# Output path for the generated point cloud file (.ply format is common)
output_pcd_path = r"D:\codes\camera_calib_IIITA\itr2\cam_2"

# --- Load Camera Intrinsics from .npy file ---
print("--- Loading Camera Intrinsics ---")
if not os.path.exists(INTRINSIC_MATRIX_FILE):
    print(f"Error: Intrinsic matrix file not found at {INTRINSIC_MATRIX_FILE}")
    exit()

try:
    K = np.load(INTRINSIC_MATRIX_FILE)
    if K.shape != (3, 3):
        print(f"Error: Loaded intrinsic matrix from {INTRINSIC_MATRIX_FILE} is not 3x3. Shape is {K.shape}")
        exit()
    
    # Extract fx, fy, cx, cy from the loaded K matrix
    # These are used in calculations and print statements later
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    print(f"Successfully loaded intrinsic matrix K from {INTRINSIC_MATRIX_FILE}")
    print("Intrinsic Matrix (K):\n", K)
    print(f"Extracted parameters: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

except Exception as e:
    print(f"Error loading or processing intrinsic matrix from {INTRINSIC_MATRIX_FILE}: {e}")
    exit()


# --- Load Images ---
print("\n--- Loading Images ---")
if not os.path.exists(rgb_image_path):
    print(f"Error: RGB image not found at {rgb_image_path}")
    exit()
if not os.path.exists(depth_image_path):
    print(f"Error: Aligned Depth image not found at {depth_image_path}")
    exit()

color_image = cv2.imread(rgb_image_path)
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

if color_image is None:
    print(f"Error loading color image from {rgb_image_path}")
    exit()
if depth_image is None:
    print(f"Error loading aligned depth image from {depth_image_path}")
    exit()

height, width = color_image.shape[:2]

# Optional warning if image size doesn't perfectly match what might be expected from cx, cy
# (cx is roughly width/2, cy is roughly height/2)
# This is just a heuristic check. The user is responsible for ensuring images match the loaded intrinsics.
expected_width_from_cx = int(cx * 2)
expected_height_from_cy = int(cy * 2)

if abs(width - expected_width_from_cx) > 20 or abs(height - expected_height_from_cy) > 20 : # Allow some tolerance
    print(f"Warning: Loaded images are {width}x{height}.")
    print(f"The loaded COLOR intrinsics (cx={cx}, cy={cy}) suggest an approximate resolution of {expected_width_from_cx}x{expected_height_from_cy}.")
    print("If these resolutions are significantly different, it WILL likely result in an incorrect point cloud.")
    print("Please ensure the images were saved at the resolution matching the loaded intrinsic matrix.")


if depth_image.dtype != np.uint16:
    print(f"Warning: Aligned depth image data type is {depth_image.dtype}, expected uint16 for Z16.")
    print("This might affect depth scaling. Ensure cv2.IMREAD_UNCHANGED was used.")

print(f"Loaded images with resolution: {width}x{height}")
print(f"Using COLOR intrinsics (fx={fx}, fy={fy}, cx={cx}, cy={cy}) from loaded file: {INTRINSIC_MATRIX_FILE}")
print(f"Depth scale: {depth_scale}")


# --- Create Point Cloud ---
print("\n--- Creating Point Cloud ---")
print("Converting image to point cloud...")

u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
Z = depth_image.astype(np.float32) * depth_scale

# Calculate X and Y coordinates using the loaded COLOR camera intrinsics
X = (u_coords - cx) * Z / fx
Y = (v_coords - cy) * Z / fy

points_3d_map = np.stack([X, Y, Z], axis=-1)
valid_pixels_mask = Z > 0
points_3d = points_3d_map[valid_pixels_mask]
colors_bgr = color_image[valid_pixels_mask]
colors_rgb = colors_bgr[:, [2, 1, 0]] # BGR to RGB
colors_rgb = colors_rgb / 255.0      # Normalize to [0, 1]


# --- Create Open3D Point Cloud Object ---
print("\n--- Creating Open3D object ---")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)
pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
print(f"Generated a point cloud with {len(points_3d)} points (pixels with valid depth).")


# --- Optional: Save the Point Cloud ---
print(f"\n--- Saving Point Cloud ---")
print(f"Saving point cloud to {output_pcd_path}")
try:
    o3d.io.write_point_cloud(output_pcd_path, pcd)
    print("Point cloud saved successfully.")
except Exception as e:
    print(f"Error saving point cloud: {e}")


# --- Optional: Visualize the Point Cloud ---
print("\n--- Visualizing Point Cloud ---")
print("Visualizing the point cloud. Close the visualization window to exit.")
try:
    o3d.visualization.draw_geometries([pcd],
                                      window_name="Point Cloud from Aligned Images",
                                      width=800, height=600)
    print("Visualization window closed.")
except Exception as e:
    print(f"Error during visualization: {e}")

print("\nScript finished.")