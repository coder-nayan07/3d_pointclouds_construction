import numpy as np
import cv2
import open3d as o3d
import os # To check if files exist

# --- Configuration ---
# Paths to your saved ALIGNED images
# IMPORTANT: These should point to the images saved by the previous script WITH ALIGNMENT.
rgb_image_path = r"D:\codes\camera_calib_IIITA\rgb_aligned_cam21.png"   # Path to your ALIGNED RGB image
depth_image_path = r"D:\codes\camera_calib_IIITA\depth_aligned_cam21.png" # Path to your corresponding ALIGNED Depth image

# Camera Intrinsics for the COLOR CAMERA at the resolution of your saved images (640x480).
# YOU MUST USE THE COLOR CAMERA INTRINSICS HERE because the depth was aligned to the color frame.
# These values are taken from the "Color Camera Intrinsics" section of your SDK output:
# fx, fy: Focal lengths in pixels
# cx, cy: Principal point (optical center) in pixels
fx = 382.451 # REPLACE with YOUR Color fx
fy = 382.519# REPLACE with YOUR Color fy
cx = 320.103 # REPLACE with YOUR Color cx
cy = 241.556 # REPLACE with YOUR Color cy

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

np.save("camera_intrinsic_matrix_am_basics.npy", K)


# Depth Scale (Multiplier to convert raw depth unit to meters)
# For Intel RealSense Z16 depth format, this is typically 0.001 (mm to meters).
depth_scale = 0.001 # 1 unit in depth image = 0.001 meters

# Output path for the generated point cloud file (.ply format is common)
output_pcd_path = "output_pointcloud_ALIGNED.ply" # Indicate it's from aligned data

# --- Load Images ---
print("--- Loading Images ---")
if not os.path.exists(rgb_image_path):
    print(f"Error: RGB image not found at {rgb_image_path}")
    exit()
if not os.path.exists(depth_image_path):
    print(f"Error: Aligned Depth image not found at {depth_image_path}")
    exit()

# Read color image (OpenCV reads as BGR)
color_image = cv2.imread(rgb_image_path)
# Read aligned depth image (ensure it's read as is, without scaling, typically uint16 for Z16)
# Use cv2.IMREAD_UNCHANGED to load 16-bit depth correctly
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

if color_image is None:
    print(f"Error loading color image from {rgb_image_path}")
    exit()
if depth_image is None:
    print(f"Error loading aligned depth image from {depth_image_path}")
    exit()

# Ensure images have the same dimensions (aligned depth should match color)
height, width = color_image.shape[:2]

# Optional warning if image size doesn't match the resolution hardcoded in intrinsics
expected_width_from_intrinsics = 640 # Based on the color intrinsics provided
expected_height_from_intrinsics = 480 # Based on the color intrinsics provided

if width != expected_width_from_intrinsics or height != expected_height_from_intrinsics:
     print(f"Warning: Loaded images are {width}x{height}.")
     print(f"The provided COLOR intrinsics (fx, fy, cx, cy) appear to be for {expected_width_from_intrinsics}x{expected_height_from_intrinsics}.")
     print("This mismatch WILL likely result in an incorrect point cloud.")
     print("Please ensure the images were saved at the resolution matching the intrinsics.")


# Check if depth image is the expected type (optional but good practice)
if depth_image.dtype != np.uint16:
     print(f"Warning: Aligned depth image data type is {depth_image.dtype}, expected uint16 for Z16.")
     print("This might affect depth scaling. Ensure cv2.IMREAD_UNCHANGED was used when reading the depth image.")


print(f"Loaded images with resolution: {width}x{height}")
print(f"Using COLOR intrinsics (due to alignment): fx={fx}, fy={fy}, cx={cx}, cy={cy}")
print(f"Depth scale: {depth_scale}")


# --- Create Point Cloud ---
print("\n--- Creating Point Cloud ---")
print("Converting image to point cloud...")

# Create a meshgrid of pixel coordinates (u, v)
# u corresponds to columns (x-axis in image), v corresponds to rows (y-axis in image)
u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))

# Convert depth image to float and apply depth scale
# Z is the depth value in meters
Z = depth_image.astype(np.float32) * depth_scale

# Calculate X and Y coordinates for all pixels simultaneously using the pinhole camera model
# Using COLOR camera intrinsics here because the depth is aligned to the color frame's view.
X = (u_coords - cx) * Z / fx
Y = (v_coords - cy) * Z / fy

# Stack X, Y, Z coordinates into a single array of shape (height, width, 3)
points_3d_map = np.stack([X, Y, Z], axis=-1)

# Create a mask for valid depth values (where Z > 0). Depth of 0 typically means no reading.
valid_pixels_mask = Z > 0

# Extract only the valid 3D points using the mask
points_3d = points_3d_map[valid_pixels_mask]

# Extract the corresponding colors for the valid pixels from the color image
colors_bgr = color_image[valid_pixels_mask]

# Convert colors from BGR (OpenCV default) to RGB (Open3D expects RGB)
colors_rgb = colors_bgr[:, [2, 1, 0]]

# Normalize colors to the range [0, 1] (Open3D requires this for colors)
colors_rgb = colors_rgb / 255.0


# --- Create Open3D Point Cloud Object ---
print("\n--- Creating Open3D object ---")
pcd = o3d.geometry.PointCloud()

# Assign the calculated 3D points and colors to the Open3D point cloud object
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
print("Interaction:")
print("  - Mouse left button + move: Rotate")
print("  - Mouse wheel: Zoom")
print("  - Mouse right button + move: Translate")
print("  - 'h': Print help message")
print("  - 'q' or Escape: Close the window")

try:
    o3d.visualization.draw_geometries([pcd])
    print("Visualization window closed.")
except Exception as e:
    print(f"Error during visualization: {e}")


print("\nScript finished.")