import numpy as np
import open3d as o3d
import os
from pathlib import Path

# --------- CONFIG ---------
root_dir = Path(r"D:\codes\camera_calib_IIITA\itr2")  # 🔁 CHANGE THIS to your actual path
frame_id = "1.png"                    # image filename (e.g., "0001.png")
num_cams = 3                          # 🔁 SET this to the number of camera folders you have

depth_scale = 1000.0  # mm to meters
depth_trunc = 3.0     # maximum depth (meters)
voxel_size = 0.005    # for downsampling
# --------------------------

combined_pcd = o3d.geometry.PointCloud()

for cam_idx in range(num_cams):
    cam_dir = root_dir / f"cam_{cam_idx}"
    print(f"Processing cam_{cam_idx}...")

    # Paths to images and matrices
    color_path = cam_dir / "rgb" / frame_id
    depth_path = cam_dir / "depth" / frame_id
    intrinsic_path = cam_dir / "intrinsics.npy"
    extrinsic_path = cam_dir / "extrinsic_matrix.npy"

    # Load color and depth
    color = o3d.io.read_image(str(color_path))
    depth = o3d.io.read_image(str(depth_path))
    h, w = np.asarray(color).shape[:2]

    # Load intrinsics and extrinsics
    K = np.load(str(intrinsic_path))   # 3x3
    pose = np.load(str(extrinsic_path))  # 4x4

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    # Generate RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=depth_scale, depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )

    # Create point cloud in camera coordinates
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
    pcd_cam = np.asarray(pcd.points)

    # Apply your custom transformation (camera-to-world)
    R = pose[:3, :3]
    t = pose[:3, 3]
    trans_pcd = (R @ pcd_cam.T + t[:, None]).T

    # Recreate transformed point cloud
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(trans_pcd)
    if pcd.has_colors():
        transformed_pcd.colors = pcd.colors

    combined_pcd += transformed_pcd

# Post-processing
combined_pcd = combined_pcd.voxel_down_sample(voxel_size=voxel_size)
combined_pcd, _ = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# Visualize
o3d.visualization.draw_geometries([combined_pcd], window_name="Fused Point Cloud")
