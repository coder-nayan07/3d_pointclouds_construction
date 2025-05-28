import numpy as np
import cv2
import open3d as o3d
from pathlib import Path

def create_point_cloud_from_depth_and_color(depth_image_meters, color_image_bgr, intrinsics_matrix_K, distortion_coeffs_D=None):
    fx = intrinsics_matrix_K[0, 0]
    fy = intrinsics_matrix_K[1, 1]
    cx = intrinsics_matrix_K[0, 2]
    cy = intrinsics_matrix_K[1, 2]

    height, width = depth_image_meters.shape
    
    u_coords = np.arange(width)
    v_coords = np.arange(height)
    uu, vv = np.meshgrid(u_coords, v_coords)

    valid_depth_mask = depth_image_meters > 0.001
    
    Z = depth_image_meters[valid_depth_mask]
    u_valid = uu[valid_depth_mask]
    v_valid = vv[valid_depth_mask]

    X = (u_valid - cx) * Z / fx
    Y = (v_valid - cy) * Z / fy
    
    points_cam_np = np.vstack((X, Y, Z)).T

    if color_image_bgr is not None and len(points_cam_np) > 0:
        color_vals_bgr = color_image_bgr[v_valid, u_valid]
        colors_cam_np = color_vals_bgr[:, ::-1] / 255.0
    else:
        colors_cam_np = np.array([])

    pcd_cam = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(points_cam_np)
    if colors_cam_np.size > 0 and len(colors_cam_np) == len(points_cam_np):
        pcd_cam.colors = o3d.utility.Vector3dVector(colors_cam_np)
    
    return pcd_cam

if __name__ == "__main__":
    cam_index = input("Enter camera index (i) to generate point cloud for itr3/cam_i/: ")
    if not cam_index.isdigit():
        print("Invalid index. Please enter a numeric value.")
        exit()

    cam_dir = Path(f"itr3/cam_{cam_index}")
    if not cam_dir.is_dir():
        print(f"Directory not found: {cam_dir}")
        exit()

    rgb_dir = cam_dir / "rgb"
    depth_dir = cam_dir / "depth"

    rgb_files = sorted(list(rgb_dir.glob("*.png")))
    depth_files = sorted(list(depth_dir.glob("*.npy")))

    if not rgb_files or not depth_files:
        print(f"Missing RGB or Depth files in {rgb_dir} or {depth_dir}")
        exit()

    rgb_file = rgb_files[0]
    rgb_prefix = rgb_file.name.split('_')[0]
    
    matching_depth_file = next((df for df in depth_files if df.name.startswith(rgb_prefix)), None)
    if not matching_depth_file:
        print(f"No matching depth file for {rgb_file.name}. Using first depth file.")
        matching_depth_file = depth_files[0]
    depth_file = matching_depth_file

    intrinsics_file = cam_dir / "intrinsics.npy"
    distortion_file = cam_dir / "distortion.npy"
    output_pcd_file = cam_dir / f"pointcloud_local_cam_{cam_index}.ply"

    if not all([f.exists() for f in [rgb_file, depth_file, intrinsics_file]]):
        print("Missing required file(s):")
        for f in [rgb_file, depth_file, intrinsics_file]:
            if not f.exists():
                print(f"  Missing: {f}")
        exit()

    print(f"Loading RGB image: {rgb_file}")
    color_img = cv2.imread(str(rgb_file))
    print(f"Loading depth image: {depth_file}")
    depth_img_m = np.load(str(depth_file))
    print(f"Loading intrinsics: {intrinsics_file}")
    K = np.load(str(intrinsics_file))
    D = np.load(str(distortion_file)) if distortion_file.exists() else None

    print("Generating point cloud...")
    pcd = create_point_cloud_from_depth_and_color(depth_img_m, color_img, K, D)

    if len(pcd.points) == 0:
        print("Generated point cloud is empty.")
    else:
        print(f"Point cloud generated with {len(pcd.points)} points.")
        o3d.io.write_point_cloud(str(output_pcd_file), pcd)
        print(f"Saved point cloud to {output_pcd_file}")
        o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud - cam_{cam_index}")
