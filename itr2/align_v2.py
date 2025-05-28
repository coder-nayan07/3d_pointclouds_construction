import numpy as np
import open3d as o3d
import os
from pathlib import Path

# --------- CONFIG ---------
root_dir = Path(r"D:\codes\camera_calib_IIITA\itr2")
frame_id = "1.png"
num_cams = 2

depth_scale = 1000.0  # mm to meters
depth_trunc = 3.0     # max depth
voxel_size = 0.005    # downsample

trans_step = 0.01  # meters
angle_step = np.deg2rad(1)  # radians
# --------------------------

# Load initial data
pcds = []
extrinsics = []
intrinsics = []
geometries = []
rgbd_list = []

for cam_idx in range(num_cams):
    # if cam_idx == 1:
    #     cam_idx += 1 

    cam_dir = root_dir / f"cam_{cam_idx}"
    color_path = cam_dir / "rgb" / frame_id
    depth_path = cam_dir / "depth" / frame_id
    intrinsic_path = cam_dir / "intrinsics.npy"
    extrinsic_path = cam_dir / "extrinsic_matrix.npy"

    color = o3d.io.read_image(str(color_path))
    depth = o3d.io.read_image(str(depth_path))
    h, w = np.asarray(color).shape[:2]

    K = np.load(str(intrinsic_path))  # 3x3
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    extr = np.load(str(extrinsic_path))  # 4x4

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )
    rgbd_list.append(rgbd)
    intrinsics.append(intr)
    extrinsics.append(extr)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
    pcd.transform(extr)
    pcds.append(pcd)
    geometries.append(pcd)


# Visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("Manual Fusion Adjustment")

for geo in geometries:
    vis.add_geometry(geo)

current_cam = 1
save_counter = 0

def update_current_pcd():
    global current_cam

    # Step 1: Create point cloud from RGBD
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_list[current_cam], intrinsics[current_cam]
    )

    # Step 2: Convert Open3D point cloud to NumPy array
    pcd_cam = np.asarray(pcd.points)

    # Step 3: Apply transformation using pose
    pose = extrinsics[current_cam]  # Assuming 4x4 pose matrix (C2W)

    R = pose[:3, :3]
    t = pose[:3, 3]

    # Standard camera-to-world transformation
    trans_pcd = (R @ pcd_cam.T + t[:, None]).T

    # Step 4: Create a new point cloud from transformed points
    pcd.points = o3d.utility.Vector3dVector(trans_pcd)

    # Optional: Keep original colors
    if pcd.has_colors():
        geometries[current_cam].colors = pcd.colors

    # Step 5: Update stored geometry
    geometries[current_cam].points = pcd.points
    vis.update_geometry(geometries[current_cam])


def apply_translation(dx, dy, dz):
    extrinsics[current_cam][:3, 3] += np.array([dx, dy, dz])
    update_current_pcd()

def apply_rotation(rx, ry, rz):
    R = o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
    T = extrinsics[current_cam]
    new_rot = R @ T[:3, :3]
    extrinsics[current_cam][:3, :3] = new_rot
    update_current_pcd()

def save_extrinsic(vis):
    global save_counter, current_cam
    out_path = root_dir / f"cam_{current_cam}" / f"extrinsic_adjusted_{save_counter:03d}.npy"
    np.save(out_path, extrinsics[current_cam])
    print(f"[INFO] Saved adjusted extrinsic for cam_{current_cam}: {out_path}")
    save_counter += 1
    return False


# def switch_cam(idx):
#     global current_cam
#     if idx >= num_cams or idx == 0:
#         print("[WARN] Cannot switch to cam_0 or out-of-range.")
#         return False
#     current_cam = idx
#     print(f"[INFO] Switched to cam_{current_cam}")
#     return False

# Keyboard callbacks
vis.register_key_callback(ord("A"), lambda vis: apply_translation(-trans_step, 0, 0))
vis.register_key_callback(ord("D"), lambda vis: apply_translation(trans_step, 0, 0))
vis.register_key_callback(ord("Q"), lambda vis: apply_translation(0, trans_step, 0))
vis.register_key_callback(ord("E"), lambda vis: apply_translation(0, -trans_step, 0))
vis.register_key_callback(ord("W"), lambda vis: apply_translation(0, 0, trans_step))
vis.register_key_callback(ord("X"), lambda vis: apply_translation(0, 0, -trans_step))

vis.register_key_callback(ord("I"), lambda vis: apply_rotation(angle_step, 0, 0))
vis.register_key_callback(ord("K"), lambda vis: apply_rotation(-angle_step, 0, 0))
vis.register_key_callback(ord("J"), lambda vis: apply_rotation(0, angle_step, 0))
vis.register_key_callback(ord("L"), lambda vis: apply_rotation(0, -angle_step, 0))
vis.register_key_callback(ord("U"), lambda vis: apply_rotation(0, 0, angle_step))
vis.register_key_callback(ord("O"), lambda vis: apply_rotation(0, 0, -angle_step))

vis.register_key_callback(ord("S"), save_extrinsic)

# for i in range(1, min(10, num_cams)):
#     vis.register_key_callback(ord(str(i)), lambda vis, i=i: switch_cam(i))

print("\n[INFO] Controls:")
print("  Switch cam: Keys 1–9 (except 0)")
print("  Translation: A/D (X), Q/E (Y), W/X (Z)")
print("  Rotation: I/K (X), J/L (Y), U/O (Z)")
print("  Save: S\n")

vis.run()
vis.destroy_window()
