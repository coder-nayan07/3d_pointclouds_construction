import numpy as np
import cv2
import open3d as o3d
from pathlib import Path

# Attempt to import the user's point cloud generation function
try:
    from generate_pc import create_point_cloud_from_depth_and_color
    print("Using create_point_cloud_from_depth_and_color from generate_pc.py")
except ImportError:
    print("Warning: generate_pc.py not found or create_point_cloud_from_depth_and_color not defined.")
    print("Using a placeholder function for point cloud generation.")
    print("Please ensure your generate_pc.py and its function are available for optimal results.")

    def create_point_cloud_from_depth_and_color(depth_img_m, color_img_bgr, K, D=None, depth_scale=1.0, depth_trunc=3.0):
        """
        Placeholder function to create an Open3D PointCloud from depth and color images.
        Assumes depth and color images are registered.
        Handles basic color conversion and uses Open3D's RGBDImage pipeline.
        Distortion D is accepted but NOT USED in this placeholder.
        """
        if D is not None and np.any(D):
            print("      Placeholder Warning: Distortion coefficients D were provided but are NOT used by this placeholder function.")

        height, width = depth_img_m.shape
        if color_img_bgr.shape[:2] != (height, width):
            print(f"      Placeholder Warning: Color image shape {color_img_bgr.shape[:2]} and depth image shape {(height, width)} mismatch. Resizing color.")
            color_img_bgr = cv2.resize(color_img_bgr, (width, height), interpolation=cv2.INTER_NEAREST)

        color_img_rgb = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2RGB)
        o3d_depth = o3d.geometry.Image(depth_img_m.astype(np.float32))
        o3d_color = o3d.geometry.Image(color_img_rgb.astype(np.uint8))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, o3d_intrinsic, project_valid_depth_only=True
        )
        if not pcd.has_points():
            print("      Placeholder: Generated point cloud is empty.")
        else:
            print(f"      Placeholder: Generated point cloud with {len(pcd.points)} points.")
        return pcd

# --- Script Settings ---
VOXEL_SIZE_DOWNSAMPLE_INPUT_PCD = 0.01
VOXEL_SIZE_DOWNSAMPLE_FINAL_PCD = 0.005
ROOT_DIR = Path("itr3")
FRAME_INDEX_TO_FUSE = 0

def main():
    print("Starting simple point cloud fusion (NO ICP, NO Bounding Box Alignment).")
    print(f"Looking for camera data under: {ROOT_DIR}")
    print(f"Voxel size for downsampling individual clouds: {VOXEL_SIZE_DOWNSAMPLE_INPUT_PCD if VOXEL_SIZE_DOWNSAMPLE_INPUT_PCD > 0 else 'Disabled'}")
    print(f"Voxel size for downsampling final fused cloud: {VOXEL_SIZE_DOWNSAMPLE_FINAL_PCD if VOXEL_SIZE_DOWNSAMPLE_FINAL_PCD > 0 else 'Disabled'}")

    if not ROOT_DIR.exists():
        print(f"ERROR: ROOT_DIR '{ROOT_DIR}' does not exist. Exiting.")
        return

    candidate_cam_folders = []
    print("\nScanning for valid camera folders...")
    for item in ROOT_DIR.iterdir():
        if item.is_dir() and item.name.startswith("cam_"):
            intr_file = item / "intrinsics.npy"
            dist_file = item / "distortion.npy"
            extr_file = item / "T_world_from_camera_refined_icp.npy"
            rgb_dir = item / "rgb"
            depth_dir = item / "depth"

            missing_files = []
            if not intr_file.exists(): missing_files.append(intr_file.name)
            if not dist_file.exists(): missing_files.append(dist_file.name)
            if not extr_file.exists(): missing_files.append(extr_file.name)
            if not rgb_dir.exists() or not any(rgb_dir.glob("*.png")): missing_files.append(f"RGB data in {rgb_dir.name}/")
            if not depth_dir.exists() or not any(depth_dir.glob("*.npy")): missing_files.append(f"Depth data in {depth_dir.name}/")

            if not missing_files:
                candidate_cam_folders.append(item.name)
                print(f"  Found valid: {item.name}")
            else:
                print(f"  Skipping '{item.name}'. Missing: {', '.join(missing_files)}")

    cam_folders_to_fuse = []
    if not candidate_cam_folders:
        print(f"\nNo valid camera directories found under {ROOT_DIR} with all required files.")
        manual_folders = input(f"Enter cam folder names (e.g., cam_0,cam_1) from {ROOT_DIR} to use, or leave blank to exit: ").strip()
        if not manual_folders:
            print("No folders specified. Exiting.")
            return
        cam_folders_to_fuse = [s.strip() for s in manual_folders.split(',')]
    else:
        print("\nFound pre-calibrated camera folders ready for fusion:")
        for i, folder_name in enumerate(candidate_cam_folders):
            print(f"  {i}: {folder_name}")
        
        while True:
            choice = input("Enter indices of folders to fuse (e.g., 0,1), 'all', or specific folder names (comma-separated, e.g., cam_0,cam_1): ").strip()
            if choice.lower() == 'all':
                cam_folders_to_fuse = candidate_cam_folders
                break
            elif not choice:
                print("No selection made. Please enter indices, 'all', or folder names.")
                continue
            else:
                selected_by_name = []
                selected_by_index = []
                invalid_indices = []
                invalid_names = []
                
                parts = [p.strip() for p in choice.split(',')]
                for part in parts:
                    if part in candidate_cam_folders: # Check if it's a direct name match
                        if part not in selected_by_name: # Avoid duplicates if named multiple times
                             selected_by_name.append(part)
                    else:
                        try:
                            idx = int(part)
                            if 0 <= idx < len(candidate_cam_folders):
                                folder_name_from_index = candidate_cam_folders[idx]
                                if folder_name_from_index not in selected_by_index and folder_name_from_index not in selected_by_name:
                                    selected_by_index.append(folder_name_from_index)
                            else:
                                invalid_indices.append(part)
                        except ValueError: # Not a name and not an int, so invalid name
                            invalid_names.append(part)
                
                cam_folders_to_fuse = selected_by_name + selected_by_index # Combine, names take precedence if listed
                
                if invalid_indices:
                    print(f"Warning: The following indices are out of range: {', '.join(invalid_indices)}")
                if invalid_names:
                    print(f"Warning: The following folder names were not found in the valid list: {', '.join(invalid_names)}")
                
                if cam_folders_to_fuse:
                    break
                else:
                    print("No valid folders selected from your input. Please try again.")


    if not cam_folders_to_fuse:
        print("No camera folders selected for fusion. Exiting.")
        return

    print(f"\nAttempting to fuse point clouds from folders: {', '.join(cam_folders_to_fuse)}")

    master_fused_pcd = o3d.geometry.PointCloud()
    successful_fusions = 0

    for cam_folder_name in cam_folders_to_fuse:
        print(f"\n--- Processing Camera Folder: {cam_folder_name} ---")
        cam_dir = ROOT_DIR / cam_folder_name
        
        # Double check if the selected folder is actually valid (could be from manual input)
        if not cam_dir.is_dir():
            print(f"  Directory {cam_dir} for selected folder '{cam_folder_name}' not found. Skipping.")
            continue

        intrinsics_file = cam_dir / "intrinsics.npy"
        distortion_file = cam_dir / "distortion.npy"
        extrinsics_file = cam_dir / "T_world_from_camera_refined_icp.npy"
        
        # Verify essential files again for manually entered names
        essential_files_ok = True
        if not intrinsics_file.exists(): print(f"  Missing: {intrinsics_file}"); essential_files_ok = False
        if not distortion_file.exists(): print(f"  Missing: {distortion_file}"); essential_files_ok = False
        if not extrinsics_file.exists(): print(f"  Missing: {extrinsics_file}"); essential_files_ok = False
        if not essential_files_ok:
            print(f"  Essential .npy files missing for {cam_folder_name}. Skipping.")
            continue

        try:
            rgb_files = sorted(list((cam_dir / "rgb").glob("*.png")))
            depth_files = sorted(list((cam_dir / "depth").glob("*.npy")))

            if not rgb_files or not depth_files:
                print(f"  No RGB/Depth files found in {cam_dir}. Skipping.")
                continue

            current_frame_idx = FRAME_INDEX_TO_FUSE
            if current_frame_idx >= len(rgb_files) or current_frame_idx >= len(depth_files):
                print(f"  Frame index {current_frame_idx} is out of bounds for {cam_folder_name} (RGBs: {len(rgb_files)}, Depths: {len(depth_files)}). Using index 0 if available, else skipping.")
                current_frame_idx = 0
                if current_frame_idx >= len(rgb_files) or current_frame_idx >= len(depth_files):
                     print(f"  Fallback to frame index 0 also out of bounds. Skipping {cam_folder_name}.")
                     continue

            rgb_file_to_load = rgb_files[current_frame_idx]
            rgb_filename_no_ext = rgb_file_to_load.stem
            matching_depth_file = cam_dir / "depth" / f"{rgb_filename_no_ext}.npy"

            if not matching_depth_file.exists():
                print(f"  Depth file {matching_depth_file.name} not found. Falling back to depth file at index {current_frame_idx}: {depth_files[current_frame_idx].name}")
                depth_file_to_load = depth_files[current_frame_idx]
            else:
                depth_file_to_load = matching_depth_file

        except IndexError:
            print(f"  Error accessing frame data for {cam_folder_name} at index {FRAME_INDEX_TO_FUSE}. Skipping.")
            continue
        
        print(f"  Loading data: RGB='{rgb_file_to_load.name}', Depth='{depth_file_to_load.name}'")
        
        try:
            color_img = cv2.imread(str(rgb_file_to_load))
            if color_img is None:
                print(f"  Failed to load color image {rgb_file_to_load}. Skipping.")
                continue
            
            depth_img_m = np.load(str(depth_file_to_load))
            K = np.load(str(intrinsics_file))
            D = np.load(str(distortion_file))
            T_world_from_cam = np.load(str(extrinsics_file))
        except Exception as e:
            print(f"  Error loading files for {cam_folder_name}: {e}. Skipping.")
            continue

        print(f"  Generating local point cloud for {cam_folder_name}...")
        pcd_local = create_point_cloud_from_depth_and_color(depth_img_m, color_img, K, D)
        
        if not pcd_local or not pcd_local.has_points():
            print(f"  Local point cloud for {cam_folder_name} is empty or failed to generate. Skipping.")
            continue
        
        print(f"  Generated local cloud with {len(pcd_local.points)} points.")

        pcd_local.transform(T_world_from_cam)
        print(f"  Transformed to world coordinates. Points: {len(pcd_local.points)}")

        if VOXEL_SIZE_DOWNSAMPLE_INPUT_PCD > 0 and pcd_local.has_points():
            points_before_downsample = len(pcd_local.points)
            pcd_local_downsampled = pcd_local.voxel_down_sample(VOXEL_SIZE_DOWNSAMPLE_INPUT_PCD)
            if pcd_local_downsampled.has_points():
                pcd_local = pcd_local_downsampled
                print(f"  Downsampled individual cloud from {points_before_downsample} to {len(pcd_local.points)} points.")
            else:
                print(f"  Downsampling individual cloud resulted in an empty cloud. Using original transformed cloud.")
        
        if pcd_local.has_points():
            master_fused_pcd += pcd_local
            successful_fusions += 1
            print(f"  Added to master cloud. Master cloud now has approx. {len(master_fused_pcd.points)} points.")
        else:
            print(f"  Skipping adding {cam_folder_name} cloud as it became empty after processing.")

    if not master_fused_pcd.has_points():
        print("\nNo point clouds were successfully generated or fused. Exiting.")
        return

    print(f"\nTotal points in raw fused cloud (from {successful_fusions} camera(s)): {len(master_fused_pcd.points)}")

    final_pcd_to_show_and_save = master_fused_pcd
    if VOXEL_SIZE_DOWNSAMPLE_FINAL_PCD > 0 and master_fused_pcd.has_points():
        points_before_final_downsample = len(master_fused_pcd.points)
        print(f"Downsampling final fused point cloud with voxel size: {VOXEL_SIZE_DOWNSAMPLE_FINAL_PCD} m")
        final_pcd_to_show_and_save = master_fused_pcd.voxel_down_sample(VOXEL_SIZE_DOWNSAMPLE_FINAL_PCD)
        if not final_pcd_to_show_and_save.has_points() and master_fused_pcd.has_points(): # check master_fused_pcd again
            print("  Final downsampling resulted in an empty cloud. Using pre-downsample cloud.")
            final_pcd_to_show_and_save = master_fused_pcd
        else:
            print(f"  Total points in final fused cloud (after downsampling): {len(final_pcd_to_show_and_save.points)} (from {points_before_final_downsample})")
    
    if final_pcd_to_show_and_save.has_points():
        output_fused_file = ROOT_DIR / f"fused_point_cloud_simple_{successful_fusions}cams_selected.ply"
        try:
            o3d.io.write_point_cloud(str(output_fused_file), final_pcd_to_show_and_save, write_ascii=False)
            print(f"Saved final fused point cloud to {output_fused_file}")
        except Exception as e:
            print(f"Error saving point cloud to {output_fused_file}: {e}")

        print("Visualizing final fused point cloud...")
        o3d.visualization.draw_geometries([final_pcd_to_show_and_save], 
                                          window_name=f"Fused Point Cloud (Simple, {successful_fusions} Selected Cams)",
                                          width=1280, height=720)
    else:
        print("Final fused point cloud is empty. Nothing to save or show.")

if __name__ == "__main__":
    main()