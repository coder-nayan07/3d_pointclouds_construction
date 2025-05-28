import numpy as np
import cv2
import open3d as o3d
from pathlib import Path

# Assuming rs_utils and generate_pc are available and work as expected
try:
    from rs_utils import get_connected_devices_serial_numbers
except ImportError:
    print("Warning: rs_utils.py not found or get_connected_devices_serial_numbers not defined. Using placeholder.")
    def get_connected_devices_serial_numbers(): # Placeholder
        return []
try:
    from generate_pc import create_point_cloud_from_depth_and_color
except ImportError:
    print("Error: generate_pc.py not found or create_point_cloud_from_depth_and_color not defined.")
    print("Please ensure this file and function are available.")
    exit()


# --- Script Settings ---
VOXEL_SIZE_DOWNSAMPLE_INPUT_PCD = 0.01
VOXEL_SIZE_DOWNSAMPLE_FINAL_PCD = 0.005
ROOT_DIR = Path("itr3")

# --- ICP Settings ---
ICP_MAX_CORRESPONDENCE_DISTANCE = 0.05
ICP_CONVERGENCE_CRITERIA = o3d.pipelines.registration.ICPConvergenceCriteria(
    relative_fitness=1e-7,
    relative_rmse=1e-7,
    max_iteration=200 # Note: Default Open3D is 30 or 100. 200k is very high.
)

# --- BOUNDING BOX for ROI in WORLD COORDINATES ---
ROI_MIN_BOUND = np.array([-0.5, -0.5, -1])
ROI_MAX_BOUND = np.array([0.5, 0.5, 1])
USE_BOUNDING_BOX_FILTER = True # As per user's provided code


def create_bounding_box(min_bound, max_bound):
    """Creates an o3d.geometry.AxisAlignedBoundingBox."""
    return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

def register_point_clouds_icp(source_pcd, target_pcd, initial_transform=np.identity(4),
                              max_correspondence_distance=0.05,
                              criteria=None):
    print(f"  ICP: Registering source ({len(source_pcd.points)} pts) to target ({len(target_pcd.points)} pts)")
    if not source_pcd.has_points() or not target_pcd.has_points():
        print("  ICP: Source or target point cloud is empty. Skipping registration.")
        # Return the initial_transform and a copy of source (which might be empty)
        return initial_transform, o3d.geometry.PointCloud(source_pcd)

    source_pcd_copy = o3d.geometry.PointCloud(source_pcd)
    target_pcd_copy = o3d.geometry.PointCloud(target_pcd)

    if not source_pcd_copy.has_normals():
        source_pcd_copy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    if not target_pcd_copy.has_normals():
        target_pcd_copy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    if criteria is None: # Use global if not provided
        criteria = ICP_CONVERGENCE_CRITERIA

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd_copy, target_pcd_copy,
        max_correspondence_distance,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria
    )
    print(f"  ICP Fitness: {reg_p2p.fitness:.4f}, Inlier RMSE: {reg_p2p.inlier_rmse:.4f}")
    
    source_pcd_transformed = o3d.geometry.PointCloud(source_pcd) 
    source_pcd_transformed.transform(reg_p2p.transformation)
    return reg_p2p.transformation, source_pcd_transformed


if __name__ == "__main__":
    print("Attempting to fuse point clouds WITH ICP REFINEMENT.")
    if USE_BOUNDING_BOX_FILTER:
        print("Bounding box filtering IS ENABLED.")
        print(f"ROI Bounding Box: MIN={ROI_MIN_BOUND}, MAX={ROI_MAX_BOUND}")
    else:
        print("Bounding box filtering IS DISABLED.")
    print("Looking for camera data under:", ROOT_DIR)


    detected_serials = get_connected_devices_serial_numbers()
    if not detected_serials:
        print("No RealSense devices connected. Will use disk data if available.")
    else:
        print("\nConnected RealSense devices:")
        for s in detected_serials:
            print(f"  - {s}")

    candidate_cam_folders = []
    if not ROOT_DIR.exists():
        print(f"ERROR: ROOT_DIR '{ROOT_DIR}' does not exist.")
        exit()
        
    for item in ROOT_DIR.iterdir():
        if item.is_dir() and item.name.startswith("cam_"):
            intr_file = item / "intrinsics.npy"
            dist_file = item / "distortion.npy"
            extr_file = item / "T_world_from_camera.npy"
            rgb_dir = item / "rgb"
            depth_dir = item / "depth"

            if intr_file.exists() and dist_file.exists() and extr_file.exists() and \
               rgb_dir.exists() and depth_dir.exists() and \
               any(rgb_dir.glob("*.png")) and any(depth_dir.glob("*.npy")):
                candidate_cam_folders.append(item.name)

    cam_folders_to_fuse = []
    if not candidate_cam_folders:
        print(f"\nNo valid camera directories found under {ROOT_DIR} with all required files.")
        manual_folders = input(f"Enter cam folder names (e.g., cam_0,cam_1) from {ROOT_DIR} to use, or leave blank to exit: ")
        if not manual_folders:
            exit()
        cam_folders_to_fuse = [s.strip() for s in manual_folders.split(',')]
    else:
        print("\nFound pre-calibrated camera folders ready for fusion:")
        for i, folder_name in enumerate(candidate_cam_folders):
            print(f"  {i}: {folder_name}")
        choice = input("Enter indices of folders to fuse (e.g., 0,1), 'all', or specific folder names (comma-separated): ")
        if choice.lower() == 'all':
            cam_folders_to_fuse = candidate_cam_folders
        else:
            try:
                indices = [int(x.strip()) for x in choice.split(',')]
                for idx in indices:
                    if 0 <= idx < len(candidate_cam_folders):
                        cam_folders_to_fuse.append(candidate_cam_folders[idx])
                    else:
                        print(f"Warning: Index {idx} is out of range. Skipping.")
            except ValueError:
                cam_folders_to_fuse = [s.strip() for s in choice.split(',')]

    if not cam_folders_to_fuse:
        print("No camera folders selected for fusion. Exiting.")
        exit()

    print(f"\nAttempting to fuse point clouds from folders: {cam_folders_to_fuse}")

    master_fused_pcd = o3d.geometry.PointCloud()
    processed_pcds_for_visualization = [] 
    all_final_extrinsics = {} 
    frame_index_to_fuse = 0 

    if USE_BOUNDING_BOX_FILTER:
        world_bbox = create_bounding_box(ROI_MIN_BOUND, ROI_MAX_BOUND)

    for i, cam_folder_name in enumerate(cam_folders_to_fuse):
        print(f"\n--- Processing Camera Folder: {cam_folder_name} ---")
        cam_dir = ROOT_DIR / cam_folder_name
        if not cam_dir.is_dir():
            print(f"Directory {cam_dir} not found. Skipping.")
            continue

        intrinsics_file = cam_dir / "intrinsics.npy"
        distortion_file = cam_dir / "distortion.npy"
        extrinsics_file = cam_dir / "T_world_from_camera.npy"
        
        try:
            rgb_files = sorted(list((cam_dir / "rgb").glob("*.png")))
            depth_files = sorted(list((cam_dir / "depth").glob("*.npy")))
            
            if not rgb_files or not depth_files:
                print(f"  No RGB/Depth files found in {cam_dir}. Skipping.")
                continue
            current_frame_idx = frame_index_to_fuse if frame_index_to_fuse < len(rgb_files) else 0
            if current_frame_idx >= len(rgb_files) or current_frame_idx >= len(depth_files):
                print(f"  Frame index {current_frame_idx} out of bounds for {cam_folder_name}. Skipping.")
                continue
            rgb_file_to_load = rgb_files[current_frame_idx]
            rgb_filename_no_ext = rgb_file_to_load.stem
            matching_depth_file = cam_dir / "depth" / f"{rgb_filename_no_ext}.npy"
            if not matching_depth_file.exists():
                matching_depth_file = depth_files[current_frame_idx] 
            depth_file_to_load = matching_depth_file
        except IndexError:
            print(f"  Error accessing frame data for {cam_folder_name}. Skipping.")
            continue

        required_files_present = True
        for f_check_path_str in [rgb_file_to_load, depth_file_to_load, intrinsics_file, distortion_file, extrinsics_file]:
            f_check = Path(f_check_path_str)
            if not f_check.exists():
                print(f"    Missing: {f_check}")
                required_files_present = False
        if not required_files_present:
            print(f"  Missing required files for {cam_folder_name}. Check paths and ensure calibration was run.")
            continue
        
        print(f"  Loading data: RGB='{rgb_file_to_load.name}', Depth='{depth_file_to_load.name}'")
        color_img = cv2.imread(str(rgb_file_to_load))
        if color_img is None:
            print(f"  Failed to load color image {rgb_file_to_load}. Skipping.")
            continue
        depth_img_m = np.load(str(depth_file_to_load))
        K = np.load(str(intrinsics_file))
        D = np.load(str(distortion_file))
        T_world_from_cam_extrinsic_initial = np.load(str(extrinsics_file))

        print(f"  Generating local point cloud for {cam_folder_name}...")
        pcd_local = create_point_cloud_from_depth_and_color(depth_img_m, color_img, K, D)
        
        if not pcd_local.has_points():
            print(f"  Local point cloud for {cam_folder_name} is empty. Skipping.")
            continue

        current_pcd_in_world_full_res = o3d.geometry.PointCloud(pcd_local)
        current_pcd_in_world_full_res.transform(T_world_from_cam_extrinsic_initial)
        print(f"  Transformed to world with initial extrinsic. Points: {len(current_pcd_in_world_full_res.points)}")

        # pcd_to_process is the cloud after initial transform and ROI filtering (if enabled).
        # This is the version that, after ICP refinement, will be added to the master cloud.
        pcd_to_process = o3d.geometry.PointCloud(current_pcd_in_world_full_res)

        if USE_BOUNDING_BOX_FILTER and pcd_to_process.has_points():
            points_before_filter = len(pcd_to_process.points)
            pcd_to_process = pcd_to_process.crop(world_bbox)
            print(f"  Applied ROI bounding box filter to current camera's cloud. Points: {len(pcd_to_process.points)} (from {points_before_filter})")
            if not pcd_to_process.has_points():
                print(f"  Cloud empty after ROI filter for {cam_folder_name}. Skipping.")
                all_final_extrinsics[cam_folder_name] = np.copy(T_world_from_cam_extrinsic_initial) 
                continue
        
        # pcd_for_icp_source is derived from pcd_to_process, potentially downsampled for ICP efficiency.
        pcd_for_icp_source = o3d.geometry.PointCloud(pcd_to_process) 

        if VOXEL_SIZE_DOWNSAMPLE_INPUT_PCD > 0 and pcd_for_icp_source.has_points():
            pcd_downsampled = pcd_for_icp_source.voxel_down_sample(VOXEL_SIZE_DOWNSAMPLE_INPUT_PCD)
            if pcd_downsampled.has_points():
                print(f"  Downsampled cloud from {len(pcd_for_icp_source.points)} to {len(pcd_downsampled.points)} points for ICP source.")
                pcd_for_icp_source = pcd_downsampled
            else:
                print(f"  Downsampling for ICP source resulted in empty cloud. Using original (filtered) cloud for ICP source.")
        
        if not pcd_for_icp_source.has_points():
            print(f"  Cloud for ICP source is empty for {cam_folder_name}. Skipping ICP, will use initial extrinsic.")
            all_final_extrinsics[cam_folder_name] = np.copy(T_world_from_cam_extrinsic_initial)
            if pcd_to_process.has_points(): # Add the non-input-downsampled, ROI-filtered cloud
                 master_fused_pcd += pcd_to_process
                 processed_pcds_for_visualization.append(o3d.geometry.PointCloud(pcd_to_process))
            continue

        # --- ICP REGISTRATION ---
        if i == 0:
            # Base for master_fused_pcd is the ROI-filtered, non-input-downsampled cloud
            master_fused_pcd = o3d.geometry.PointCloud(pcd_to_process) 
            print(f"  {cam_folder_name} is the first camera. Its cloud (ROI filtered, not input-downsampled for ICP) is the base.")
            processed_pcds_for_visualization.append(o3d.geometry.PointCloud(pcd_to_process))
            all_final_extrinsics[cam_folder_name] = np.copy(T_world_from_cam_extrinsic_initial)
        else:
            if not master_fused_pcd.has_points():
                print("  Master fused PCD is empty. Adding current cloud (ROI filtered, not input-downsampled for ICP) with initial extrinsic.")
                if pcd_to_process.has_points(): 
                     master_fused_pcd = o3d.geometry.PointCloud(pcd_to_process)
                     processed_pcds_for_visualization.append(o3d.geometry.PointCloud(pcd_to_process))
                all_final_extrinsics[cam_folder_name] = np.copy(T_world_from_cam_extrinsic_initial)
                continue

            print(f"  Refining alignment of {cam_folder_name} to accumulated cloud using ICP...")
            
            # Prepare target for ICP
            target_for_icp = o3d.geometry.PointCloud(master_fused_pcd) # Work with a copy

            if USE_BOUNDING_BOX_FILTER and target_for_icp.has_points(): # Filter target with ROI
                points_before_filter_target = len(target_for_icp.points)
                target_for_icp = target_for_icp.crop(world_bbox)
                print(f"  ICP Target: Applied ROI filter. Points: {len(target_for_icp.points)} (from {points_before_filter_target})")

            if VOXEL_SIZE_DOWNSAMPLE_INPUT_PCD > 0 and target_for_icp.has_points(): # Downsample target
                 temp_target_downsampled = target_for_icp.voxel_down_sample(VOXEL_SIZE_DOWNSAMPLE_INPUT_PCD)
                 if temp_target_downsampled.has_points():
                     print(f"  ICP Target: Downsampled from {len(target_for_icp.points)} to {len(temp_target_downsampled.points)} pts.")
                     target_for_icp = temp_target_downsampled
                 else:
                     print(f"  ICP Target: Downsampling resulted in empty cloud. Using (filtered) cloud pre-downsample.")
            
            # ICP source (pcd_for_icp_source) is already in world coords via T_world_from_cam_extrinsic_initial.
            # So, initial_transform for ICP is identity, as we are refining its already world-posed position.
            icp_refinement_matrix, pcd_icp_registered_source_downsampled = register_point_clouds_icp(
                pcd_for_icp_source, # This is (ROI filtered, potentially input-downsampled)
                target_for_icp,     # This is (accumulated, ROI filtered, potentially input-downsampled)
                np.identity(4),     # Initial guess for refinement
                ICP_MAX_CORRESPONDENCE_DISTANCE,
                ICP_CONVERGENCE_CRITERIA
            )
            
            current_final_extrinsic = icp_refinement_matrix @ T_world_from_cam_extrinsic_initial
            all_final_extrinsics[cam_folder_name] = current_final_extrinsic
            
            # Prepare the cloud to add to the master: use the non-input-downsampled (but ROI filtered)
            # pcd_to_process, and apply the ICP refinement to it.
            pcd_to_add_to_master = o3d.geometry.PointCloud(pcd_to_process)
            pcd_to_add_to_master.transform(icp_refinement_matrix) # Apply ICP refinement
            
            if pcd_to_add_to_master.has_points():
                master_fused_pcd += pcd_to_add_to_master
                processed_pcds_for_visualization.append(o3d.geometry.PointCloud(pcd_to_add_to_master))
                print(f"  Added refined cloud (from {len(pcd_to_process.points)} original pts, now {len(pcd_to_add_to_master.points)}) to master.")
            else:
                print("  ICP registration or subsequent transform resulted in an empty cloud. Not adding.")

    # --- Save Refined Extrinsics ---
    if all_final_extrinsics:
        print("\n--- Saving Refined Extrinsic Matrices ---")
        for cam_name, final_ext_matrix in all_final_extrinsics.items():
            output_path = ROOT_DIR / cam_name / "T_world_from_camera_refined_icp.npy"
            try:
                np.save(output_path, final_ext_matrix)
                print(f"  Saved refined extrinsic for {cam_name} to: {output_path}")
            except Exception as e:
                print(f"  Error saving refined extrinsic for {cam_name} to {output_path}: {e}")
    else:
        print("\nNo refined extrinsics to save.")


    if not master_fused_pcd.has_points():
        print("\nNo point clouds were successfully generated or fused. Exiting.")
        exit()

    print(f"\nTotal points in fused cloud (before final downsampling): {len(master_fused_pcd.points)}")

    final_pcd_to_show = master_fused_pcd
    if VOXEL_SIZE_DOWNSAMPLE_FINAL_PCD > 0 and master_fused_pcd.has_points():
        print(f"Downsampling final fused point cloud with voxel size: {VOXEL_SIZE_DOWNSAMPLE_FINAL_PCD} m")
        final_pcd_to_show = master_fused_pcd.voxel_down_sample(VOXEL_SIZE_DOWNSAMPLE_FINAL_PCD)
        print(f"Total points in final fused cloud (after downsampling): {len(final_pcd_to_show.points)}")

    if final_pcd_to_show.has_points():
        output_fused_file = ROOT_DIR / f"fused_point_cloud_icp_roi_{len(cam_folders_to_fuse)}cams.ply"
        o3d.io.write_point_cloud(str(output_fused_file), final_pcd_to_show, write_ascii=False)
        print(f"Saved fused point cloud to {output_fused_file}")

        print("Visualizing final fused point cloud...")
        geometries_to_draw = [final_pcd_to_show]
        if USE_BOUNDING_BOX_FILTER: # Only draw bbox if it was used
            bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(world_bbox)
            bbox_lines.paint_uniform_color([0.8, 0.2, 0.2]) # Red color for bbox
            geometries_to_draw.append(bbox_lines)
        
        o3d.visualization.draw_geometries(geometries_to_draw, window_name="Fused Point Cloud (ICP Refined, ROI Filtered)", width=1280, height=720)
        
    else:
        print("Final fused point cloud is empty. Nothing to save or show.")