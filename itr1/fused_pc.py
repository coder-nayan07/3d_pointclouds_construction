import numpy as np
import open3d as o3d
import cv2
import os

# Constants
DEPTH_SCALE = 1000.0  # Scale factor for depth to meters

# Path to your data (RGB, Depth, Intrinsics, Extrinsics for each camera)
DATA_PATH = "D:\codes\camera_calib_IIITA"
NUM_CAMERAS = 2  # Total number of cameras

# Function to load camera data dynamically
def load_camera_data(camera_id):
    color_img_path = os.path.join(DATA_PATH, f'cam_{camera_id}', 'rgb', '1.png')  # RGB image
    depth_img_path = os.path.join(DATA_PATH, f'cam_{camera_id}', 'depth', '1.png')  # Depth image
    intrinsic_path = os.path.join(DATA_PATH, f'cam_{camera_id}', 'camera_intrinsic_matrix.npy')  # Intrinsic matrix
    extrinsic_path = os.path.join(DATA_PATH, f'cam_{camera_id}', 'camera_extrinsic_matrix.npy')  # Extrinsic matrix
    
    # Load RGB and Depth images
    color_img = cv2.imread(color_img_path)
    depth_img_raw = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED) / DEPTH_SCALE  # Convert depth to meters
    
    # Load Intrinsic and Extrinsic matrices
    intrinsic_matrix = np.load(intrinsic_path)
    extrinsic_matrix = np.load(extrinsic_path)
    
    return color_img, depth_img_raw, intrinsic_matrix, extrinsic_matrix

# Function to create point cloud from depth and RGB images
def create_point_cloud(depth_image, color_image, intrinsic_matrix):
    height, width = depth_image.shape
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_image
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    colors = color_image.reshape((-1, 3)) / 255.0  # Normalize RGB

    # Filter out invalid depth points
    valid_points = z.flatten() > 0
    points = points[valid_points]
    colors = colors[valid_points]

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

# Main processing loop
if __name__ == "__main__":
    pcd_list = []  # List to hold point clouds of all cameras
    
    # Load data for all cameras and create point clouds
    for cam_id in range(NUM_CAMERAS):
        print(f"Loading data for camera {cam_id}...")
        color_img, depth_img, K, E = load_camera_data(cam_id)
        
        # Create point cloud for this camera
        pcd = create_point_cloud(depth_img, color_img, K)
        
        # Apply the extrinsic matrix to the point cloud (Camera to World transformation)
        pcd.transform(E)
        
        # Store point cloud
        pcd_list.append(pcd)
        
    # Visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Paint each point cloud a different color for distinction
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()
