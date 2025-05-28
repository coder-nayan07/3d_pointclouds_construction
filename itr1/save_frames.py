import pyrealsense2 as rs
import cv2
import numpy as np
import os
import time

# --- Configuration ---
# Directory to save images (aligned depth and color)
output_directory = r"D:\codes\camera_calib_IIITA\cam_1"
frame_counter = 0                     # Counter for naming files
max_frames_to_save = 1              # Save exactly 2 pairs

# Configure stream resolution and format
# Note: Adjust these based on your camera's capabilities and desired output
# It's good practice to use resolutions where intrinsics are well-defined (like 640x480)
rgb_width = 640
rgb_height = 480
depth_width = 640
depth_height = 480 # Depth resolution should ideally match or be lower than RGB for alignment
fps = 30

# --- Setup ---
# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)
print(f"Saving exactly {max_frames_to_save} aligned pairs of frames to '{output_directory}'")

# Create a context object. This object owns the sensors and frames.
pipeline = rs.pipeline()

# Configure the pipeline
config = rs.config()
config.enable_stream(rs.stream.color, rgb_width, rgb_height, rs.format.bgr8, fps)
# Use rs.format.z16 for raw 16-bit depth values
config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)

# Create an align object
# rs.align allows us to align the depth frame to other frames
# The "color" stream is chosen as the target stream for alignment
align_to = rs.stream.color
align = rs.align(align_to)

# --- Start Streaming ---
try:
    profile = pipeline.start(config)
    print("Streaming started. Waiting for initial frames and aligning...")

    # Skip a few frames to allow camera to stabilize
    for i in range(30):
        pipeline.wait_for_frames()

    print(f"Starting capture loop. Will save {max_frames_to_save} aligned pairs and then stop.")

    # --- Capture and Save Loop ---
    while True:
        # Wait for a coherent frameset of depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to the color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame() # Color frame is also part of aligned_frames

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            print("Skipping frame: aligned streams not available")
            continue # Skip this frame

        frame_counter += 1

        # Convert aligned depth and color frames to numpy arrays
        # Note: aligned_depth_image will have the same dimensions as color_image
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # --- Save the frames ---
        # Use os.path.join for cross-platform compatibility
        rgb_filename = os.path.join(output_directory, f"rgb_aligned_cam2_{frame_counter}.png")
        depth_filename = os.path.join(output_directory, f"depth_aligned_cam2_{frame_counter}.png") # Indicate it's aligned

        # Save images using OpenCV
        # Save color as BGR (default for imwrite)
        cv2.imwrite(rgb_filename, color_image)
        # Save 16-bit depth as UNCHANGED PNG
        cv2.imwrite(depth_filename, aligned_depth_image)

        print(f"Saved aligned frame {frame_counter}: {rgb_filename}, {depth_filename}")

        # --- Display the frames (Optional) ---
        # Apply colormap to depth image for better visualization
        # Scale the depth data for 8-bit visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Display the images
        cv2.imshow("RealSense Color (Aligned)", color_image)
        cv2.imshow("RealSense Aligned Depth Colormap", depth_colormap)

        # --- Check if max frames reached ---
        if frame_counter >= max_frames_to_save:
            print(f"Successfully saved {max_frames_to_save} aligned pairs. Stopping.")
            break # Exit the loop after saving the desired number of pairs

        # --- Add waitKey for display updates and optional early exit ---
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27: # Allow exiting early with 'q' or Escape
            print("Early exit requested by user.")
            break


finally:
    # --- Stop Streaming and Cleanup ---
    pipeline.stop()
    print("Pipeline stopped.")
    cv2.destroyAllWindows()
    print("OpenCV windows closed.")