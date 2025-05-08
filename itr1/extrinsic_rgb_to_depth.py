import pyrealsense2 as rs
import numpy as np

try:
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable streams (adjust resolution/fps as needed, though extrinsics are resolution-independent)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming to get stream profiles
    profile = pipeline.start(config)

    # Get the stream profiles for depth and color
    depth_profile = profile.get_stream(rs.stream.depth)
    color_profile = profile.get_stream(rs.stream.color)

    # Get the extrinsic parameters from depth to color
    # This gives you the transformation to move points from the depth frame's
    # coordinate system to the color frame's coordinate system.
    print("Attempting to get extrinsics from Depth to Color...")
    try:
        depth_to_color_extrinsics = depth_profile.get_extrinsics_to(color_profile)

        print("\n--- Extrinsics from Depth to Color ---")
        print("Rotation Matrix:")
        # The rotation is a 3x3 matrix
        rotation = np.asarray(depth_to_color_extrinsics.rotation).reshape(3, 3)
        print(rotation)

        print("\nTranslation Vector (meters):")
        # The translation is a 3x1 vector, usually in meters
        translation = np.asarray(depth_to_color_extrinsics.translation)
        print(translation) # Typically in meters [Tx, Ty, Tz]

        # You can combine R and T into a 4x4 homogeneous matrix if needed:
        # Extrinsic_Matrix = | R   T |
        #                    | 0   1 |
        Extrinsic_Matrix_Depth_To_Color = np.eye(4)
        Extrinsic_Matrix_Depth_To_Color[:3, :3] = rotation
        Extrinsic_Matrix_Depth_To_Color[:3, 3] = translation
        print("\nHomogeneous Extrinsic Matrix (Depth -> Color):")
        print(Extrinsic_Matrix_Depth_To_Color)

    except RuntimeError as e:
        print(f"Could not get extrinsics from Depth to Color. Error: {e}")
        print("Ensure both Depth and Color streams are enabled.")


    # Get the extrinsic parameters from color to depth (the inverse transformation)
    print("\nAttempting to get extrinsics from Color to Depth...")
    try:
        color_to_depth_extrinsics = color_profile.get_extrinsics_to(depth_profile)

        print("\n--- Extrinsics from Color to Depth ---")
        # The rotation is a 3x3 matrix
        rotation_inv = np.asarray(color_to_depth_extrinsics.rotation).reshape(3, 3)
        print("Rotation Matrix:")
        print(rotation_inv)

        print("\nTranslation Vector (meters):")
        # The translation is a 3x1 vector, usually in meters
        translation_inv = np.asarray(color_to_depth_extrinsics.translation)
        print(translation_inv) # Typically in meters [Tx, Ty, Tz]

        # You can combine R_inv and T_inv into a 4x4 homogeneous matrix
        Extrinsic_Matrix_Color_To_Depth = np.eye(4)
        Extrinsic_Matrix_Color_To_Depth[:3, :3] = rotation_inv
        Extrinsic_Matrix_Color_To_Depth[:3, 3] = translation_inv
        print("\nHomogeneous Extrinsic Matrix (Color -> Depth):")
        print(Extrinsic_Matrix_Color_To_Depth)


    except RuntimeError as e:
         print(f"Could not get extrinsics from Color to Depth. Error: {e}")


finally:
    # Stop streaming
    pipeline.stop()
    print("\nStreaming stopped.")