import numpy as np
import os

output_path = r"D:\codes\camera_calib_IIITA\itr2\cam_2\intrinsics.npy"
default_K = np.array([
    [380 ,0.0, 320],
    [0.0, 380, 240],
    [0.0, 0.0, 1.0],
], dtype=np.float32)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
np.save(output_path, default_K)

print(f"Matrix saved to {output_path}")
print(default_K)
