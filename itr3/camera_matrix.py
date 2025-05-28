import cv2
import numpy as np
import glob

# Checkerboard config
cb_size = (9, 6)  # number of inner corners
objp = np.zeros((np.prod(cb_size), 3), np.float32)
objp[:, :2] = np.indices(cb_size).T.reshape(-1, 2)
objp *= 0.0235  # square size in meters

objpoints = []
imgpoints = []

images = glob.glob('d:\codes\camera_calib_IIITA\itr3\cam_0\intrinsic/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, cb_size)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# Calibrate
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera Matrix:\n", K)
path = "d:\codes\camera_calib_IIITA\itr3\cam_0\intrinsics.npy"
path2 = "d:\codes\camera_calib_IIITA\itr3\cam_0\distortion.npy"
np.save(path,K)
print("Distortion Coefficients:\n", dist)
