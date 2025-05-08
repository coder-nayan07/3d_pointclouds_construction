import pyrealsense2 as rs
import numpy as np
from math import atan,sqrt,pi

def initialize_camera():
    # start the frames pipe
    p = rs.pipeline()
    conf = rs.config()
    conf.enable_stream(rs.stream.accel)
    conf.enable_stream(rs.stream.gyro)
    prof = p.start(conf)
    return p


def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])


def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])


p = initialize_camera()
try:
    while True:
        f = p.wait_for_frames()
        accel = accel_data(f[0].as_motion_frame().get_motion_data())
        gyro = gyro_data(f[1].as_motion_frame().get_motion_data())
        roll = 180*atan(-accel[0]/sqrt(accel[1]**2 + accel[2]**2))/pi
        pitch = 180*atan(accel[2]/sqrt(accel[0]**2 + accel[1]**2))/pi
        # print("pitc: ", pitch)
        print("roll: ", roll)

finally:
    p.stop()