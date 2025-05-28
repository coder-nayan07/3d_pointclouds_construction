import pyrealsense2 as rs
import numpy as np
from math import atan,sqrt,pi
from time import time,time_ns

RateRoll = 0
RatePitch = 0
RateYaw = 0

RateCalibrationRoll = 0
RateCalibrationPitch = 0
RateCalibrationYaw = 0

RateCalibrationNumber = 0
AccX = 0 
AccY = 0 
AccZ = 0

AngleRoll = 0
AnglePitch = 0

bias = 0.03

KalmanAngleRoll=0
KalmanUncertaintyAngleRoll=2*2
KalmanAnglePitch=0
KalmanUncertaintyAnglePitch=2*2
Kalman1DOutput = [0,0]


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

def kalman_1d(KalmanState,  KalmanUncertainty,  KalmanInput,  KalmanMeasurement):
  KalmanState=KalmanState+bias*KalmanInput
  KalmanUncertainty=KalmanUncertainty + bias * bias * 4 * 4
  KalmanGain = KalmanUncertainty * 1/(1*KalmanUncertainty + 3 * 3)
  KalmanState=KalmanState+KalmanGain * (KalmanMeasurement-KalmanState)
  KalmanUncertainty=(1-KalmanGain) * KalmanUncertainty
  Kalman1DOutput[0]=KalmanState
  Kalman1DOutput[1]=KalmanUncertainty

p = initialize_camera()
LoopTimer = time_ns()

try:
    while 1:
        f = p.wait_for_frames()
        accel = accel_data(f[0].as_motion_frame().get_motion_data())
        gyro = gyro_data(f[1].as_motion_frame().get_motion_data())
        roll = 180*atan(accel[0]/sqrt(accel[1]**2 + accel[2]**2))/pi
        pitch = -180*atan(accel[2]/sqrt(accel[0]**2 + accel[1]**2))/pi
        # 0->y
        # 1->z
        # 2->x
        # print("pitc: ", pitch)
        RateRoll=gyro[2]
        RatePitch=gyro[0]
        RateYaw=gyro[1]
        AccX=accel[2]
        AccY=accel[0]
        AccZ=accel[1]
        AngleRoll=roll
        AnglePitch=pitch
        # print("roll: ", roll)

        RateRoll-=RateCalibrationRoll
        RatePitch-=RateCalibrationPitch
        RateYaw-=RateCalibrationYaw
        kalman_1d(KalmanAngleRoll, KalmanUncertaintyAngleRoll, RateRoll, AngleRoll)
        KalmanAngleRoll=Kalman1DOutput[0]
        KalmanUncertaintyAngleRoll=Kalman1DOutput[1]
        kalman_1d(KalmanAnglePitch, KalmanUncertaintyAnglePitch, RatePitch, AnglePitch)
        KalmanAnglePitch=Kalman1DOutput[0]
        KalmanUncertaintyAnglePitch=Kalman1DOutput[1]
        # print(f"Roll Angle [°] {KalmanAngleRoll}")
        print(f" Pitch Angle [°] {KalmanAnglePitch:.2f}")
        while (time_ns() - LoopTimer < 4000000):
            pass
        LoopTimer=time_ns()
finally:
    p.stop()
  