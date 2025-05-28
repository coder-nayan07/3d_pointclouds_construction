from doggy import Doggy
import threading
import time 
shiro = Doggy()
import numpy as np
from math import pi,sin,cos,radians
from numpy import interp
# from pygame_visual import *

STEP_TIME = 150 #ms
REST = [0.6,12.5,-4.97,-2.86,-14.02,3.8,-1.19,-14.9,3.5,1.1,11.9,-6.55]

shiro.connect()
shiro.enable_torque()
shiro.pid(1900,4000,50)#new pid values
i = 300
shiro.set_time(2000)
shiro.move(shiro.move_xyz([0,0,i,0,0,i,0,0,i,0,0,i]))

def traj(t):
    if t > 4:
        t = t%4
    X = 0
    if t <= 3:
        Z = 0
        Y = interp(t,[0,3],[50,-50])
    if t > 3 and t <= 4:
        theta = interp(t,[3,4],[0,pi])
        Y = -50*cos(theta)
        Z = -50*sin(theta)
    return X,Y,Z
def rotation(r,p,y):
    r_x1,r_y1,r_z1 = 0,0,0
    r_x2,r_y2,r_z2 = 0,0,0
    r_x3,r_y3,r_z3 = 0,0,0
    r_x4,r_y4,r_z4 = 0,0,0
    global offset_z
    b = 321.5
    r,p,y = radians(r),radians(p),radians(y)
    # pitch
    p = interp(p,[-pi/2,pi/2],[-350,350])
    r_z1 = -p
    r_z2 = -p
    r_z3 = p
    r_z4 = p

    # roll
    h = offset_z - b*sin(r)/2
    H = offset_z + b*sin(r)/2

    r_x1 = -h*sin(r)
    r_x2 = -H*sin(r)
    r_x3 = -h*sin(r)
    r_x4 = -H*sin(r)

    r_z1 += h*cos(r) - offset_z
    r_z2 += H*cos(r) - offset_z
    r_z3 += h*cos(r) - offset_z
    r_z4 += H*cos(r) - offset_z

    # yaw
    # Use only when X = 0 | Y = 0
    theta = 1.079933399
    R = 341
    X = 341*cos(1.079933399+y) - (321.5/2)
    Y = 341*sin(1.079933399+y) - (601.5/2)
    r_x2 += X
    r_y2 += Y
    r_z2 += 0
    r_x3 += -X
    r_y3 += -Y
    r_z3 += 0
    X = 341*cos(1.079933399-y) - (321.5/2)
    Y = 341*sin(1.079933399-y) - (601.5/2)
    r_x1 += -X
    r_y1 += Y
    r_z1 += 0
    r_x4 += X
    r_y4 += -Y
    r_z4 += 0
    return r_x1,r_y1,r_z1,r_x2,r_y2,r_z2,r_x3,r_y3,r_z3,r_x4,r_y4,r_z4

offset_x = 0
offset_y = 0
offset_z = 300
x = 0
y = 0
z = 0

i = 200
t = 0 
roll = 3
pitch = 1 #value = 2
time_changed = 0
x1,y1,z1 = 0
x2,y2,z2 = 0
x3,y3,z3 = 0
x4,y4,z4 = 0
z1,z2,z3,z4 = 300
shiro.move(shiro.move_xyz([offset_x+x1,offset_y+y1,offset_z+z1,
                            offset_x+x2,offset_y+y2,offset_z+z2,
                            offset_x+x3,-offset_y+y3,offset_z+z3,
                            offset_x+x4,-offset_y+y4,offset_z+z4]))
# time.sleep()
r_x1,r_y1,r_z1,r_x2,r_y2,r_z2,r_x3,r_y3,r_z3,r_x4,r_y4,r_z4 = rotation(+roll,0,0)
shiro.move(shiro.move_xyz([offset_x+x1,offset_y+y1,offset_z+z1,
                            offset_x+x2,offset_y+y2,offset_z+z2,
                            offset_x+x3,-offset_y+y3,offset_z+z3,
                            offset_x+x4,-offset_y+y4,offset_z+z4]))
z2 = 250
shiro.move(shiro.move_xyz([offset_x+x1,offset_y+y1,offset_z+z1,
                            offset_x+x2,offset_y+y2,offset_z+z2,
                            offset_x+x3,-offset_y+y3,offset_z+z3,
                            offset_x+x4,-offset_y+y4,offset_z+z4]))
time.sleep(3)
shiro.set_time(2000)
shiro.move(REST)
print("Done")
shiro.disable_torque()




