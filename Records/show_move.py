import roboticstoolbox as rtb
import swift
from spatialmath import SE3
from spatialmath.base import *
import spatialgeometry as sg
from cmath import pi, sqrt
from glob import glob
import os
import numpy as np

def create_4x4_matrix(data):
    # Create a 4x4 identity matrix
    mat = np.eye(4)

    # Set the top left 3x3 submatrix to the rotation matrix
    mat[:3, :3] = x2r(data[3:6])

    # Set the rightmost 3x1 column to the translation vector
    mat[:3, 3] = data[0:3]
    
    return SE3(mat)

# init environtment
env = swift.Swift()
env.launch(realtime=True)

# Create an obstacles
box = sg.Cuboid([2,2,-0.1], pose=SE3(0,0,0))
env.add(box)

# load robot 
UR5 = rtb.models.UR5()
env.add(UR5)

os.path.abspath(os.getcwd())
# load path from txt file
tcp_file = glob("./Records/Pick_A_1/Record_tcp.txt")
with open(tcp_file[0]) as f:
    data = f.readlines()

    for i in range(len(data)):
        data[i] = data[i].split(',')
        data[i] = [float(j) for j in data[i]]


UR5.q = UR5.ik_lm_wampler(create_4x4_matrix(data[0]), ilimit=500, slimit=500, q0=[-np.pi / 2, -np.pi / 2, 0, -np.pi / 2, np.pi / 2, 0])[0]
env.step()

for i in range(len(data)):
    UR5.q = UR5.ik_lm_wampler(create_4x4_matrix(data[i*20]), ilimit=500, slimit=500, q0=UR5.q)[0]
    env.step()


