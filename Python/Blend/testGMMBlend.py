import numpy as np
import matplotlib.pyplot as plt
from spatialmath.base import x2r, r2x
import swift
import spatialgeometry as sg
import roboticstoolbox as rtb
from spatialmath import SE3
from Blend import Blend
import time
import math
import mujoco
import os


def fetch_data_from_records(path: str) -> np.ndarray:
    """Fetch the demonstration data from the records

    Args:
        path (str): Path to the records. Example:
                  "./Records/Down_A/**/Record_tcp.txt"
        skip_size (int, optional): Skip every n-th data point. Defaults to 5.

    Returns:
        np.ndarray: Data from the records. Shape: (n_demonstrations, n_steps, n_degrees_of_freedom)
    """
    from glob import glob
    import numpy as np

    # Iterate the files
    # Open the file and read the data line by line
    with open(path, "r", encoding="utf-8") as f:
        # Read the data
        data: list = f.readlines()
        
        # Iterate the data
        for i in range(len(data)):
            # Split the data and convert to float                
            data[i] = data[i].replace('[', '').replace(']', '')
            data[i] = data[i].split()
            data[i] = [float(x) for x in data[i]]
            
            if len(data[i]) != 6:
                print(f"Error: {path}, line: {i}")
                continue
            
            if i > 0:
                for j in range(3, 6):
                    if abs(data[i][j] - data[i-1][j]) > 5:
                        if (data[i][j] - data[i-1][j]) > np.pi:
                            data[i][j] -= 2*np.pi
                        elif (data[i][j] - data[i-1][j]) < -np.pi:
                            data[i][j] += 2*np.pi
    return np.array(data)

def plot(data):
    
    fig_paths = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection="3d")
    
    # Add the measurements
    for i in range(len(data)):
        set: dict = {
            "x": data[i, :, 0],
            "y": data[i, :, 1],
            "z": data[i, :, 2]
        }
        ax.plot3D(set["x"], set["y"], set["z"],
                    color="black", alpha=0.25)

    ax.set_title(f"GMM with {len(data)} demonstrations")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()

def create_4x4_matrix(data):
    # Create a 4x4 identity matrix
    mat = np.eye(4)
    # Set the top left 3x3 submatrix to the rotation matrix
    mat[:3, :3] = x2r(data[3:6], 'eul')
    # Set the rightmost 3x1 column to the translation vector
    mat[:3, 3] = data[0:3]
    return SE3(mat)

def toSE3(arr):
    arrSE3=np.zeros(shape=(len(arr),4,4))
    for i in range(len(arr)):
        arrSE3[i] = create_4x4_matrix(arr[i])
    return arr

def toEuler(arrSE3):
    """
    Convert SE3 object to an array of [x,y,z, ox,oy,oz] (o = orientation)
    """
    arr=np.zeros(shape=(len(arrSE3),6))
    for i in range(len(arrSE3)):
        R_arr = arrSE3[i].R
        xyz= r2x(R_arr, "eul")
        t_arr = arrSE3[i].t
        arr[i] = np.concatenate((t_arr, xyz))
    return arr

def adddotstcp(traj, colorref):
    for tcp in traj:
        mark = sg.Sphere(0.01, pose=SE3(tcp[:3]), color=colorref)
        env.add(mark)

def adddotsjoints(traj, colorref):
    for joint_pos in traj:
        q = UR5.fkine(joint_pos)
        mark = sg.Sphere(0.01, pose=q, color=colorref)
        env.add(mark)
# TCP
#up_b: np.ndarray = fetch_data_from_records("./Python/Blend/Up_B_Tcp.txt")
#up_b: np.ndarray = np.loadtxt("./Records/Up_B/1/record_tcp.txt", delimiter=',', skiprows=0)
#up_b = up_b[::10]
down_b: np.ndarray = fetch_data_from_records("./Python/Blend/Down_B_Tcp.txt")
#down_b: np.ndarray = np.loadtxt("./Records/Down_B/1/record_tcp.txt", delimiter=',', skiprows=0)
#down_b = down_b[::10]
#down_a: np.ndarray = fetch_data_from_records("./Python/Blend/Down_A_Tcp.txt")
up_a: np.ndarray = fetch_data_from_records("./Python/Blend/Up_A_Tcp.txt")
#up_a: np.ndarray = np.loadtxt("./Records/Up_A/1/record_tcp.txt", delimiter=',', skiprows=0)
#up_a = up_a[::10]
# Joints
up_b_j: np.ndarray = fetch_data_from_records("./Python/Blend/Up_B_j.txt")
#up_b_j: np.ndarray = np.loadtxt("./Records/Up_B/1/record_j.txt", delimiter=',', skiprows=0)
down_b_j: np.ndarray = fetch_data_from_records("./Python/Blend/Down_B_j.txt")
#down_a_j: np.ndarray = fetch_data_from_records("./Python/Blend/Down_A_j.txt")
up_a_j: np.ndarray = fetch_data_from_records("./Python/Blend/Up_A_j.txt")


box = sg.Cuboid([2,2,-0.1], pose=SE3(0,0,0))
UR5 = rtb.models.UR5()
blendClass = Blend(UR5=UR5, box=box)
q0 =  [-np.pi / 2, -np.pi / 2, 0, -np.pi / 2, np.pi / 2, 0]
UR5.q = q0

swiftEnv = False
if swiftEnv:
    env = swift.Swift()
    env.launch(realtime=True)
    # Create an obstacles
    #env.add(box)
    # Add robot to env
    env.add(UR5)
    # Move the robot to the start position
    env.step()


model = mujoco.MjModel.from_xml_path("./Example_exercise/universal_robots_ur5e/scene.xml")
data = mujoco.MjData(model)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam.fixedcamid = 0
scene = mujoco.MjvScene(model, maxgeom=10000)
run = True
exit(1)

# Connection paths
#connectionTraj = traj.makeTraj(down_b_j[-1], up_a_j[0])
#returnToStart = traj.makeTraj(down_a[-1], up_b[0])

######################## TODO ########################
# 1. fix python intellisense somehow
# 2. Find out how to combine the blends new paths that are created
# 3. Add rotation
# 4. Try with joint angles
# 5. Simulate with Swift

startpoint = create_4x4_matrix(down_b[-1])#SE3(down_b[-1,:3])
endpoint = create_4x4_matrix(up_a[0])#SE3(up_a[0,:3])
mark = sg.Sphere(0.01, pose=startpoint, color=(0,1,0))
#env.add(mark)
mark = sg.Sphere(0.01, pose=endpoint, color=(1,0,0))
#env.add(mark)
connectionTraj = blendClass.makeTraj(startpoint, endpoint)
connectionTraj = toEuler(connectionTraj)


via = np.asarray([0,40,0,40,0])
dur = np.asarray([10,10,10,10])
tb = np.asarray([1,1,1,1,1])*1.5
res=blendClass.lspb(via,dur,tb)
plt.plot(res[2],via,'*',res[3],res[4])#,'.')




# Downsample to 50 steps
down_b = down_b[::3]

blendedPath1 = blendClass.blendTraj(down_b, connectionTraj, 20, bsize1=10, bsize2=10, plot=True)
exit(1)
blendedPath2 = blendClass.blendTraj(blendedPath1, up_a, 20, bsize1=10, bsize2=20, plot=True)


#adddotstcp(blendedPath1[:,:3], (0,0,1))
adddotstcp(blendedPath2[:,:3], (1,0,0))

