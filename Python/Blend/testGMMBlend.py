import numpy as np
import matplotlib.pyplot as plt
from spatialmath.base import *
import swift
import spatialgeometry as sg
import roboticstoolbox as rtb
from spatialmath import SE3
from Blend import Trajectory
from roboticstoolbox.tools import trajectory




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

    # Add the GMM path
    #ax.plot3D(path_gmm["x"], path_gmm["y"],
    #            path_gmm["z"], color="red", alpha=1.0, label="GMM")
    # Mean path
    #ax.plot3D(path_mean["x"], path_mean["y"],
    #            path_mean["z"], color="green", alpha=1.0, label="Mean")
    ax.set_title(f"GMM with {len(data)} demonstrations")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()

def create_4x4_matrix(data):
    # Create a 4x4 identity matrix
    mat = np.eye(4)
    # Set the top left 3x3 submatrix to the rotation matrix
    mat[:3, :3] = x2r(data[3:6])
    # Set the rightmost 3x1 column to the translation vector
    mat[:3, 3] = data[0:3]
    return SE3(mat)

def adddots(traj, colorref):
    for joint_pos in traj:
        q = UR5.fkine(joint_pos)
        mark = sg.Sphere(0.01, pose=q, color=colorref)
        env.add(mark)

# TCP
#up_b: np.ndarray = fetch_data_from_records("./Python/Blend/Up_B_Tcp.txt")
#down_b: np.ndarray = fetch_data_from_records("./Python/Blend/Down_B_Tcp.txt")
#down_a: np.ndarray = fetch_data_from_records("./Python/Blend/Down_A_Tcp.txt")
#up_a: np.ndarray = fetch_data_from_records("./Python/Blend/Up_A_Tcp.txt")

# Joints
up_b: np.ndarray = fetch_data_from_records("./Python/Blend/Up_B_j.txt")
down_b: np.ndarray = fetch_data_from_records("./Python/Blend/Down_B_j.txt")
down_a: np.ndarray = fetch_data_from_records("./Python/Blend/Down_A_j.txt")
up_a: np.ndarray = fetch_data_from_records("./Python/Blend/Up_A_j.txt")

#Convert to trajectory
#up_b = trajectory.Trajectory('jtraj', 2, up_b)
#down_b = trajectory.Trajectory('jtraj', 2, down_b)
#down_a = trajectory.Trajectory('jtraj', 2, down_a)
#up_a = trajectory.Trajectory('jtraj', 2, up_a)

#plot(up_b)
#plot(down_a)
#plot(joined_path)
#plt.show()


#env = swift.Swift()
#env.launch(realtime=True)

# Create an obstacles
box = sg.Cuboid([2,2,-0.1], pose=SE3(0,0,0))
#env.add(box)

UR5 = rtb.models.UR5()
#env.add(UR5)

# Move the robot to the start position
traj = Trajectory(UR5=UR5)
q0 =  [-np.pi / 2, -np.pi / 2, 0, -np.pi / 2, np.pi / 2, 0]
UR5.q = q0
#env.step()

# Connection paths
connectionTraj = traj.makeTraj(down_b[-1], up_a[0])
returnToStart = traj.makeTraj(down_a[-1], up_b[0])

via = np.asarray([0,50,30,40,0])
print(via)
dur = np.asarray([10,15,20,20])
print(dur)
tb = np.asarray([1,1,1,1,1])*5
print(tb)
res = traj.lspb(via, dur, tb)


plt.plot(res[2],via,'*',res[3],res[4])
plt.show()

#t = np.arange(0,10)
#x = traj.traj_poly(0,1,0,0,0,0,t)
#y = traj.traj_poly(0,1,0,0,0,0,t)
#z = traj.traj_poly(0,1,0,0,0,0,t)

#xyz = np.stack([x,y,z])
#print('xyz shape', xyz.shape)
#xyz = np.squeeze(xyz)
#print('xyz shape after squeeze', xyz.shape)


#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.plot(x[0,0,:],'r')
#plt.show()
#ax.plot(x[0,0,:], x[1,0,:], x[2,0,:], 'r', label='x')  # plot x
#ax.plot(y[0,0,:], y[1,0,:], y[2,0,:], 'g', label='y')  # plot y
#ax.plot(z[0,0,:], z[1,0,:], z[2,0,:], 'g', label='z')  # plot z

#ax.plot(x[0,0,:],'r',x[1,0,:],'g',x[2,0,:],'b', label='x')
#ax.plot(t, y[0,0,:],'r',y[1,0,:],'g',y[2,0,:],'b')
#ax.plot(x[0, 0, :] + y[0, 0, :], x[1, 0, :] + y[1, 0, :], x[2, 0, :] + y[2, 0, :], 'b', label='x+y')

#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.legend()

#plt.show()


exit(1)
#blendtesttraj = traj.blendTwoPointsTraj(connectionTraj.q, up_a, 1)
blendtesttraj = traj.blendTwoPointsTraj(down_a, returnToStart.q,1)

adddots(returnToStart.q, (0,1,0)) 
adddots(down_a, (1,0,0)) # Starts up and moves down
for joint_pos in blendtesttraj:
    UR5.q = joint_pos
    q = UR5.fkine(joint_pos)
    mark = sg.Sphere(0.01, pose=q, color=(0,0,1))
    env.add(mark)
    env.step()

env.hold()

"""
joined_path = np.concatenate([up_b, down_b, connectionTraj.q, up_a, down_a, returnToStart.q])

for joint_pos in joined_path:#joined_path:
    UR5.q = joint_pos#traj.inverse_kinematics(create_4x4_matrix(temp), q0)#UR5.ik_lm_wampler(create_4x4_matrix(data[i*20]), ilimit=500, slimit=500, q0=UR5.q)[0]
    q = UR5.fkine(joint_pos)
    mark = sg.Sphere(0.01, pose=q, color=(0,0,1))
    env.add(mark)
    env.step()
"""


#adddots(up_b, (1,0,0)) # Starts up and moves down
#adddots(down_b, (0,0,1)) # Starts down and moves up
#adddots(up_a, (0,0,0)) # Starts up and moves down
#adddots(down_a, (0,1,0)) # Starts down and moves up
#env.hold()