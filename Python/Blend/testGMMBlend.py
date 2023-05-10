import numpy as np
import matplotlib.pyplot as plt
from spatialmath.base import x2r, r2x
import swift
from spatialgeometry import Sphere, Cuboid
import roboticstoolbox as rtb
from spatialmath import SE3
from Blend import Blend
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import time
from Python.Gripper.RobotiqGripper import RobotiqGripper
import Python.GMM.GMM as GMM

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
        mark = Sphere(0.01, pose=SE3(tcp[:3]), color=colorref)
        env.add(mark)

def adddotsjoints(traj, colorref):
    for joint_pos in traj:
        q = UR5.fkine(joint_pos)
        mark = Sphere(0.01, pose=q, color=colorref)
        env.add(mark)
        UR5.q = joint_pos
        env.step()
# TCP
#up_b: np.ndarray = fetch_data_from_records("./Python/Blend/Up_B_Tcp.txt")
#up_b: np.ndarray = np.loadtxt("./Records/Up_B/1/record_tcp.txt", delimiter=',', skiprows=0)
#up_b = up_b[::10]
#down_b: np.ndarray = fetch_data_from_records("./Python/Blend/Down_B_Tcp.txt")
#down_b: np.ndarray = np.loadtxt("./Records/Down_B/1/record_tcp.txt", delimiter=',', skiprows=0)
#down_b = down_b[::10]
#down_a: np.ndarray = fetch_data_from_records("./Python/Blend/Down_A_Tcp.txt")
#up_a: np.ndarray = fetch_data_from_records("./Python/Blend/Up_A_Tcp.txt")
#up_a: np.ndarray = np.loadtxt("./Records/Up_A/1/record_tcp.txt", delimiter=',', skiprows=0)
#up_a = up_a[::10]
# Joints
# GMM
data: np.ndarray = GMM.fetch_data_from_records(path="./Records/Up_B/**/Record_j.txt", skip_size=10)
GMM_translation: GMM = GMM.GMM(data=data, n_components=8)
up_b_j, covariances = GMM_translation.get_path()

data: np.ndarray = GMM.fetch_data_from_records(path="./Records/Down_B/**/Record_j.txt", skip_size=10)
GMM_translation: GMM = GMM.GMM(data=data, n_components=8)
down_b_j, covariances = GMM_translation.get_path()

data: np.ndarray = GMM.fetch_data_from_records(path="./Records/Up_A/**/Record_j.txt", skip_size=10)
GMM_translation: GMM = GMM.GMM(data=data, n_components=8)
up_a_j, covariances = GMM_translation.get_path()

data: np.ndarray = GMM.fetch_data_from_records(path="./Records/Down_A/**/Record_j.txt", skip_size=10)
GMM_translation: GMM = GMM.GMM(data=data, n_components=8)
down_a_j, covariances = GMM_translation.get_path()

# Original
up_b_j: np.ndarray = np.loadtxt("./Records/Up_B/1/record_j.txt", delimiter=',', skiprows=0)
down_b_j: np.ndarray = np.loadtxt("./Records/Down_B/1/record_j.txt", delimiter=',', skiprows=0)
up_a_j: np.ndarray = np.loadtxt("./Records/Up_A/1/record_j.txt", delimiter=',', skiprows=0)
down_a_j: np.ndarray = np.loadtxt("./Records/Down_A/1/record_j.txt", delimiter=',', skiprows=0)

#up_b_v: np.ndarray = np.loadtxt("./Records/Up_B/1/velocity.txt", delimiter=',', skiprows=0)
#down_b_v: np.ndarray = np.loadtxt("./Records/Down_B/1/velocity.txt", delimiter=',', skiprows=0)
#up_a_v: np.ndarray = np.loadtxt("./Records/Up_A/1/velocity.txt", delimiter=',', skiprows=0)
#down_a_v: np.ndarray = np.loadtxt("./Records/Down_A/1/velocity.txt", delimiter=',', skiprows=0)

# Down sample
up_b_j = up_b_j[::50]
down_b_j = down_b_j[::50]
up_a_j = up_a_j[::50]
down_a_j = down_a_j[::50]

box = Cuboid([1,1,-0.10], base=SE3(0.30,0.34,-0.05), color=[0,0,1])
UR5 = rtb.models.UR5()
blendClass = Blend(UR5=UR5, box=box)
q0 =  np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
UR5.q = q0

swiftEnv = True
if swiftEnv:
    env = swift.Swift()
    env.launch(realtime=True)
    # Create an obstacles
    env.add(box)
    # Add robot to env
    env.add(UR5)
    # Move the robot to the start position
    env.step()

# Connection paths
home_to_start = blendClass.makeTraj(q0, up_b_j[0])
connection_b_a = blendClass.makeTraj(down_b_j[-1], up_a_j[0])
connection_a_b = blendClass.makeTraj(down_a_j[-1], up_b_j[0])
return_to_start = blendClass.makeTraj(down_b_j[-1], q0)


blendedPath1 = blendClass.blendJointTraj(home_to_start.q, up_b_j, 20, bsize1=10, bsize2=10, plot=True)

adddotsjoints(blendedPath1,(1,0,0))
exit(1)


#combined_traj = np.concatenate([home_to_start.q, up_b_j, down_b_j, connectionTraj.q, return_to_start.q])
#combined_traj = np.concatenate([home_to_start.q, up_b_j, down_b_j, connectionTraj.q, up_a_j, down_a_j, return_to_start.q])
#combined_traj_v = np.concatenate([home_to_start.qd, up_b_v, down_b_v, connectionTraj.qd, up_a_v, down_a_v, return_to_start.qd])

# Setup move trajectories
move_to_pickup = np.concatenate([home_to_start.q, up_b_j])
move_insert_return = np.concatenate([down_b_j, connection_b_a.q, up_a_j, down_a_j, connection_a_b.q, up_b_j])
return_to_home = np.concatenate([down_b_j, return_to_start.q])

#adddotsjoints(move_to_pickup,(0,0,1))
#adddotsjoints(move_insert_return,(0,1,0))
#adddotsjoints(return_to_home,(1,0,0))

#adddotsjoints(home_to_start.q,(0,0,1))
#adddotsjoints(up_b_j,(0,1,0))
#adddotsjoints(down_b_j,(1,0,0))
#adddotsjoints(connectionTraj.q,(1,0,1))
#adddotsjoints(up_a_j,(0,1,1))
#adddotsjoints(down_a_j,(1,1,1))
#adddotsjoints(return_to_start.q,(1,1,0))
#adddotsjoints(combined_traj,(1,0,0))
exit(1)
#run robot
IP = "192.168.1.131"
rtde_c = RTDEControl(IP)
rtde_r = RTDEReceive(IP)

gripper = RobotiqGripper()
gripper.connect(IP, 63352)
gripper.activate()
gripper.move_and_wait_for_pos(position=0, speed=5, force=25)

print("Starting RTDE test script...")
# Target in the robot base
VELOCITY = 0.5
ACCELERATION = 0.2
BLEND = 0

# Move asynchronously in joint space to new_q, we specify asynchronous behavior by setting the async parameter to
# 'True'. Try to set the async parameter to 'False' to observe a default synchronous movement, which cannot be stopped
# by the stopJ function due to the blocking behaviour.
rtde_c.moveJ(q0, VELOCITY, ACCELERATION, False)
rtde_c.speedStop()
time.sleep(1)

speed = 0.1
for joint in move_to_pickup:
    rtde_c.servoJ(joint, 0,0, speed, 0.2, 100)
    time.sleep(speed)

# Grip the object
gripper.move_and_wait_for_pos(position=255, speed=5, force=25)
time.sleep(1.0)

for joint in move_insert_return:
    rtde_c.servoJ(joint, 0,0, speed, 0.2, 100)
    time.sleep(speed)
    
# release the object
gripper.move_and_wait_for_pos(position=0, speed=5, force=25)
time.sleep(1.0)

for joint in return_to_home:
    rtde_c.servoJ(joint, 0,0, speed, 0.2, 100)
    time.sleep(speed)

rtde_c.speedStop()
time.sleep(0.2)
print("Stopped movement")
rtde_c.stopScript()
rtde_c.disconnect()
exit(1)

######################## TODO ########################
# 1. fix python intellisense somehow
# 2. Find out how to combine the blends new paths that are created
# 4. Try with joint angles
# 5. Simulate with Swift

startpoint = create_4x4_matrix(down_b[-1])#SE3(down_b[-1,:3])
endpoint = create_4x4_matrix(up_a[0])#SE3(up_a[0,:3])
mark = Sphere(0.01, pose=startpoint, color=(0,1,0))
#env.add(mark)
mark = Sphere(0.01, pose=endpoint, color=(1,0,0))
#env.add(mark)
connectionTraj = blendClass.makeTraj(startpoint, endpoint)
connectionTraj = toEuler(connectionTraj)


# Downsample to 50 steps
#down_b = down_b[::3]

#blendedPath1 = blendClass.blendTraj(down_b, connectionTraj, 20, bsize1=10, bsize2=10, plot=True)
exit(1)
blendedPath2 = blendClass.blendTraj(blendedPath1, up_a, 20, bsize1=10, bsize2=20, plot=True)


#adddotstcp(blendedPath1[:,:3], (0,0,1))
adddotstcp(blendedPath2[:,:3], (1,0,0))

