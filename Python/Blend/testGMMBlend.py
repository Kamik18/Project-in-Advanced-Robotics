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
import threading
import math


RUNNING: bool = True
IP:str = "192.168.1.131"

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
        env.step(0.1)

def getData(method: str = "") -> tuple:
    """Get the data from the records

    Args:
        method (str): Method to use for fetching the data. Options: "", "DMP", "GMM"

    Returns:
        tuple: Tuple containing the data for the four paths
    """
    if method == "DMP":
        up_b_j = np.loadtxt("Python/DMP/Out/DMP_Joint_UP_B_smoothing.txt", delimiter=",")[::5]
        down_b_j = np.loadtxt("Python/DMP/Out/DMP_Joint_DOWN_B_smoothing.txt", delimiter=",")[::3]
        up_a_j = np.loadtxt("Python/DMP/Out/DMP_Joint_Up_A_smoothing.txt", delimiter=",")[::10]
        down_a_j = np.loadtxt("Python/DMP/Out/DMP_Joint_DOWN_A_smoothing.txt", delimiter=",")[::5]        
    elif method == "GMM":
        data: np.ndarray = GMM.fetch_data_from_records(path="Records/Up_B/**/record_j.txt", skip_size=50)
        GMM_translation: GMM = GMM.GMM(data=data, n_components=8)
        up_b_j, cov_up_b = GMM_translation.get_path()

        data: np.ndarray = GMM.fetch_data_from_records(path="Records/Down_B/**/record_j.txt", skip_size=50)
        GMM_translation: GMM = GMM.GMM(data=data, n_components=8)
        down_b_j, cov_down_b = GMM_translation.get_path()

        data: np.ndarray = GMM.fetch_data_from_records(path="Records/Up_A/**/record_j.txt", skip_size=50)
        GMM_translation: GMM = GMM.GMM(data=data, n_components=8)
        up_a_j, cov_up_a = GMM_translation.get_path()

        data: np.ndarray = GMM.fetch_data_from_records(path="Records/Down_A/**/record_j.txt", skip_size=50)
        GMM_translation: GMM = GMM.GMM(data=data, n_components=8)
        down_a_j, cov_down_a = GMM_translation.get_path()
        return up_b_j, down_b_j, up_a_j, down_a_j, cov_up_b, cov_down_b, cov_up_a, cov_down_a
    else:
        # Original
        up_b_j: np.ndarray = np.loadtxt("./Records/Up_B/1/record_j.txt", delimiter=',', skiprows=0)[::50]
        down_b_j: np.ndarray = np.loadtxt("./Records/Down_B/1/record_j.txt", delimiter=',', skiprows=0)[::50]
        up_a_j: np.ndarray = np.loadtxt("./Records/Up_A/1/record_j.txt", delimiter=',', skiprows=0)[::50]
        down_a_j: np.ndarray = np.loadtxt("./Records/Down_A/1/record_j.txt", delimiter=',', skiprows=0)[::50]
    return up_b_j, down_b_j, up_a_j, down_a_j

def blendPath(plot: bool = False):
    # Connection paths
    home_to_start = blendClass.makeTraj(q0, up_b_j[0])
    return_to_start = blendClass.makeTraj(down_b_j[-1], q0)

    move_to_pickup = blendClass.blendJointTraj(home_to_start.q, up_b_j, 5, plot=False)
    blendedPath2 = blendClass.blendJointTraj2(down_b_j, up_a_j, 
                                            np.array([down_b_j[-20], down_b_j[-1], up_a_j[0], up_a_j[20]]),
                                            5, plot=False)
    blendedPath3 = blendClass.blendJointTraj2(down_a_j, up_b_j, 
                                            np.array([down_a_j[-20], down_a_j[-1], up_b_j[0],up_b_j[20]]), 
                                            5, plot=False)
    return_to_home = blendClass.blendJointTraj(down_b_j, return_to_start.q, 5, plot=False)

    def remove_near_points(points:np.ndarray, start:int = 10, end:int = 10) -> np.ndarray:
        if len(points) < (start + end):
            return points

        MIN_DIST = 0.01
        temp_points:list = []
        for i in range(start):
            temp_points.append(points[i])

        skipped:int = 0
        for i in range(start, len(points) - end):
            # discard point if distance to previous is too close
            if (np.linalg.norm(temp_points[-1] - points[i]) > MIN_DIST) or (skipped > 5):
                temp_points.append(points[i])
                skipped = 0
            else:
                skipped += 1

        # Append the last 
        for i in range(len(points) - end, len(points)):
            temp_points.append(points[i])
        return np.array(temp_points)

    move_to_pickup = remove_near_points(move_to_pickup, start=1)
    blendedPath2 = remove_near_points(blendedPath2)
    blendedPath3 = remove_near_points(blendedPath3, start=30)
    return_to_home = remove_near_points(return_to_home, end=1)

    move_insert_return = np.concatenate([blendedPath2, blendedPath3])

    if plot:
        adddotsjoints(move_to_pickup,(1,0,0))
        adddotsjoints(move_insert_return,(0,1,0))
        adddotsjoints(return_to_home,(1,0,0))
    
    return move_to_pickup, move_insert_return, return_to_home

def oriPath(plot : bool = False):
    home_to_start = blendClass.makeTraj(q0, up_b_j[0])
    return_to_start = blendClass.makeTraj(down_b_j[-1], q0)
    connection_b_a = blendClass.makeTraj(down_b_j[-1], up_a_j[0])
    connection_a_b = blendClass.makeTraj(down_a_j[-1], up_b_j[0])
    # Setup move trajectories
    move_to_pickup = np.concatenate([home_to_start.q, up_b_j])
    move_insert_return = np.concatenate([down_b_j, connection_b_a.q, up_a_j, down_a_j, connection_a_b.q, up_b_j])
    return_to_home = np.concatenate([down_b_j, return_to_start.q])

    if plot:
        adddotsjoints(move_to_pickup,(0,0,1))
        adddotsjoints(move_insert_return,(0,1,0))
        adddotsjoints(return_to_home,(1,0,0))
    
    return move_to_pickup, move_insert_return, return_to_home

def log():
    global RUNNING

    rtde_r = RTDEReceive(IP)

    while RUNNING:
        start_time = time.time()

        # Read joint acceleration, velocity
        acc = rtde_r.getTargetQdd()
        vel = rtde_r.getActualQd()

        # Append to file
        with open("./Records/experiments/acc.txt", "a") as f:
            f.write(f"{acc}\n")
        with open("./Records/experiments/vel.txt", "a") as f:
            f.write(f"{vel}\n")
        
        # Wait for timestep
        delta_time = time.time() - start_time
        if delta_time < 0.02:
            time.sleep(0.02 - delta_time)
        else:
            print(f"Warning: delta_time: {delta_time}")
            exit(1)

def runRobot(speed,move_to_pickup, move_insert_return, return_to_home):
    #run robot
    rtde_c = RTDEControl(IP)
    print("Starting RTDE test script...")
    VELOCITY = 0.5
    ACCELERATION = 0.2
    BLEND = 0

    rtde_c.moveJ(q0, VELOCITY, ACCELERATION, False)
    rtde_c.speedStop()
    time.sleep(1)

    gripper = RobotiqGripper()
    gripper.connect(IP, 63352)
    gripper.activate()
    gripper.move_and_wait_for_pos(position=0, speed=5, force=25)

    # Thread for force plot
    log_thread = threading.Thread(target=log)
    log_thread.start()

    # Move asynchronously in joint space to new_q, we specify asynchronous behavior by setting the async parameter to
    # 'True'. Try to set the async parameter to 'False' to observe a default synchronous movement, which cannot be stopped
    # by the stopJ function due to the blocking behaviour.



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
    RUNNING = False
    time.sleep(0.2)
    print("Stopped movement")
    rtde_c.stopScript()
    rtde_c.disconnect()


# Standard environment
box = Cuboid([1,1,-0.10], base=SE3(0.30,0.34,-0.05), color=[0,0,1])
UR5 = rtb.models.UR5()
blendClass = Blend(UR5=UR5, box=box)
q0 =  np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2])
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

up_b_j, down_b_j, up_a_j, down_a_j = getData("DMP")
#up_b_j, down_b_j, up_a_j, down_a_j, cov_up_b, cov_down_b, cov_up_a, cov_down_a = getData("GMM")
# up_b_j, down_b_j, up_a_j, down_a_j = getData()       

#move_to_pickup, move_insert_return, return_to_home = oriPath(swiftEnv)
move_to_pickup, move_insert_return, return_to_home = blendPath(swiftEnv)


speed = 0.1
#runRobot(speed, move_to_pickup, move_insert_return, return_to_home)

exit(1)
def create_outer_ellipsoid(tcp, xr, yr, zr, num_points=1000):
    """
    Create an ellipsoid with specified center and radii and return only the outer points.
    
    Args:
    - xc, yc, zc: coordinates of the center of the ellipsoid
    - xr, yr, zr: radii of the ellipsoid along the x, y, and z axes
    - num_points: number of points to generate (default: 100)
    
    Returns:
    - ellipsoid_points: a (N, 3) array of x, y, and z coordinates of the outer ellipsoid points
    """
    # Generate random points on a sphere
    u = np.random.rand(num_points)
    v = np.random.rand(num_points)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    
    # Convert spherical to Cartesian coordinates
    xc, yc, zc = tcp.t[0], tcp.t[1], tcp.t[2]
    x = xr * np.sin(phi) * np.cos(theta) + xc
    y = yr * np.sin(phi) * np.sin(theta) + yc
    z = zr * np.cos(phi) + zc

    # Append the center point
    x = np.append(x, xc)
    y = np.append(y, yc)
    z = np.append(z, zc)
    
    # Calculate distance of each point from the center
    d = np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2)
        
    return np.column_stack((x, y, z))
    
# Create sphere around points based on covariance
tcp = UR5.fkine(up_b_j[0])

index: int = 15
print(len(down_b_j))
print(len(cov_down_b))
print(down_b_j[index])
print(cov_down_b[index])

xr, yr, zr = 4, 3, 2
ellipsoid_points = create_outer_ellipsoid(tcp, xr, yr, zr)
# Print number of outer points
print("Number of outer points:", len(ellipsoid_points))




# plot tcp in 3D matplotlib plot   
fig_paths = plt.figure(figsize=(10, 5))
ax = fig_paths.add_subplot(111, projection='3d')
#plot center point
ax.scatter(tcp.t[0], tcp.t[1], tcp.t[2],c='r',marker='o')
ax.scatter(ellipsoid_points[:, 0], ellipsoid_points[:, 1], ellipsoid_points[:, 2], c='b', marker='.')
"""
# Plot ellipsoid points
ax.scatter([p[0] for p in ellipsoid_points], 
           [p[1] for p in ellipsoid_points], 
           [p[2] for p in ellipsoid_points], 
           c='b', marker='.')
"""
ax.set_xlim([tcp.t[0]-xr-1, tcp.t[0]+xr+1])
ax.set_ylim([tcp.t[1]-yr-1, tcp.t[1]+yr+1])
ax.set_zlim([tcp.t[2]-zr-1, tcp.t[2]+zr+1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


