import numpy as np
import matplotlib.pyplot as plt
from spatialmath.base import *
import swift
import spatialgeometry as sg
import roboticstoolbox as rtb
from spatialmath import SE3
from Blend import Trajectory




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

    # Fetch the files
    files: list = glob(pathname=path, recursive=True)

    # List of demonstrations
    demonstrations: list = []

    # Iterate the files
    for file in files:
        # Open the file and read the data line by line
        with open(file, "r", encoding="utf-8") as f:
            # Read the data
            data: list = f.readlines()

            # Skip every n-th data point
            max_delta: float = 0

            # Iterate the data
            for i in range(len(data)):
                # Split the data and convert to float                
                data[i] = data[i].replace('[', '').replace(']', '')
                data[i] = data[i].split()
                data[i] = [float(x) for x in data[i]]
                
                if len(data[i]) != 6:
                    print(f"Error: {file}, line: {i}")
                    continue
                
                if i > 0:
                    for j in range(3, 6):
                        if abs(data[i][j] - data[i-1][j]) > 5:
                            if (data[i][j] - data[i-1][j]) > np.pi:
                                data[i][j] -= 2*np.pi
                            elif (data[i][j] - data[i-1][j]) < -np.pi:
                                data[i][j] += 2*np.pi

                        delta = abs(data[i][j] - data[i-1][j])
                        if delta > max_delta:
                            max_delta = delta

            if max_delta > np.pi:
                print(f"Error: {file}, max delta: {max_delta}")
                continue

            # Append the data to a list for all the demonstrations
            demonstrations.append(data)

    # Copy the data from the demonstrations to ndarray
    X: np.ndarray = np.empty((len(demonstrations), len(
        demonstrations[0]), len(demonstrations[0][0])))
    for i in range(len(demonstrations)):
        for j in range(len(demonstrations[i])):
            for k in range(len(demonstrations[i][j])):
                X[i, j, k] = demonstrations[i][j][k]

    # Return the data
    return np.squeeze(X)

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

up_b: np.ndarray = fetch_data_from_records("./Python/Blend/Up_B_Tcp.txt")
down_b: np.ndarray = fetch_data_from_records("./Python/Blend/Down_B_Tcp.txt")
down_a: np.ndarray = fetch_data_from_records("./Python/Blend/Down_A_Tcp.txt")
up_a: np.ndarray = fetch_data_from_records("./Python/Blend/Up_A_Tcp.txt")
joined_path = np.concatenate([up_b, down_a])

#plot(up_b)
#plot(down_a)
#plot(joined_path)
#plt.show()


env = swift.Swift()
env.launch(realtime=True)

# Create an obstacles
box = sg.Cuboid([2,2,-0.1], pose=SE3(0,0,0))
env.add(box)

UR5 = rtb.models.UR5()
env.add(UR5)

# Move the robot to the start position
traj = Trajectory(UR5=UR5, box=box)
q0 =  [-np.pi / 2, -np.pi / 2, 0, -np.pi / 2, np.pi / 2, 0]
UR5.q = q0
env.step()
for temp in up_b:
    temp2 = create_4x4_matrix(temp), q0
    UR5.q = traj.inverse_kinematics(create_4x4_matrix(temp), q0)#UR5.ik_lm_wampler(create_4x4_matrix(data[i*20]), ilimit=500, slimit=500, q0=UR5.q)[0]
    env.step()
