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
import Python.Gripper.RobotiqGripper as RobotiqGripper
import Python.GMM.GMM as GMM
import threading
import Python.DMP.DMP_Global as DMP
import copy


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
        #UR5.q = joint_pos
        #env.step(0.1)

def getData(method: str = "") -> tuple:
    """Get the data from the records

    Args:
        method (str): Method to use for fetching the data. Options: "", "DMP", "GMM"

    Returns:
        tuple: Tuple containing the data for the four paths
    """
    if method == "DMP":
        dmp_spec = DMP.DMP_SPC()
        dmp_data,_ = dmp_spec.maindmp()
        down_a_j, down_b_j,up_a_j, up_b_j = dmp_data['Down_A'], dmp_data['Down_B'], dmp_data['UP_A'], dmp_data['UP_B']
        
        #down_a_j, down_b_j,up_a_j, up_b_j = dmp_spec.read_out_file(skip_lines=5)
        #down_b_j = dmp_spec.read_out_new_pos_file(skip_lines=1,DOWN_B=True, UP_B=False, UP_A=False, DOWN_A=False)
        #up_b_j = dmp_spec.read_out_new_pos_file(skip_lines=5,UP_B=True, DOWN_B=False, UP_A=False, DOWN_A=False)
         
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

def printGMMcov(stored_traj, desired_traj, comb_traj, comb_traj_opti):
    #fig2, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=6, ncols=2,figsize=(6,8))
    #fig2, ((ax0, ax0_c), (ax1, ax1_c), (ax2, ax2_c), (ax3, ax3_c), (ax4, ax4_c), (ax5, ax5_c)) = plt.subplots(nrows=6, ncols=2,figsize=(12,8))    
    print('desired',len(desired_traj))
    print('stored', len(stored_traj))
    print('comb', len(comb_traj))
    print('comb_opti', len(comb_traj_opti))
    x_len_s = np.linspace(0, 4.42, len(stored_traj))
    x_len_c = np.linspace(0, 6.67, len(comb_traj))
    x_len_d = np.linspace(0, 5.22, len(desired_traj))
    x_len_oc = np.linspace(0, 7.47, len(comb_traj_opti))
    fig2, ((ax0, ax0_c, ax0_o), (ax1, ax1_c, ax1_o), (ax2, ax2_c, ax2_o), (ax3, ax3_c, ax3_o), (ax4, ax4_c, ax4_o), (ax5, ax5_c, ax5_o)) = plt.subplots(nrows=6, ncols=3,figsize=(15,8))
    ax0.set_title('GMR optimization')
    ax0_c.set_title('Gap between two trajectories')
    ax0_o.set_title('Optimized gap between two trajectories')
    ax0.scatter(x_len_s,stored_traj[:,0], label='j0_bf', marker='.')
    ax1.scatter(x_len_s,stored_traj[:,1], label='j1_bf', marker='.')
    ax2.scatter(x_len_s,stored_traj[:,2], label='j2_bf', marker='.')
    ax3.scatter(x_len_s,stored_traj[:,3], label='j3_bf', marker='.')
    ax4.scatter(x_len_s,stored_traj[:,4], label='j4_bf', marker='.')
    ax5.scatter(x_len_s,stored_traj[:,5], label='j5_bf', marker='.')

    ax0.scatter(x_len_d,desired_traj[:,0], label='j0_af', marker='.')
    ax1.scatter(x_len_d,desired_traj[:,1], label='j1_af', marker='.')
    ax2.scatter(x_len_d,desired_traj[:,2], label='j2_af', marker='.')
    ax3.scatter(x_len_d,desired_traj[:,3], label='j3_af', marker='.')
    ax4.scatter(x_len_d,desired_traj[:,4], label='j4_af', marker='.')
    ax5.scatter(x_len_d,desired_traj[:,5], label='j5_af', marker='.')
    ax0.legend(); ax1.legend(); ax2.legend(); ax3.legend(); ax4.legend(); ax5.legend()
    
    x_len_c = np.linspace(0, 4.0, len(comb_traj))
    ax0_c.scatter(x_len_c,comb_traj[:,0], label='j0_bf', marker='.')
    ax1_c.scatter(x_len_c,comb_traj[:,1], label='j1_bf', marker='.')
    ax2_c.scatter(x_len_c,comb_traj[:,2], label='j2_bf', marker='.')
    ax3_c.scatter(x_len_c,comb_traj[:,3], label='j3_bf', marker='.')
    ax4_c.scatter(x_len_c,comb_traj[:,4], label='j4_bf', marker='.')
    ax5_c.scatter(x_len_c,comb_traj[:,5], label='j5_bf', marker='.')
    ax0_c.legend(); ax1_c.legend(); ax2_c.legend(); ax3_c.legend(); ax4_c.legend(); ax5_c.legend()

    ax0_o.scatter(x_len_oc,comb_traj_opti[:,0], label='j0_af', marker='.')
    ax1_o.scatter(x_len_oc,comb_traj_opti[:,1], label='j1_af', marker='.')
    ax2_o.scatter(x_len_oc,comb_traj_opti[:,2], label='j2_af', marker='.')
    ax3_o.scatter(x_len_oc,comb_traj_opti[:,3], label='j3_af', marker='.')
    ax4_o.scatter(x_len_oc,comb_traj_opti[:,4], label='j4_af', marker='.')
    ax5_o.scatter(x_len_oc,comb_traj_opti[:,5], label='j5_af', marker='.')
    ax0_o.legend(); ax1_o.legend(); ax2_o.legend(); ax3_o.legend(); ax4_o.legend(); ax5_o.legend()

    # set the legends
    for ax in fig2.get_axes():
        ax.legend(loc='center left', bbox_to_anchor=(0, 0.8))

    fig2.text(0.09, 0.50, 'Angle [rad]', va='center', rotation='vertical')
    fig2.text(0.5, 0.06, 'Time [s]', ha='center')
    plt.show()

def quadprog(point1, point2):
    import cvxpy as cp

    # Calculate the distance between the two points to determine number of waypoints
    distance = np.linalg.norm(point1 - point2)
    # Define the number of waypoints and DOFs
    num_waypoints = int(distance/0.01)
    num_dof = 6

    # Define the waypoints (desired end-effector positions)
    waypoints = np.stack((point1, point2))

    # Define the joint limits
    joint_limits = np.array([
        [2*-np.pi, 2*np.pi],
        [2*-np.pi, 2*np.pi],
        [2*-np.pi, 2*np.pi],
        [2*-np.pi, 2*np.pi],
        [2*-np.pi, 2*np.pi],
        [2*-np.pi, 2*np.pi]
    ])

    # Define the joint variables
    q = cp.Variable((num_dof, num_waypoints))

    # Define the objective function (minimize the sum of squared joint differences)
    obj = cp.Minimize(cp.sum_squares(q[:, 1:] - q[:, :-1]))
    
    # Define the constraints
    constraints = [
        q[:, 0] == np.array(waypoints[0]),  # Starting joint configuration
        q[:, -1] == np.array(waypoints[1]),  # Ending joint configuration
        q >= joint_limits[:, 0][:, np.newaxis],  # Joint lower limits
        q <= joint_limits[:, 1][:, np.newaxis]  # Joint upper limits
    ]
    
    # Define the problem and solve it
    problem = cp.Problem(obj, constraints)
    problem.solve()

    # Retrieve the optimal joint configurations
    joint_configurations = q.value

    # Print the results
    new_joint_configurations = np.zeros((num_waypoints,num_dof))
    for i in range(num_waypoints):
        new_joint_configurations[i,:] = joint_configurations[:, i]
    
    
    return new_joint_configurations

def blendPath(plot: bool = False):
    up_b_j_dmp, down_b_j_dmp, up_a_j_dmp, down_a_j_dmp = getData("DMP")
    up_b_j, down_b_j, up_a_j, down_a_j, cov_up_b, cov_down_b, cov_up_a, cov_down_a = getData("GMM")
    up_b_j, down_b_j, up_a_j, down_a_j = getData()

    stored_up_b = copy.deepcopy(up_b_j)
    stored_down_b = copy.deepcopy(down_b_j)
    stored_up_a = copy.deepcopy(up_a_j)
    stored_down_a = copy.deepcopy(down_a_j)

    try:
        def reduce_dist(prev_point: np.ndarray, curr_point: np.ndarray, cov: np.ndarray) -> np.ndarray:
            cov *= 1.96
            for i in range(len(prev_point)):
                if (prev_point[i] - curr_point[i]) > cov[i]:
                    curr_point[i] += cov[i]
                elif (prev_point[i] - curr_point[i]) < -cov[i]:
                    curr_point[i] -= cov[i]
                else:
                    # Reduce the distance
                    curr_point[i] = (prev_point[i] + curr_point[i]) / 2
                    
            return curr_point
        
        # Optimize up_b_j
        # Flip up_b_j and the covariance
        up_b_j = np.flip(up_b_j, axis=0)
        cov_up_b = np.flip(cov_up_b, axis=0)
        # Reduce the first point in the up_b_j path to the next with the covariance
        up_b_j[0] = reduce_dist(prev_point=down_b_j[0], curr_point=up_b_j[0], cov=cov_up_b[0])
        for i in range(1, len(up_b_j)):
            up_b_j[i] = reduce_dist(prev_point=up_b_j[i-1], curr_point=up_b_j[i], cov=cov_up_b[i])
        # Flip up_b_j and the covariance
        up_b_j = np.flip(up_b_j, axis=0)
        cov_up_b = np.flip(cov_up_b, axis=0)
        
        # Optimize down_b_j
        # Flip down_b_j and the covariance
        down_b_j = np.flip(down_b_j, axis=0)
        cov_down_b = np.flip(cov_down_b, axis=0)
        # Reduce the first point in the down_b_j path to the next with the covariance
        down_b_j[0] = reduce_dist(prev_point=up_a_j[0], curr_point=down_b_j[0], cov=cov_down_b[0])
        for i in range(1, len(down_b_j)):
            down_b_j[i] = reduce_dist(prev_point=down_b_j[i-1], curr_point=down_b_j[i], cov=cov_down_b[i])
        # Flip down_b_j and the covariance
        down_b_j = np.flip(down_b_j, axis=0)
        cov_down_b = np.flip(cov_down_b, axis=0)
        
        # Optimize up_a_j
        # Flip up_a_j and the covariance
        up_a_j = np.flip(up_a_j, axis=0)
        cov_up_a = np.flip(cov_up_a, axis=0)
        # Reduce the first point in the up_a_j path to the next with the covariance
        up_a_j[0] = reduce_dist(prev_point=down_a_j[0], curr_point=up_a_j[0], cov=cov_up_a[0])
        for i in range(1, len(up_a_j)):
            up_a_j[i] = reduce_dist(prev_point=up_a_j[i-1], curr_point=up_a_j[i], cov=cov_up_a[i])
        # Flip up_a_j and the covariance
        up_a_j = np.flip(up_a_j, axis=0)
        cov_up_a = np.flip(cov_up_a, axis=0)
        
        # Optimize down_a_j
        # Flip down_a_j and the covariance
        down_a_j = np.flip(down_a_j, axis=0)
        cov_down_a = np.flip(cov_down_a, axis=0)
        # Reduce the first point in the down_a_j path to the next with the covariance
        down_a_j[0] = reduce_dist(prev_point=q0, curr_point=down_a_j[0], cov=cov_down_a[0])
        for i in range(1, len(down_a_j)):
            down_a_j[i] = reduce_dist(prev_point=down_a_j[i-1], curr_point=down_a_j[i], cov=cov_down_a[i])
        # Flip down_a_j and the covariance
        down_a_j = np.flip(down_a_j, axis=0)
        cov_down_a = np.flip(cov_down_a, axis=0)
        
    except Exception as e:
        print(e)

    # Object A GMM
    #qp = quadprog(stored_up_a[-1], stored_down_a[0])
    #printGMMcov(stored_up_a,np.concatenate([stored_up_a, qp]), np.concatenate([stored_up_a, stored_down_a]), np.concatenate([stored_up_a, qp, stored_down_a]))
    # Object B GMM
    #qp = quadprog(stored_up_b[-1], stored_down_b[0])
    #printGMMcov(stored_up_b, up_b_j, np.concatenate([stored_up_b, stored_down_b]), np.concatenate([up_b_j, down_b_j]))

    # Object A QP
    #qp = quadprog(stored_up_a[-1], stored_down_a[0])
    #printGMMcov(stored_up_a,np.concatenate([stored_up_a, qp]), np.concatenate([stored_up_a, stored_down_a]), np.concatenate([stored_up_a, qp, stored_down_a]))
    # Object B QP
    #qp = quadprog(stored_up_b[-1], stored_down_b[0])
    #printGMMcov(stored_up_b, up_b_j, np.concatenate([stored_up_b, stored_down_b]), np.concatenate([up_b_j, down_b_j]))
    
    
    # Merge the paths
    #up_b_j = up_b_j_dmp 
    #down_b_j = down_b_j_dmp
    #up_a_j = up_a_j_dmp
    #down_a_j = down_a_j_dmp
    
    #up_b_j, down_b_j, up_a_j, down_a_j= getData()
    # Connection paths
    home_to_start = blendClass.makeTraj(q0, up_b_j[0])
    return_to_start = blendClass.makeTraj(down_b_j[-1], q0)

    move_to_pickup = blendClass.blend_with_viapoints(home_to_start.q, up_b_j,  
                                                np.array([home_to_start.q[-20], up_b_j[0], up_b_j[20]]),
                                                5, plot=False)
    
    blendedPath2 = blendClass.blend_with_viapoints(down_b_j, up_a_j, 
                                            np.array([down_b_j[-20], down_b_j[-1], up_a_j[0], up_a_j[20]]),
                                            5, plot=True)
    
    
    
    blendedPath3 = blendClass.blend_with_viapoints(down_a_j, up_b_j, 
                                            np.array([down_a_j[-20], down_a_j[-1], up_b_j[0],up_b_j[20]]), 
                                            5, plot=True)
    
    return_to_home = blendClass.blend_with_viapoints(down_b_j, return_to_start.q, 
                                                     np.array([down_b_j[-20], return_to_start.q[0], return_to_start.q[20]]),
                                                     5, plot=False)
    
    # Add Pick B QP
    qp1 = quadprog(move_to_pickup[-1], blendedPath2[0])
    print(len(qp1))
    move_to_pickup = np.concatenate([move_to_pickup, qp1])
    # Add Insert A QP
    qp2 = quadprog(blendedPath2[-1], blendedPath3[0])
    print(len(qp2))
    blendedPath2 = np.concatenate([blendedPath2, qp2])
    # Add Place B QP
    qp3 = quadprog(blendedPath3[-1], return_to_home[0])
    blendedPath3 = np.concatenate([blendedPath3, qp3])

    exit(1)
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
        adddotsjoints(return_to_home,(0,0,1))
        env.hold()
    return move_to_pickup, move_insert_return, return_to_home

def oriPath(plot : bool = False):
    up_b_j, down_b_j, up_a_j, down_a_j = getData()
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

    folder = "DMP_QP"

    rtde_r = RTDEReceive(IP)
    with open(f"./Records/experiments/{folder}/acc.txt", "w") as f:
        f.write(f" ")
    with open(f"./Records/experiments/{folder}/vel.txt", "w") as f:
        f.write(f" ")
    with open(f"./Records/experiments/{folder}/pos.txt", "w") as f:
        f.write(f" ")
    with open(f"./Records/experiments/{folder}/tcp.txt", "w") as f:
        f.write(f" ")
    with open(f"./Records/experiments/{folder}/tcp_speed.txt", "w") as f:
        f.write(f" ")

    while RUNNING:
        start_time = time.time()

        # Read joint acceleration, velocity
        acc = rtde_r.getTargetQdd()
        vel = rtde_r.getActualQd()
        tcp = rtde_r.getActualTCPPose()
        pos = rtde_r.getActualQ()
        tcp_speed = rtde_r.getActualTCPSpeed()
        # Append to file
        with open(f"./Records/experiments/{folder}/acc.txt", "a") as f:
            f.write(f"{acc}\n")
        with open(f"./Records/experiments/{folder}/vel.txt", "a") as f:
            f.write(f"{vel}\n")
        with open(f"./Records/experiments/{folder}/pos.txt", "a") as f:
            f.write(f"{pos}\n")
        with open(f"./Records/experiments/{folder}/tcp.txt", "a") as f:
            f.write(f"{tcp}\n")
        with open(f"./Records/experiments/{folder}/tcp_speed.txt", "a") as f:
            f.write(f"{tcp_speed}\n")
            
        
        # Wait for timestep
        delta_time = time.time() - start_time
        if delta_time < 0.02:
            time.sleep(0.02 - delta_time)
        else:
            print(f"Warning: delta_time: {delta_time}")
            exit(1)

def runRobot(speed,move_to_pickup, move_insert_return, return_to_home):
    global RUNNING
    
    #run robot
    rtde_c = RTDEControl(IP)
    print("Starting RTDE test script...")
    VELOCITY = 0.5
    ACCELERATION = 0.2
    BLEND = 0

    rtde_c.moveJ(q0, VELOCITY, ACCELERATION, False)
    rtde_c.speedStop()
    time.sleep(1)

    #gripper = RobotiqGripper()
    #gripper.connect(IP, 63352)
    #gripper.activate()
    #gripper.move_and_wait_for_pos(position=0, speed=5, force=25)

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
    #gripper.move_and_wait_for_pos(position=255, speed=5, force=25)
    time.sleep(1.0)

    for joint in move_insert_return:
        rtde_c.servoJ(joint, 0,0, speed, 0.2, 100)
        time.sleep(speed)
        
    # release the object
    #gripper.move_and_wait_for_pos(position=0, speed=5, force=25)
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


#move_to_pickup, move_insert_return, return_to_home = oriPath(swiftEnv)
move_to_pickup, move_insert_return, return_to_home = blendPath(swiftEnv)

speed = 0.1
if not swiftEnv:
    runRobot(speed, move_to_pickup, move_insert_return, return_to_home)

