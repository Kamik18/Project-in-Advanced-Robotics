import numpy as np
import matplotlib.pyplot as plt
from spatialmath.base import x2r
import swift
import spatialgeometry as sg
import roboticstoolbox as rtb
from spatialmath import SE3
from Blend import Trajectory
import time


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
#up_b: np.ndarray = np.loadtxt("./Records/Up_B/1/record_tcp.txt", delimiter=',', skiprows=0)
down_b: np.ndarray = fetch_data_from_records("./Python/Blend/Down_B_Tcp.txt")
#down_a: np.ndarray = fetch_data_from_records("./Python/Blend/Down_A_Tcp.txt")
up_a: np.ndarray = fetch_data_from_records("./Python/Blend/Up_A_Tcp.txt")

# Joints
#up_b_j: np.ndarray = fetch_data_from_records("./Python/Blend/Up_B_j.txt")
#up_b_j: np.ndarray = np.loadtxt("./Records/Up_B/1/record_j.txt", delimiter=',', skiprows=0)
down_b_j: np.ndarray = fetch_data_from_records("./Python/Blend/Down_B_j.txt")
#down_a_j: np.ndarray = fetch_data_from_records("./Python/Blend/Down_A_j.txt")
up_a_j: np.ndarray = fetch_data_from_records("./Python/Blend/Up_A_j.txt")

#env = swift.Swift()
#env.launch(realtime=True)

# Create an obstacles
box = sg.Cuboid([2,2,-0.1], pose=SE3(0,0,0))
#env.add(box)

UR5 = rtb.models.UR5()
#env.add(UR5)

# Move the robot to the start position
traj = Trajectory(UR5=UR5, box=box)
q0 =  [-np.pi / 2, -np.pi / 2, 0, -np.pi / 2, np.pi / 2, 0]
UR5.q = q0
#env.step()

# Connection paths
connectionTraj = traj.makeTraj(down_b_j[-1], up_a_j[0])
#returnToStart = traj.makeTraj(down_a[-1], up_b[0])

### MAKE THREE DOTS ###
p1 = UR5.fkine(down_b_j[-100])
mark1 = sg.Sphere(0.01, pose=p1, color=(0,0,1))
p2 = UR5.fkine(connectionTraj.q[0])
mark2 = sg.Sphere(0.01, pose=p2, color=(0,1,0))
p3 = UR5.fkine(connectionTraj.q[10])
mark3 = sg.Sphere(0.01, pose=p3, color=(1,0,0))
#env.add(mark1)
#env.add(mark2)
#env.add(mark3)
#env.hold()

"""
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(p1.t[0], p2.t[0], p3.t[0],c='r',marker='o')
ax1.scatter(p1.t[1], p2.t[1], p3.t[1],c='b',marker='o')
ax1.scatter(p1.t[2], p2.t[2], p3.t[2],c='g',marker='o')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Plot Points')
#ax2.set_xlim(0,1)
#ax2.set_ylim(0,1)
#ax2.set_zlim(0,1)



via_y = np.asarray([0,20,30,40,20])
dur_y = np.asarray([10,15,20,30])
tb_y = np.asarray([1,1,1,1,1])*10
"""
######################## TODO ########################
# 1. fix python intellisense somehow
# 2. Find out how to combine the blends new paths that are created
# 3. Add rotation
# 4. Try with joint angles
# 5. Simulate with Swift

acc = 2
time = 2
#print('p1.t[0]', p1.t[0], 'p2.t[0]', p2.t[0], 'p3.t[0]', p3.t[0])
via_x = np.asarray([p1.t[0],p2.t[0],p3.t[0]])
dur_x = np.asarray([time,time])
tb_x = np.asarray([1,1,1])*acc
res_x = traj.lspb(via_x, dur_x, tb_x)

via_y = np.asarray([p1.t[1],p2.t[1],p3.t[1]])
dur_y = np.asarray([time,time])
tb_y = np.asarray([1,1,1])*acc
res_y = traj.lspb(via_y, dur_y, tb_y)

via_z = np.asarray([p1.t[2],p2.t[2],p3.t[2]])
dur_z = np.asarray([time,time])
tb_z = np.asarray([1,1,1])*acc
res_z = traj.lspb(via_z, dur_z, tb_z)

print('p1', p1.t)
print('p2', p2.t)
print('p3', p3.t)

fig3 = plt.figure(figsize=(12,12))
ax3 = fig3.add_subplot(211)
ax3.plot(res_x[2],via_x,'*',res_x[3],res_x[4], label='x')
ax3.plot(res_y[2],via_y,'*',res_y[3],res_y[4], label='y')
ax3.plot(res_z[2],via_z,'*',res_z[3],res_z[4], label='z')
ax3.legend()

#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111, projection='3d')
#ax1.plot(res_x[3], res_x[4], res_y[4], res_z[4], 'r')

trans = np.ndarray(shape=(len(res_x[3]),3))

for i in range(len(res_x[4])):
    trans[i] = np.array([res_x[4][i],res_y[4][i],res_z[4][i]])

print('trans[0]', trans[0])
print('trans[-1]', trans[-1])
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111, projection='3d')
ax2 = fig3.add_subplot(212, projection='3d')
ax2.plot(trans[:,0], trans[:,1], trans[:,2],c='r')
#ax2.plot(res_x[4], res_y[4], res_z[4], c='r')
ax2.scatter(p1.t[0], p1.t[1], p1.t[2],c='r',marker='o')
ax2.scatter(p2.t[0], p2.t[1], p2.t[2],c='b',marker='o')
ax2.scatter(p3.t[0], p3.t[1], p3.t[2],c='g',marker='o')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Line through Points')
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
ax2.set_zlim(0,1)
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


"""
#blendtesttraj = traj.blendTwoPointsTraj(connectionTraj.q, up_a, 1)
#blendtesttraj = traj.blendTwoPointsTraj(down_a, returnToStart.q,1)

adddots(returnToStart.q, (0,1,0)) 
adddots(down_a, (1,0,0)) # Starts up and moves down
for joint_pos in blendtesttraj:
    UR5.q = joint_pos
    q = UR5.fkine(joint_pos)
    mark = sg.Sphere(0.01, pose=q, color=(0,0,1))
    env.add(mark)
    env.step()

env.hold()
Privacy Badget
Adblocker

joined_path = np.concatenate([up_b, down_b, up_a, down_a])
 
for joint_pos in joined_path:
    UR5.q = joint_pos
    #UR5.q = traj.inverse_kinematics(create_4x4_matrix(joint_pos), q0)
    q = UR5.fkine(joint_pos)
    mark = sg.Sphere(0.01, pose=q, color=(0,0,1))
    env.add(mark)
    env.step()

"""
#adddots(up_b, (1,0,0)) # Starts up and moves down
#adddots(down_b_j, (0,0,1)) # Starts down and moves up
#adddots(connectionTraj.q, (0,1,0))
#adddots(up_a_j, (0,0,0)) # Starts up and moves down
#adddots(down_a, (0,1,0)) # Starts down and moves up
#env.hold()
