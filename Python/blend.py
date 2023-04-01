import roboticstoolbox as rtb
import numpy as np
import swift
import time


Joint_pos = [
        # Start Pos
        [-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0] ,
        # Pick up Pos
        [-np.pi / 2, -2.082, -2.139, -0.491, np.pi / 2, 0],
        # Move start
        [-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0],
        # Move end
        [-2.253, -2.154, -0.808, -1.751, np.pi/2, -0.681],
        # Drop end
        [-2.253, -2.426, -1.419, -0.868, np.pi/2, -0.681],
]

# init environtment 
env = swift.Swift()
env.launch(realtime=True)

# load robot 
UR5 = rtb.models.UR5()
env.add(UR5)


# Caluclate the Robots forward kinematics and place the robot
UR5.q = Joint_pos[0]
env.step()
time.sleep(2)

# Create trajectory
num_steps = 50
traj_pickup = rtb.jtraj(Joint_pos[0], Joint_pos[1], num_steps)

# Move robot to pick up joint configuration
for joint_pos in traj_pickup.q:
    UR5.q = joint_pos
    env.step()

# Move robot to up again
for joint_pos in reversed(traj_pickup.q):
    UR5.q = joint_pos
    env.step()

# Create move trajectory
traj_move = rtb.jtraj(Joint_pos[2], Joint_pos[3], num_steps)

# Move robot to start of drop-off pos
for joint_pos in traj_move.q:
    UR5.q = joint_pos
    env.step()

# Create drop trajectory
traj_drop = rtb.jtraj(Joint_pos[3], Joint_pos[4], num_steps)

# Move robot to drop-off pos
for joint_pos in traj_drop.q:
    UR5.q = joint_pos
    env.step()

"""
import roboticstoolbox as rtb
import numpy as np

# Define start and end poses
T_start = np.eye(4)
T_start[0, 3] = 0.1  # move 0.1 units in x direction
T_end = np.eye(4)
T_end[1, 3] = 0.1  # move 0.1 units in y direction

# Calculate midpoint
T_mid = (T_start + T_end) / 2

# Calculate blend trajectory
n_steps = 10
traj = []
for i in range(n_steps):
    t = i / (n_steps - 1)
    T_blend = (1 - t)**2 * T_start + 2 * t * (1 - t) * T_mid + t**2 * T_end
    traj.append(T_blend)

# Move robot along trajectory
for T in traj:
    # Set robot's joint configuration to match current pose
    joint_pos = robot.ikine(T)
    robot.q = joint_pos

    # Update visualization
    env.step()
"""