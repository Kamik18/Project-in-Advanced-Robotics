import roboticstoolbox as rtb
from roboticstoolbox.tools import trajectory
import numpy as np
import swift
import time
from spatialmath import SE3
from spatialmath.base import *
import spatialgeometry as sg
from cmath import pi, sqrt
import transforms3d.quaternions as txq
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import winsound

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

np.set_printoptions(linewidth=100, threshold=6, precision=6, suppress=True)


#####################################################################
# Definitions
# q0 is in upright position
Q0 = [-np.pi / 2, -np.pi / 2, 0, -np.pi / 2, np.pi / 2, 0]
DIST_THRESHOLD = 0.1
# Steps per second - 20 steps is equal to a step each 0.05 seconds
SPS = 20

#####################################################################
# Methods

def inverse_kinematics(trans, q0):
    '''
    Function to calculate the inverse kinematic and find alternative solutions

    Args:
    - trans (SE3): desired position
    - q0 (ndarray[6]): array of joint values with 6 joints

    Returns:
    - joints (ndarray[6]): numpy array containing 6 joint angles
    '''
    # Calc inverse kinematic
    joints = UR5.ik_lm_wampler(trans, ilimit=500, slimit=500, q0=q0)[0]

    # Validate the initial guess
    if (not UR5.iscollided(joints, box)):
        dist = distance_to_point(start_pos=UR5.fkine(joints), end_pos=trans)
        if (dist < DIST_THRESHOLD):
            return joints
    
    # Check alternative solutions
    for attempt in range(len(UR5.q)):
        print("")
        # Try flip one joint 180 degree, as the new initial guess
        joints[attempt] += pi 
        joints = UR5.ik_lm_wampler(trans, ilimit=500, slimit=500, q0=joints)[0]

        # Chack if a solution is found                        
        if (not UR5.iscollided(joints, box)):
            dist = distance_to_point(start_pos=UR5.fkine(joints), end_pos=trans)
            if (dist < DIST_THRESHOLD):
                return joints
    
    # Return the error
    return -1

def distance_to_point(start_pos, end_pos):
    '''
    Function to calculate the distance to a point
    Args:
    - start_pos(SE3): start position
    - end_pos(SE3): desired position

    Returns:
    - distance (int): Euclidian distance between end effector and desired goal position in cartisian space
    ''' 
    # Calculate the distance
    distance = sqrt(((start_pos.t[0] - end_pos.t[0]) ** 2) + ((start_pos.t[1] - end_pos.t[1]) ** 2) + ((start_pos.t[2] - end_pos.t[2]) ** 2)).real
    # Return the distance
    return distance

#def makeTraj(start_pos, end_pos, bJoin, start_joint, end_joint, time_vec):
def makeTraj(start_pos, end_pos, time_vec):
    """
    Args:
    - start_pos (SE3): Transformation matrix with starting position
    - end_pos (SE3): Transformation matrix with ending position

    Returns:
    - trajectory (Trajectory instance): The return value is an object that contains position, velocity and acceleration data.
    """

    if isinstance(start_pos, SE3) and isinstance(end_pos, SE3):
        print("Cartesian space")
        joint_pos_start = inverse_kinematics(start_pos, Q0)
        joint_pos_end = inverse_kinematics(end_pos, joint_pos_start)
        return rtb.jtraj(joint_pos_start, joint_pos_end, time_vec)
        # Catersian space doesnt work problerly with the parabolic blend
        c_traj = rtb.ctraj(start_pos, end_pos, time_vec)
        # Calculate the joint positions
        joint_pos = []
        joint_pos.append(inverse_kinematics(c_traj[0], Q0))
        for i in range(len(c_traj)-1):
            joint_pos.append(inverse_kinematics(c_traj[i+1], joint_pos[i]))

        return trajectory.Trajectory("jtraj", time_vec, np.asarray(joint_pos), istime=True)

    if isinstance(start_pos, list) and isinstance(end_pos, list):
        joint_pos_start = start_pos
        joint_pos_end = end_pos
        return rtb.jtraj(joint_pos_start, joint_pos_end, time_vec)

    print('error in types of start_pos and end_pos')
    return -1  
    
    

def blendTraj(traj1, traj2, step_pr_sec, duration, printpath=False):
    """
    
    """
    
    # Blend time tau is greater than line interpolator
    if len(traj1) != len(traj2):
        print('traj not same size')
        return -1
        new_traj = traj1.q[20:20+len(traj2.q)]
        #traj1 = traj1.q[-len(traj2.q):]
        for i in range(len(new_traj)):
            #new_traj[i] = traj1.q[20+i]
            print('traj1:\t\t', traj1.q[20+i])
            print('new traj:\t', new_traj[i], '\n')
        
        # name is used for plotting
        test = trajectory.Trajectory("jtraj", traj2.t, new_traj, istime=True)
        traj1 = test

        for joint_pos in traj1.q:
            q = UR5.fkine(joint_pos)
            mark = sg.Sphere(0.01, pose=q, color=(0.0,0.0,1.0))
            env.add(mark)
        env.hold()
        exit(1)
        
        #traj1.q = new_traj
    
    # End of traj1 has to be same location as start of traj2, within 0.1
    if not np.allclose(traj1.q[-1], traj2.q[0], atol=0.1):
        print('traj not same location')
        return -1
    
    tau = duration
    
    #  duration of interpolation is calculated as length of traj1 divided by step_pr_sec
    T1 = len(traj1)/step_pr_sec
    # Define midpoint
    T_mid = traj1.q[-1]
    
    # Calculate the velocity of the two trajectories
    v1 = (traj1.q[0] - traj1.q[-1]) / (0 - T1)
    v2 = (traj2.q[-1] - traj1.q[-1]) / T1
    
    # Calculate the coefficients
    K = (v2 - v1) / (4 * tau)
    K2 = -T1 + tau
    K3 = v1
    K4 = -T1
    K5 = T_mid
    
    a = K
    b = 2 * K * K2 + K3
    c = K * pow(K2, 2) + K3 * K4 + K5

    # Create time vector
    time_vec = np.linspace(0, tau, int(tau*step_pr_sec))
    # Start
    t = T1 - tau
    # Create new trajectory with temp data
    blended_traj = rtb.jtraj(v1, v2, time_vec)
    for i in range(len(time_vec)):
        blended_traj.q[i] = a * t * t + b * t + c 
        t += 0.1

    """
    # Create time vector
    time_vec = np.linspace(0, tau*2, int(tau*2*step_pr_sec))
    # Start
    t = T1 - tau
    # Create new trajectory with temp data
    blended_traj = rtb.jtraj(v1, v2, time_vec)
    for i in range(len(time_vec)):
        blended_traj.q[i] = a * t * t + b * t + c 
        t += 0.05
    """
    # Only print paths, if printpath is true
    if printpath:
        for joint_pos in traj1.q:
            q = UR5.fkine(joint_pos)
            mark = sg.Sphere(0.01, pose=q, color=(1.0,0.0,0.0))
            env.add(mark)
            
        for joint_pos in blended_traj.q:
            q = UR5.fkine(joint_pos)
            mark = sg.Sphere(0.01, pose=q, color=(0.0,1.0,0.0))
            env.add(mark)

        for joint_pos in traj2.q:
            q = UR5.fkine(joint_pos)
            mark = sg.Sphere(0.01, pose=q, color=(1.0,0.0,0.0))
            env.add(mark)
    # Combine trajectories
    # Make trajectory of given size
    entireduration = T1+tau
    time_vec = np.linspace(0, entireduration, int(entireduration*step_pr_sec))
    comb_traj = rtb.jtraj(v1, v2, time_vec)
    
    # Combine traj1, blended_traj and traj2
    len_blend = len(blended_traj.q)
    len1 = int((len(traj1.q)-len_blend))
    len2 = len1+len_blend
    comb_traj.q[0:len1] = traj1.q[0:len1]
    comb_traj.q[len1:len2] = blended_traj.q
    comb_traj.q[len2:] = traj2.q[len1:]

    # Return the new trajectory
    return comb_traj

def adddots(traj, colorref=(1.0,0.0,0.0)):
    for joint_pos in traj.q:
        q = UR5.fkine(joint_pos)
        mark = sg.Sphere(0.01, pose=q, color=colorref)
        env.add(mark)

def moveTraj(traj):
    for joint_pos in traj.q:
        UR5.q = joint_pos
        env.step()

def test():
    # Define two Cartesian space trajectories as lists of positions and orientations
    traj1_pos = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    traj1_ori = [[1, 0, 0, 0], [0.707, 0.0, 0.707, 0.0], [0.0, 1.0, 0.0, 0.0]]
    traj2_pos = [[3, 4, 5], [4, 5, 6], [5, 6, 7]]
    traj2_ori = [[0.0, 1.0, 0.0, 0.0], [0.707, 0.0, 0.707, 0.0], [1, 0, 0, 0]]

    # Define the duration of the interpolation
    duration = 2.0

    # Calculate the step size for the interpolation
    num_steps = len(traj1_pos)
    step_size = duration / (num_steps - 1)

    # Initialize the blended trajectory as empty lists
    blended_traj_pos = []
    blended_traj_ori = []

    # Perform linear interpolation between the two trajectories
    for i in range(num_steps):
        # Calculate the blend factor (between 0 and 1)
        t = i * step_size / duration
        
        # Linearly interpolate the position and orientation
        blended_pos = (1 - t) * np.array(traj1_pos[i]) + t * np.array(traj2_pos[i])
        blended_ori = txq.slerp(traj1_ori[i], traj2_ori[i], t)

        # Append the blended position and orientation to the blended trajectory
        blended_traj_pos.append(blended_pos)
        blended_traj_ori.append(blended_ori)

def test2():
    t = np.array([0, 1, 2, 3, 4, 5])
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 2], [1, 2]])
    x = points[:, 0]
    y = points[:, 1]
    x = x.reshape(-1,1)
    y = x.reshape(-1,1)
    print('1.x: ',x.shape)
    print('2.y: ',y.shape)
    
    theta = 2 * np.pi * np.linspace(0, 1, 5)
    y = np.c_[np.cos(theta), np.sin(theta)]
    print('len(y)', len(theta))
    print('y', y.shape)
    """
    plt.figure()
    plt.plot(t, x, t, x, 'o')
    plt.title('x vs. t')
    plt.axis([-0.5, 5.5, -0.5, 1.5])
    """
    tq = np.linspace(0, 5, 8)#np.arange(0, 5.1, 0.1)
    tq2 = np.linspace(0, 5, 51)
    slope0 = 0
    slopeF = 0
    print('3.x', x)
    print('4.slope0', slope0)
    print('5.slopeF', slopeF)
    #tes = np.concatenate(([slope0], x, [slopeF]))
    tes = np.array([0, 0, 1, 1, 0, 0, 1,0]).transpose()
    tes = tes.reshape(-1,1)
    print('6.tes', tes.shape)
    cs_x = CubicSpline(tq, tes)#np.concatenate(([slope0], x, [slopeF])))
    #cs_y = CubicSpline(t, np.concatenate(([slope0], y, [slopeF])))
    xq = cs_x(tq2)
    #yq = cs_y(tq)
    
    plt.figure()
    plt.plot(t, x, t, x, 'o', tq, xq, ':.')
    plt.axis([-0.5, 5.5, -0.5, 1.5])
    plt.show()
#def f(x):  
#        return 1/(1 + (x**2))  
def test3():
    a = -1      
    b = 1
    n = 6   
    xArray = np.linspace(a,b,n)
    yArray = [0, 1, 1, 0, 0,1]#f(xArray)
    plt.figure()
    plt.plot(xArray, yArray, xArray, yArray, 'o', label="Lines, " + str(n) + " points")
    
    
    x = np.linspace(a,b,101)
    cs = CubicSpline(xArray, yArray, True)  # fit a cubic spline
    y = cs(x) # interpolate/extrapolate
    plt.figure()
    plt.plot(xArray, yArray,x, y, label="Interpolation, " + str(n) + " points")
    plt.show()
#####################################################################
# Main
#test3()
#exit(1)

Transform_pos = [
    # Start Transformation:
    SE3(np.array([[0, -1, 0, 0.2876],    
                  [0, 0, 1, 0.4443],    
                  [-1, 0, 0, 0.1638],    
                  [0, 0, 0, 1]]), check=False),
    # pick up Transformation:
    SE3(np.array([[0, -1, 0, 0.2576],    
                  [0, 0, 1, 0.3844],    
                  [-1, 0, 0, 0.469],    
                  [0, 0, 0, 1]]), check=False),
    # move end Transformation:
    SE3(np.array([[0, -1, 0, 0.5355],    
                [0, 0, 1, -2269],    
                [-1, 0, 0, 0.4515],    
                [0, 0, 0, 1]]), check=False),
    # drop off transformation:
    SE3(np.array([[0, -1, 0, 0.5266],    
                [0, 0, 1, -0.2173],    
                [-1, 0, 0, 0.1675],    
                [0, 0, 0, 1]]), check=False)
]

joint_pos = [
    # Pick up
    [0.7866979241371155, -1.2816428703120728, 1.972131077443258, -2.2549549541869105, -1.5587285200702112, 1.2535892724990845],
    # Start move
    [0.7379084229469299, -1.668586870233053, 1.5848916212665003, -1.552712408160307, -1.547173802052633, 1.2541890144348145],
    # Move end
    [-0.5928247610675257, -1.3727052968791504, 1.3051970640765589, -1.5188871336034317, -1.5476864019977015, 1.254164695739746],
    # Drop off
    [-0.5876477400409144, -1.2064689558795472, 1.8478692213641565, -2.192582746545309, -1.5472973028766077, 1.254176378250122]
]

# init environtment 
env = swift.Swift()
env.launch(realtime=True)

# Create an obstacles
box = sg.Cuboid([2,2,-0.1], pose=SE3(0,0,0))
env.add(box)

"""
for joint_pos in t1.q:
    UR5.q = joint_pos
    #print(count, ": ", joint_pos)
    count += 1
    env.step()
"""

# load robot 
UR5 = rtb.models.UR5()
env.add(UR5)
# ik_lm_wampler is the chosen inverse kinematic function used. First part of returned values are the q_values
joint_pos_start = inverse_kinematics(Transform_pos[1], Q0)

# Caluclate the Robots forward kinematics and place the robot
UR5.q = joint_pos[0]#joint_pos_start#[-np.pi/2, 0,0,0,0,0]
env.step()
#time.sleep(1)

# Create time vector
duration = 2
time_vec = np.linspace(0,duration, SPS*duration)
#traj1 = makeTraj(joint_pos[0], joint_pos[1], time_vec)
traj1 = makeTraj(Transform_pos[0], Transform_pos[1], time_vec)
traj2 = makeTraj(Transform_pos[1], Transform_pos[2], time_vec)
traj3 = makeTraj(Transform_pos[2], Transform_pos[3], time_vec)
# start to finish
#traj2 = makeTraj(joint_pos[1],joint_pos[2], time_vec)
#traj3 = makeTraj(joint_pos[2],joint_pos[3], time_vec)

adddots(traj1, (0.0,0.0,1.0))
adddots(traj2, (1.0,0.0,0.0))
adddots(traj3, (0.0,1.0,0.0))
moveTraj(traj1)
moveTraj(traj2)
moveTraj(traj3)

exit(1)

# Make blend
blend_duration = 1
blend1 = blendTraj(traj1, traj2, SPS, blend_duration, False)

#traj3 = makeTraj(Transform_pos[2], Transform_pos[3], time_vec)
blend2 = blendTraj(traj2, traj3, SPS, blend_duration, False)
combined_blend = np.concatenate([blend1.q[0:int(2*(len(blend1.q)/3))], blend2.q[int(len(blend2.q)/3):]])


exit(1)
#run robot
rtde_c = RTDEControl("192.168.1.131")
rtde_r = RTDEReceive("192.168.1.131")
init_q = rtde_r.getActualQ()

#print('init_q: ', init_q)



print("Starting RTDE test script...")
# Target in the robot base
frequency = 440  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
time.sleep(1)
VELOCITY = 0.2
ACCELERATION = 0.1
BLEND = 0

# Move asynchronously in joint space to new_q, we specify asynchronous behavior by setting the async parameter to
# 'True'. Try to set the async parameter to 'False' to observe a default synchronous movement, which cannot be stopped
# by the stopJ function due to the blocking behaviour.
#tmp = traj2.q[0]+0.1
#rtde_c.moveJ(tmp, VELOCITY, ACCELERATION, False)
rtde_c.moveJ(traj1.q[0], VELOCITY, ACCELERATION, False)
rtde_c.speedStop()
time.sleep(1)

for joint in traj1.qd: #combined_blend:
    print(joint)
    rtde_c.speedJ(joint, time=0.05)
    time.sleep(0.05)
for joint in traj2.qd: #combined_blend:
    print(joint)
    rtde_c.speedJ(joint, time=0.05)
    time.sleep(0.05)
for joint in traj3.qd: #combined_blend:
    print(joint)
    rtde_c.speedJ(joint, time=0.05)
    time.sleep(0.05)
    #rtde_c.servoJ(joint, VELOCITY, ACCELERATION, 0.1, 0.1, 100)
#rtde_c.moveJ(new_comb, False)

rtde_c.speedStop()
time.sleep(0.2)
# Stop the movement before it reaches new_q
#rtde_c.stopL(0.5)
print("Stopped movement")
rtde_c.stopScript()
exit(1)

# Target in the Z-Axis of the TCP
target = rtde_r.getActualTCPPose()
print("target: " ,target)
target[2] += 0.10

# Move asynchronously in cartesian space to target, we specify asynchronous behavior by setting the async parameter to
# 'True'. Try to set the async parameter to 'False' to observe a default synchronous movement, which cannot be stopped
# by the stopL function due to the blocking behaviour.
rtde_c.moveL(target, 0.25, 0.5, True)
time.sleep(0.2)
# Stop the movement before it reaches target
rtde_c.stopL(0.5)

# Move back to initial joint configuration
rtde_c.moveL(init_q)

# Stop the RTDE control script
rtde_c.stopScript()
