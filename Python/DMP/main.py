from __future__ import division, print_function
from dmp_position import PositionDMP
from dmp_orientation import RotationDMP
from dmp_joint import JointDMP
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3
from spatialmath.base import q2r   
import swift
import spatialgeometry as sg
import math


bSimulation = True
bPLOT = True
bTimeDifferece = True
bSMOTHER   = True
bSaveFiles = True
bOriention = True
bDifferntTime = True
TRAINING_TIME = 10.0
DMP_TIME = 10.0

DMP_J  = True
DMP_TCP = False

J_GOAL_POS_A = np.array([-76.43, -17.78, 48.36, -43.94, 316.89, 104.93])
J_GOAL_POS_A = np.deg2rad(J_GOAL_POS_A)

J_GOAL_POS_B = np.array([5.40, -30.95, 42.53, -103.12, 264.42, 78.42])
J_GOAL_POS_B = np.deg2rad(J_GOAL_POS_B)

#Up_B/1
FileName = 'Records/Up_A/11/'
sOutPath = 'Python/DMP/Out/'

def euler_from_quaternion(x, y, z, w):
    """
    Converts quaternion to euler roll, pitch, yaw.
    Args:
        x (float): x component of quaternion
        y (float): y component of quaternion
        z (float): z component of quaternion
        w (float): w component of quaternion

    Returns:
        tuple: roll, pitch, yaw in radians
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def quat_to_angle_axis(q):
    """Converts quaternion to angle axis.

    Args:
        q (_type_): quaternion

    Returns:
        _type_: angle axis np.array([x,y,z])
    """

    angle_axis = np.empty((len(q),3))
    for i in range(len(q)-1):
        r = R.from_quat(q[i])
        angle_axis[i] = r.as_rotvec()

    return angle_axis

def trans_from_pos_quat(pos_list, quat_list, bList):
    """
    Converts position and quaternion to transformation matrix.
    
    Args:
        pos_list (list): list of positions
        quat_list (list): list of quaternions
        bList (bool): if True, returns a list of transformation matrices, else returns a single transformation matrix
        
    Returns:
        list: list of transformation matrices
    """

    homgenTransList = []
    if bList:
        
        for i in range(len(quat_list)-1):
            quat = quat_list[i]
            pos = pos_list[i]
            x_0 = q2r(quat)
            homgenTrans = np.zeros([4,4], dtype='f')
            homgenTrans[0:3,0:3] = x_0
            homgenTrans[0:3,3] = pos
            homgenTrans[3,3] = 1
            homgenTransList.append(SE3(homgenTrans, check=False))

        return homgenTransList
    else:
        quat = quat_list
        pos = pos_list
        x_0 = q2r(quat)
        homgenTrans = np.zeros([4,4], dtype='f')
        homgenTrans[0:3,0:3] = x_0
        homgenTrans[0:3,3] = pos
        homgenTrans[3,3] = 1

        return SE3(homgenTrans, check=False)

def get_q_from_trans(HomgenTrans,UR5,q_init):
    """
    Converts transformation matrix to joint angles.

    Args:
        HomgenTrans (list): list of transformation matrices
        UR5 (roboticstoolbox): roboticstoolbox UR5 object
        q_init (list): initial joint angles
    
    Returns:
        list: list of joint angles
    """

    q = np.empty((len(HomgenTrans),6))
    for i, trans in enumerate(HomgenTrans): 
        sol = UR5.ikine_LM(trans, q0=q_init)
        q_init = sol.q
        q[i] = sol.q

    return q

def add_marker(TransformationMatrix, color, bSinglePOint=False):
    """
    Adds a marker to the simulation environment.

    Args:
        TransformationMatrix (list): list of transformation matrices
        color (list): list of colors
        bSinglePOint (bool): if True, adds a single marker to the environment, else adds multiple markers
    """
    if bSinglePOint:
        marker = sg.Sphere(0.005, pose=TransformationMatrix, color=color)
        env.add(marker)
        return
    else:    
        for i in TransformationMatrix:
            marker = sg.Sphere(0.005, pose=i, color=color)
            env.add(marker)

def read_demo_files(filename, skip_lines=4):
    
    """Read demo files.

    Args:
        filename (string): folder directory
        skip_lines (int, optional): Skip every x lines. Defaults to 4.

    Returns:
        np.array: demo data and demo joint angles data
    """
    tuples =[]
    with open(filename + "record_tcp.txt", "r") as f:
        for i, line in enumerate(f):
            # Check if the line number is a multiple of skip_lines-1
            if i % skip_lines == skip_lines-1:
                values = tuple(map(float, line.strip()[1:-1].split(',')))
                tuples.append(values)
    demo = np.array(tuples)

    tuples_joints =[]
    with open(filename + "record_j.txt", "r") as f:
        for i, line in enumerate(f):
            if i % skip_lines == skip_lines-1:
                values = tuple(map(float, line.strip()[1:-1].split(',')))
                tuples_joints.append(values)

    demo_joint = np.array(tuples_joints)

    return demo, demo_joint

def convert_angleaxis_to_quat(demo_o):
    """Convert angle axis to quaternion.

    Args:
        demo_o (np.array): angle axis 

    Returns:
        Quaternion: quaternion
    """

    theta = [np.linalg.norm(v) for v in demo_o]
    axis = [v/np.linalg.norm(v) for v in demo_o]
    demo_q = np.array([Quaternion(axis=a,radians=t) for (a,t) in zip(axis,theta)])

    for i in range(len(demo_q)-1):
        if np.array([demo_q[i][0], demo_q[i][1], demo_q[i][2], demo_q[i][3]]).dot(np.array([demo_q[i+1][0], demo_q[i+1][1], demo_q[i+1][2], demo_q[i+1][3]])) < 0:
            demo_q[i+1] *= -1

    return demo_q

def quaternion_to_np_array(dmp_r):
    """Convert quaternion to numpy array.

    Args:
        dmp_r (Quaternion array): _description_

    Returns:
        np.array: x,y,z,w
    """

    result_quat_array = np.empty((len(dmp_r),4))
    for n, d in enumerate(dmp_r):
        result_quat_array[n] = [d[0],d[1],d[2],d[3]]
    
    return result_quat_array





if __name__ == '__main__':
    
    demo,demo_joint = read_demo_files(FileName, skip_lines=10)
    N = 8
    cs_alpha = -np.log(0.0001)
    if DMP_J:
        
        tau = TRAINING_TIME
        t_train = np.arange(0, tau, TRAINING_TIME/ len(demo_joint))

        tau = DMP_TIME 
        t = np.arange(0, tau, DMP_TIME / len(demo_joint))

    
        ## encode DMP 
        dmp_q = JointDMP(NDOF=6,n_bfs=N, alpha=48, beta=12, cs_alpha=cs_alpha)
        dmp_q.train(demo_joint, t_train, tau)

        ## integrate DMP
        q_out, dq_out, ddq_out = dmp_q.rollout(t_train, tau, FX=True)
        J_GOAL_POS_A = demo_joint[-1].copy()
        J_GOAL_POS_A = np.pi
        dmp_q.gp = J_GOAL_POS_A
        q_out_new_pos, dq_out_new_pos, ddq_out_new_pos = dmp_q.rollout(t_train, tau, FX=True)

        ## plot DMP
        if bPLOT:
            fig5, axs = plt.subplots(6, 1, sharex=True)
            fig5.suptitle('DMP-Q ', fontsize=16)
            axs[0].plot(t_train, demo_joint[:, 0], '--', label='Demonstration', color='red')
            axs[0].plot(t, q_out[:, 0], label='DMP', color='blue')
            axs[0].plot(t, q_out_new_pos[:, 0], label='DMP-gp', color='green')
            axs[0].set_xlabel('t (s)')
            axs[0].set_ylabel('q1 (rad)')

            axs[1].plot(t_train, demo_joint[:, 1], '--', label='Demonstration', color='red')
            axs[1].plot(t, q_out[:, 1], label='DMP', color='blue')
            axs[1].plot(t, q_out_new_pos[:, 1], label='DMP-gp', color='green')
            axs[1].set_xlabel('t (s)')
            axs[1].set_ylabel('q2 (rad)')

            axs[2].plot(t_train, demo_joint[:, 2], '--', label='Demonstration', color='red')
            axs[2].plot(t, q_out[:, 2], label='DMP', color='blue')
            axs[2].plot(t, q_out_new_pos[:, 2], label='DMP-gp', color='green')
            axs[2].set_xlabel('t (s)')
            axs[2].set_ylabel('q3 (rad)')

            axs[3].plot(t_train, demo_joint[:, 3], '--', label='Demonstration', color='red')
            axs[3].plot(t, q_out[:, 3], label='DMP', color='blue')
            axs[3].plot(t, q_out_new_pos[:, 3], label='DMP-gp', color='green')
            axs[3].set_xlabel('t (s)')
            axs[3].set_ylabel('q4 (rad)')

            axs[4].plot(t_train, demo_joint[:, 4], '--', label='Demonstration', color='red')
            axs[4].plot(t, q_out[:, 4], label='DMP', color='blue')
            axs[4].plot(t, q_out_new_pos[:, 4], label='DMP-gp', color='green')
            axs[4].set_xlabel('t (s)')
            axs[4].set_ylabel('q5 (rad)')

            axs[5].plot(t_train, demo_joint[:, 5], '--', label='Demonstration', color='red')
            axs[5].plot(t, q_out[:, 5], label='DMP', color='blue')
            axs[5].plot(t, q_out_new_pos[:, 5], label='DMP-gp', color='green')
            axs[5].set_xlabel('t (s)')
            axs[5].set_ylabel('q6 (rad)')
            axs[5].legend()



            plt.show()

        if bSimulation:
            env = swift.Swift()
            env.launch(realtime=True)

            UR5 = rtb.models.UR5() 
            UR5.base = SE3(0.4,0.25,0)
            UR5.payload(1.390, [0,0, 0.057])
            env.add(UR5)

            box = sg.Cuboid([1,1,-0.10], base=SE3(0,-0.14,-0.05), color=[0,0,1])
            env.add(box)
            
            q_init = demo_joint[0]
            UR5.q = q_init        
            env.step()
            for q in q_out:
                trans = UR5.fkine(q)
                add_marker(trans, [0,0,1],True)
                UR5.q = q
                env.step(dt=0.02)
            for q in demo_joint:
                trans = UR5.fkine(q)
                add_marker(trans, [1,0,0],True)
                UR5.q = q
                env.step(dt=0.02)

            for q in q_out_new_pos:
                trans = UR5.fkine(q)
                add_marker(trans, [0,1,0],True)
                UR5.q = q
                env.step(dt=0.02)
            env.hold()



    if DMP_TCP:
        demo_p = demo[:, 0:3]
        demo_o = demo[:,3:demo.shape[-1]]
    
        demo_q = convert_angleaxis_to_quat(demo_o)

        #convert the demo_q to eult angle
        demo_e = np.empty((len(demo_q),3))
        for n, d in enumerate(demo_q):
            demo_e[n] = euler_from_quaternion(d[0],d[1],d[2],d[3])

        #convert the demo_q to numpy array
        demo_quat_array = np.empty((len(demo_q),4))
        for n, d in enumerate(demo_q):
            demo_quat_array[n] = [d[0],d[1],d[2],d[3]]

        ###  TRAINING  ###
    
        tau_train = TRAINING_TIME
        t_train = np.arange(0, tau_train, TRAINING_TIME/ len(demo))
        print("tau_train: ", tau_train)
        
        N = 7 #Number of basis functions: Increasing the number of basis functions can make the trajectory smoother by allowing for more fine-grained control over the shape of the trajectory.
        
        #Position...
        dmp = PositionDMP(n_bfs=N, alpha=40.0,beta=10.0)
        dmp.p0 = (demo_p[0])
        dmp.gp = (demo_p[len(demo_p)-1])
        dmp.train(demo_p, t_train, tau_train)
        
        # Rotation...
        dmp_rotation = RotationDMP(n_bfs=N, alpha=40.0, beta=10.0)
        dmp_rotation.p0 = (demo_q[0])
        dmp_rotation.gp = (demo_q[len(demo_q)-1])
        dmp_rotation.train(demo_q, t_train, tau_train)


        tau = DMP_TIME 
        t = np.arange(0, tau, DMP_TIME / len(demo))
        

        dmp_p, dmp_dp, dmp_ddp = dmp.rollout(t, tau)
        dmp_r, dmp_dr, dmp_ddr = dmp_rotation.rollout(t, tau)
        #Convert the res of quat to numpy array
        result_quat_array = quaternion_to_np_array(dmp_r)

        new_goal_pos = demo_p[-1].copy()
        print("pos_goal_pos: ", new_goal_pos)
        
        new_goal_pos[0] = new_goal_pos[0] - 0.2
        dmp.gp = new_goal_pos
        quat_goal_pos = Quaternion(demo_quat_array[len(demo_quat_array)-1])
        dmp_rotation.gp = quat_goal_pos
        print("quat_goal_pos: ", quat_goal_pos)

        dmp_p_new_goal, dmp_dp_new_goal, dmp_ddp_new_goal = dmp.rollout(t, tau)
        dmp_r_new_goal, dmp_dr_new_goal, dmp_ddr_new_goal = dmp_rotation.rollout(t, tau)

        result_quat_array_new_goal = quaternion_to_np_array(dmp_r_new_goal)

        
        if bSimulation:
            env = swift.Swift()
            env.launch(realtime=True)

            UR5 = rtb.models.UR5() 
            UR5.base = SE3(0.4,0.25,0)
            UR5.payload(1.390, [0,0, 0.057])
            env.add(UR5)

            box = sg.Cuboid([1,1,-0.10], base=SE3(0,-0.14,-0.05), color=[0,0,1])
            env.add(box)
            
            q_init = demo_joint[0]
            print("q_init: ", q_init)
            
            UR5.q = q_init        
            env.step()

            # demo trajectory
            homgenTransList =[]
            homgenTransList = trans_from_pos_quat(demo_p, demo_quat_array, True)  
            add_marker(homgenTransList, [1,0,0])


            print("Done demo")

            # dmp trajectory
            homgenTransList =[]
            homgenTransList = trans_from_pos_quat(dmp_p, result_quat_array, True)
            add_marker(homgenTransList, [0,1,0])

            # dmp trajectory - new goal
            homgenTransList =[]
            homgenTransList = trans_from_pos_quat(dmp_p_new_goal, result_quat_array_new_goal, True)
            add_marker(homgenTransList, [0,1,1])
            sol = UR5.ikine_LM(homgenTransList[-1], q0=demo_joint[-1])
            print("q_final: ", sol.q)
            # for trans in homgenTransList:
            #     add_marker(trans, [1,0.5,1],True)
            #     sol = UR5.ikine_LM(trans, q0=q_init)
            #     q_init = sol.q
            #     UR5.q = sol.q
            #     env.step(dt=0.02)
                
            print("Done")
            


        if bSaveFiles:
            # Save the result to a file
            path = FileName.replace("/", "_")
            path = path.replace("Records", "DMP")
            angle_axis = quat_to_angle_axis(result_quat_array)
            np.savetxt(sOutPath + path + 'smoothing.txt', np.hstack((dmp_p, angle_axis)), delimiter=',', fmt='%1.4f')

            angle_axis = quat_to_angle_axis(result_quat_array_new_goal)
            np.savetxt(sOutPath +  path +'new_goal_pos.txt', np.hstack((dmp_p_new_goal, angle_axis)), delimiter=',', fmt='%1.4f')
            
        if bPLOT:
            if bTimeDifferece:
                # Position DMP 3D    
                fig1 = plt.figure(1)
                fig1.suptitle('Position DMP', fontsize=16)
                ax = plt.axes(projection='3d')
                ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], '--', label='Demonstration', color='red')
                ax.plot3D(dmp_p[:, 0], dmp_p[:, 1],dmp_p[:, 2], label='DMP', color='blue')
                ax.plot3D(dmp_p_new_goal[:, 0], dmp_p_new_goal[:, 1],dmp_p_new_goal[:, 2], label='DMP-gp', color='green')
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z') 
                ax.legend()
                
                
                # 2D plot the DMP against the original demonstration X, y, Z dir
                fig2, axs = plt.subplots(3, 1, sharex=True)
                fig2.suptitle('Position DMP', fontsize=16)
                axs[0].plot(t_train, demo_p[:, 0], '--',label='Demonstration', color='red')
                axs[0].plot(t, dmp_p[:, 0], label='DMP', color='blue')
                axs[0].plot(t, dmp_p_new_goal[:, 0], label='DMP-gp', color='green')
                
                axs[0].set_xlabel('t (s)')
                axs[0].set_ylabel('X')

                
                axs[1].plot(t_train, demo_p[:, 1], '--', label='Demonstration', color='red')
                axs[1].plot(t, dmp_p[:, 1], label='DMP', color='blue')
                axs[1].plot(t, dmp_p_new_goal[:, 1], label='DMP-gp', color='green')
                axs[1].set_xlabel('t (s)')
                axs[1].set_ylabel('Y')

                axs[2].plot(t_train, demo_p[:, 2], '--', label='Demonstration', color='red')
                axs[2].plot(t, dmp_p[:, 2], label='DMP', color='blue')
                axs[2].plot(t, dmp_p_new_goal[:, 2], label='DMP-gp', color='green')
                axs[2].set_xlabel('t (s)')
                axs[2].set_ylabel('Z')
                axs[2].legend()

                #------------------------------------------------------------------------------------------#
                # PLOT QUATERNION IN 3D

                if bOriention:
                    # Rotation DMP 3D    
                    fig4 = plt.figure(4)
                    fig4.suptitle('Rotation DMP (Quaternion)', fontsize=16)
                    ax = plt.axes(projection='3d')
                    ax.plot3D(demo_quat_array[:, 1], demo_quat_array[:, 2], demo_quat_array[:, 3], '--', label='Demonstration', color='red')
                    ax.plot3D(result_quat_array[:, 1], result_quat_array[:, 2],result_quat_array[:, 3], label='DMP', color='blue')
                    ax.plot3D(result_quat_array_new_goal[:, 1], result_quat_array_new_goal[:, 2],result_quat_array_new_goal[:, 3], label='DMP-gp', color='green')
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.legend()


                    # 2D plot the DMP against the original demonstration X, y, Z dir
                    fig5, axs = plt.subplots(4, 1, sharex=True)
                    fig5.suptitle('Rotation DMP (Quaternion) ', fontsize=16)
                    axs[0].plot(t_train, demo_quat_array[:, 0], '--', label='Demonstration', color='red')
                    axs[0].plot(t, result_quat_array[:, 0], label='DMP', color='blue')
                    axs[0].plot(t, result_quat_array_new_goal[:, 0], label='DMP-gp', color='green')
                    axs[0].set_xlabel('t (s)')
                    axs[0].set_ylabel('Real')
                    
                    axs[1].plot(t_train, demo_quat_array[:, 1], '--', label='Demonstration', color='red')
                    axs[1].plot(t, result_quat_array[:, 1], label='DMP', color='blue')
                    axs[1].plot(t, result_quat_array_new_goal[:, 1], label='DMP-gp', color='green')
                    axs[1].set_xlabel('t (s)')
                    axs[1].set_ylabel('Img 1')

                    axs[2].plot(t_train, demo_quat_array[:, 2], '--', label='Demonstration', color='red')
                    axs[2].plot(t, result_quat_array[:, 2], label='DMP', color='blue')
                    axs[2].plot(t, result_quat_array_new_goal[:, 2], label='DMP-gp', color='green')
                    axs[2].set_xlabel('t (s)')
                    axs[2].set_ylabel('Img 2')

                    axs[3].plot(t_train, demo_quat_array[:, 3], '--', label='Demonstration', color='red')
                    axs[3].plot(t, result_quat_array[:,3], label='DMP', color='blue')
                    axs[3].plot(t, result_quat_array_new_goal[:, 3], label='DMP-gp', color='green')
                    axs[3].set_xlabel('t (s)')
                    axs[3].set_ylabel('Img 3')
                    axs[3].legend()

        

            plt.show()
            env.hold()