from __future__ import division, print_function
from dmp_position import PositionDMP
from dmp_orientation import RotationDMP
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion

import math
 
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
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


if __name__ == '__main__':
    
    # Load a demonstration file containing robot positions.
    demo = np.loadtxt("pydmp_exercise\pydmp_exercise\demo.dat", delimiter=" ", skiprows=1)

    tau = 0.002 * len(demo)
    t = np.arange(0, tau, 0.002)
    
    
    demo_p = demo[:, 0:3]
    demo_o = demo[:,3:demo.shape[-1]]
  

    theta = [np.linalg.norm(v) for v in demo_o]
    axis = [v/np.linalg.norm(v) for v in demo_o]
    demo_q = np.array([Quaternion(axis=a,radians=t) for (a,t) in zip(axis,theta)])

    for i in range(len(demo_q)-1):
        if np.array([demo_q[i][0], demo_q[i][1], demo_q[i][2], demo_q[i][3]]).dot(np.array([demo_q[i+1][0], demo_q[i+1][1], demo_q[i+1][2], demo_q[i+1][3]])) < 0:
            demo_q[i+1] *= -1


    #convert the demo_q to eult angle
    demo_e = np.empty((len(demo_q),3))
    for n, d in enumerate(demo_q):
        demo_e[n] = euler_from_quaternion(d[0],d[1],d[2],d[3])

    #convert the demo_q to numpy array
    demo_quat_array = np.empty((len(demo_q),4))
    for n, d in enumerate(demo_q):
        demo_quat_array[n] = [d[0],d[1],d[2],d[3]]



    N = 200  # TODO: Try changing the number of basis functions to see how it affects the output.
    
    #Position...
    dmp = PositionDMP(n_bfs=N, alpha=48.0)
    dmp.train(demo_p, t, tau)

    # Rotation...
    dmp_rotation = RotationDMP(n_bfs=N, alpha=48.0)
    dmp_rotation.train(demo_q, t, tau)


    # Generate a new trajectory by executing the DMP system.
    dmp_p, dmp_dp, dmp_ddp = dmp.rollout(t, tau)
    dmp_r, dmp_dr, dmp_ddr = dmp_rotation.rollout(t, tau)
    
    #Convert the res of quat to numpy array
    result_quat_array = np.empty((len(dmp_r),4))
    for n, d in enumerate(dmp_r):
        result_quat_array[n] = [d[0],d[1],d[2],d[3]]

    result_eulr = np.empty((len(result_quat_array),3))
    for n, d in enumerate(result_quat_array):
        result_eulr[n] = euler_from_quaternion(d[0],d[1],d[2],d[3])





    # Position DMP 3D    
    fig1 = plt.figure(1)
    fig1.suptitle('Position DMP', fontsize=16)
    ax = plt.axes(projection='3d')
    ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], label='Demonstration')
    ax.plot3D(dmp_p[:, 0], dmp_p[:, 1],dmp_p[:, 2], label='DMP')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # 2D plot the DMP against the original demonstration X, y, Z dir
    fig2, axs = plt.subplots(3, 1, sharex=True)
    fig2.suptitle('Position DMP', fontsize=16)
    axs[0].plot(t, demo_p[:, 0], label='Demonstration')
    axs[0].plot(t, dmp_p[:, 0], label='DMP')
    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('X')
    
    axs[1].plot(t, demo_p[:, 1], label='Demonstration')
    axs[1].plot(t, dmp_p[:, 1], label='DMP')
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Y')

    axs[2].plot(t, demo_p[:, 2], label='Demonstration')
    axs[2].plot(t, dmp_p[:, 2], label='DMP')
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Z')
    axs[2].legend()

    # 2D plot the DMP against the original demonstration X, y, Z dir
    fig3, axs = plt.subplots(3, 1, sharex=True)
    fig3.suptitle('Position DMP Error', fontsize=16)
    axs[0].plot(t, demo_p[:, 0] - dmp_p[:, 0], label='Error')
    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('X')
    
    axs[1].plot(t, demo_p[:, 1] - dmp_p[:, 1], label='Error')
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Y')

    axs[2].plot(t, demo_p[:, 2] - dmp_p[:, 2], label='Error')
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Z')
    axs[2].legend()


    #------------------------------------------------------------------------------------------#

    # Rotation DMP 3D    
    fig4 = plt.figure(4)
    fig4.suptitle('Rotation DMP (Quaternion)', fontsize=16)
    ax = plt.axes(projection='3d')
    ax.plot3D(demo_quat_array[:, 1], demo_quat_array[:, 2], demo_quat_array[:, 3], label='Demonstration')
    ax.plot3D(result_quat_array[:, 1], result_quat_array[:, 2],result_quat_array[:, 3], label='DMP')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()


    # 2D plot the DMP against the original demonstration X, y, Z dir
    fig5, axs = plt.subplots(4, 1, sharex=True)
    fig5.suptitle('Rotation DMP (Quaternion) ', fontsize=16)
    axs[0].plot(t, demo_quat_array[:, 0], label='Demonstration')
    axs[0].plot(t, result_quat_array[:, 0], label='DMP')
    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('Real')
    
    axs[1].plot(t, demo_quat_array[:, 1], label='Demonstration')
    axs[1].plot(t, result_quat_array[:, 1], label='DMP')
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Img 1')

    axs[2].plot(t, demo_quat_array[:, 2], label='Demonstration')
    axs[2].plot(t, result_quat_array[:, 2], label='DMP')
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Img 2')

    axs[3].plot(t, demo_quat_array[:, 3], label='Demonstration')
    axs[3].plot(t, result_quat_array[:,3], label='DMP')
    axs[3].set_xlabel('t (s)')
    axs[3].set_ylabel('Img 3')
    axs[3].legend()



    fig6, axs = plt.subplots(3, 1, sharex=True)
    fig6.suptitle('Rotation DMP (Eulr)', fontsize=16)
    axs[0].plot(t, demo_e[:, 0], label='Demonstration')
    axs[0].plot(t, result_eulr[:, 0], label='DMP')
    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('Roll x')
    
    axs[1].plot(t, demo_e[:, 1], label='Demonstration')
    axs[1].plot(t, result_eulr[:, 1], label='DMP')
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Pitch y')

    axs[2].plot(t, demo_e[:, 2], label='Demonstration')
    axs[2].plot(t, result_eulr[:, 2], label='DMP')
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Yaw z')
    axs[2].legend()


    fig7, axs = plt.subplots(3, 1, sharex=True)
    fig7.suptitle('Rotation DMP (Eulr)- Error', fontsize=16)
    axs[0].plot(t, demo_e[:, 0] - result_eulr[:, 0], label='Error')
    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('Roll x')
    
    axs[1].plot(t, demo_e[:, 1] - result_eulr[:, 1], label='Error')
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Pitch y')

    axs[2].plot(t, demo_e[:, 2] - result_eulr[:, 2], label='Error')
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Yaw z')
    axs[2].legend()

    plt.show()