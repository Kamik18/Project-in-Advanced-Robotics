from __future__ import division, print_function
from dmp_position import PositionDMP
from dmp_orientation import RotationDMP
from dmp_joint import JointDMP
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3
from spatialmath.base import q2r   
import swift
import spatialgeometry as sg
import DMP_Global as dmp_spc



if __name__ == '__main__':

    dmp_spc = dmp_spc.DMP_SPC()    
    
    demo,demo_joint = dmp_spc.read_demo_files(dmp_spc.FileName, skip_lines=10)

    print('demo_tcp: ', demo.shape)
    print('demo_joint: ', demo_joint.shape)

    N = 100
    cs_alpha = -np.log(0.0001)
   
   
    if dmp_spc.DMP_J:
        tau = dmp_spc.TRAINING_TIME
        t_train = np.arange(0, tau, dmp_spc.TRAINING_TIME/ len(demo_joint))
        ## encode DMP 
        tau = dmp_spc.DMP_TIME 
        t = np.arange(0, tau, dmp_spc.DMP_TIME / len(demo_joint))

        ## encode DMP 
        dmp_q = JointDMP(NDOF=6,n_bfs=N, alpha=48, beta=12, cs_alpha=cs_alpha)
        dmp_q.p0 = demo_joint[0].copy()
        dmp_q.gp = demo_joint[-1].copy()
        dmp_q.train(demo_joint, t_train, tau)

        ## integrate DMP
        q_out, dq_out, ddq_out = dmp_q.rollout(t_train, tau, FX=True)
        
        dmp_spc.plot_traj_profile(demo_joint,q_out)

        
        if dmp_spc.DMP_NEW_POS:
            start_pos = np.array([5.78, -46.9, 75.02, -117.3, -87.5, -53.30])
            dmp_q.p0 = dmp_spc.J_GOAL_POS_DOWN_B
            dmp_q.gp = np.deg2rad(start_pos)
            q_out_new_pos, dq_out_new_pos, ddq_out_new_pos = dmp_q.rollout(t_train, tau, FX=True)


        if dmp_spc.bSaveFiles:
            # Save the result to a file
            path = dmp_spc.FileName.split('/')[1]
            path += '_'
            np.savetxt(dmp_spc.sOutPath + path + 'smoothing.txt', q_out, delimiter=',', fmt='%1.6f')
            if dmp_spc.DMP_NEW_POS:
                np.savetxt(dmp_spc.sOutPath +  path +'new_goal_pos.txt', q_out_new_pos, delimiter=',', fmt='%1.6f')
           


        ## plot DMP
        if dmp_spc.bPLOT:
            xlabel = 't [$s$]'
            title = 'DMP-Joint'
            DEMO_LABEL = 'Demo'
            DMP_LABEL = 'DMP'
            DMP_LABEL_NEW_POS = 'DMP-gp'

            fig5, axs = plt.subplots(6, 1, sharex=True)
            fig5.suptitle(title, fontsize=16)
            axs[0].plot(t_train, demo_joint[:, 0], '--', label=DEMO_LABEL, color=dmp_spc.DEOM_COLOR)
            axs[0].plot(t, q_out[:, 0], label=DMP_LABEL, color=dmp_spc.DMP_COLOR)
            if dmp_spc.DMP_NEW_POS:
                axs[0].plot(t, q_out_new_pos[:, 0], label=DMP_LABEL_NEW_POS, color=dmp_spc.DMP_COLOR_NEW_POS)
            axs[0].set_xlabel(xlabel)
            axs[0].set_ylabel('$q_1$ [$rad$]')

            axs[1].plot(t_train, demo_joint[:, 1], '--', label=DEMO_LABEL, color=dmp_spc.DEOM_COLOR)
            axs[1].plot(t, q_out[:, 1], label=DMP_LABEL, color=dmp_spc.DMP_COLOR)
            if dmp_spc.DMP_NEW_POS:
                axs[1].plot(t, q_out_new_pos[:, 1], label=DMP_LABEL_NEW_POS, color=dmp_spc.DMP_COLOR_NEW_POS)
            axs[1].set_xlabel(xlabel)
            axs[1].set_ylabel('$q_2$ [$rad$]')

            axs[2].plot(t_train, demo_joint[:, 2], '--', label=DEMO_LABEL, color=dmp_spc.DEOM_COLOR)
            axs[2].plot(t, q_out[:, 2], label=DMP_LABEL, color=dmp_spc.DMP_COLOR)
            if dmp_spc.DMP_NEW_POS:
                axs[2].plot(t, q_out_new_pos[:, 2], label=DMP_LABEL_NEW_POS, color=dmp_spc.DMP_COLOR_NEW_POS)
            axs[2].set_xlabel(xlabel)
            axs[2].set_ylabel('$q_3$ [$rad$]')

            axs[3].plot(t_train, demo_joint[:, 3], '--', label=DEMO_LABEL, color=dmp_spc.DEOM_COLOR)
            axs[3].plot(t, q_out[:, 3], label=DMP_LABEL, color=dmp_spc.DMP_COLOR)
            if dmp_spc.DMP_NEW_POS:
                axs[3].plot(t, q_out_new_pos[:, 3], label=DMP_LABEL_NEW_POS, color=dmp_spc.DMP_COLOR_NEW_POS)
            axs[3].set_xlabel(xlabel)
            axs[3].set_ylabel('$q_4$ [$rad$]')

            axs[4].plot(t_train, demo_joint[:, 4], '--', label=DEMO_LABEL, color=dmp_spc.DEOM_COLOR)
            axs[4].plot(t, q_out[:, 4], label=DMP_LABEL, color=dmp_spc.DMP_COLOR)
            if dmp_spc.DMP_NEW_POS:
                axs[4].plot(t, q_out_new_pos[:, 4], label=DMP_LABEL_NEW_POS, color=dmp_spc.DMP_COLOR_NEW_POS)
            axs[4].set_xlabel(xlabel)
            axs[4].set_ylabel('$q_5$ [$rad$]')

            axs[5].plot(t_train, demo_joint[:, 5], '--', label=DEMO_LABEL, color=dmp_spc.DEOM_COLOR)
            axs[5].plot(t, q_out[:, 5], label=DMP_LABEL, color=dmp_spc.DMP_COLOR)
            if dmp_spc.DMP_NEW_POS:
                axs[5].plot(t, q_out_new_pos[:, 5], label=DMP_LABEL_NEW_POS, color=dmp_spc.DMP_COLOR_NEW_POS)
            axs[5].set_xlabel(xlabel)
            axs[5].set_ylabel('$q_6$ [$rad$]')
            axs[5].legend()

            plt.show()

        if dmp_spc.bSimulation:
            env = swift.Swift()
            env.launch(realtime=True)

            UR5 = rtb.models.UR5() 
            UR5.base = SE3(0,0,0)
            UR5.payload(1.390, [0,0, 0.057])
            env.add(UR5)
            box = sg.Cuboid([1,1,-0.10], base=SE3(0.3,0.3,-0.05), color=[0,0,1])
            env.add(box)
            
            q_init = demo_joint[0]
            UR5.q = q_init        
            env.step()
                       
            for q in q_out:
                trans = UR5.fkine(q)
                dmp_spc.add_marker(trans, dmp_spc.getcolor(dmp_spc.DMP_COLOR),env,True)
                UR5.q = q
                env.step(dt=0.02)
            for q in demo_joint:
                trans = UR5.fkine(q)
                dmp_spc.add_marker(trans, dmp_spc.getcolor(dmp_spc.DEOM_COLOR),env,True)
                UR5.q = q
                env.step(dt=0.02)

            if dmp_spc.DMP_NEW_POS:
                for q in q_out_new_pos:
                    trans = UR5.fkine(q)
                    dmp_spc.add_marker(trans, dmp_spc.getcolor(dmp_spc.DMP_COLOR_NEW_POS),env,True)
                    UR5.q = q
                    env.step(dt=0.02)  

            env.hold()



    if dmp_spc.DMP_TCP:
        demo_p = demo[:, 0:3]
        demo_o = demo[:,3:demo.shape[-1]]
    
        demo_q = dmp_spc.convert_angleaxis_to_quat(demo_o)

        #convert the demo_q to eult angle
        demo_e = np.empty((len(demo_q),3))
        for n, d in enumerate(demo_q):
            demo_e[n] = dmp_spc.euler_from_quaternion(d[0],d[1],d[2],d[3])

        #convert the demo_q to numpy array
        demo_quat_array = np.empty((len(demo_q),4))
        for n, d in enumerate(demo_q):
            demo_quat_array[n] = [d[0],d[1],d[2],d[3]]

        ###  TRAINING  ###
    
        tau_train = dmp_spc.TRAINING_TIME
        t_train = np.arange(0, tau_train, dmp_spc.TRAINING_TIME/ len(demo))
        print("tau_train: ", tau_train)

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


        tau = dmp_spc.DMP_TIME 
        t = np.arange(0, tau, dmp_spc.DMP_TIME / len(demo))
        

        dmp_p, dmp_dp, dmp_ddp = dmp.rollout(t, tau)
        dmp_r, dmp_dr, dmp_ddr = dmp_rotation.rollout(t, tau)
        #Convert the res of quat to numpy array
        result_quat_array = dmp_spc.quaternion_to_np_array(dmp_r)

        

        new_goal_pos = demo_p[-1].copy()
        print("pos_goal_pos: ", new_goal_pos)
        
        new_goal_pos[0] = new_goal_pos[0] - 0.2
        dmp.gp = new_goal_pos
        quat_goal_pos = Quaternion(demo_quat_array[len(demo_quat_array)-1])
        dmp_rotation.gp = quat_goal_pos
        print("quat_goal_pos: ", quat_goal_pos)

        dmp_p_new_goal, dmp_dp_new_goal, dmp_ddp_new_goal = dmp.rollout(t, tau)
        dmp_r_new_goal, dmp_dr_new_goal, dmp_ddr_new_goal = dmp_rotation.rollout(t, tau)

        result_quat_array_new_goal = dmp_spc.quaternion_to_np_array(dmp_r_new_goal)

        angle_axis_demo = dmp_spc.quat_to_angle_axis(demo_quat_array)
        angle_axis = dmp_spc.quat_to_angle_axis(result_quat_array)

        fig, axs = plt.subplots(6, 1, figsize=(10, 10))
        axs[0].plot(angle_axis_demo[:, 0], label=DMP_LABEL_NEW_POS, color='red')
        axs[0].plot(angle_axis[:, 0], label=DMP_LABEL_NEW_POS, color='red')


        if dmp_spc.bSimulation:
            env = swift.Swift()
            env.launch(realtime=True)

            UR5 = rtb.models.UR5() 
            UR5.base = SE3(0,0,0)
            UR5.payload(1.390, [0,0, 0.057])
            env.add(UR5)

            box = sg.Cuboid([1,1,-0.10], base=SE3(0,0,-0.05), color=[0,0,1])
            env.add(box)
            
            q_init = demo_joint[0]
            print("q_init: ", q_init)
            
            UR5.q = q_init        
            env.step()

            # demo trajectory
            homgenTransList =[]
            homgenTransList = dmp_spc.trans_from_pos_quat(demo_p, demo_quat_array, True)  

            for trans in homgenTransList:
                dmp_spc.add_marker(trans, [1,0.5,0.5],True)
                sol = UR5.ikine_LM(trans, q0=q_init)
                q_init = sol.q
                UR5.q = sol.q
                env.step(dt=0.02)
            env.hold()
            dmp_spc.add_marker(homgenTransList, [1,0,0])


            print("Done demo")

            # dmp trajectory
            homgenTransList =[]
            homgenTransList = dmp_spc.trans_from_pos_quat(dmp_p, result_quat_array, True)
            dmp_spc.add_marker(homgenTransList, [0,1,0])

            # dmp trajectory - new goal
            homgenTransList =[]
            homgenTransList = dmp_spc.trans_from_pos_quat(dmp_p_new_goal, result_quat_array_new_goal, True)
            dmp_spc.add_marker(homgenTransList, [0,1,1])
            
            
                
            print("Done")
            


        if dmp_spc.bSaveFiles:
            # Save the result to a file
            path = dmp_spc.FileName.replace("/", "_")
            path = path.replace("Records", "DMP")
            angle_axis = dmp_spc.quat_to_angle_axis(result_quat_array)
            np.savetxt(dmp_spc.sOutPath + path + 'smoothing.txt', np.hstack((dmp_p, angle_axis)), delimiter=',', fmt='%1.4f')

            angle_axis = dmp_spc.quat_to_angle_axis(result_quat_array_new_goal)
            np.savetxt(dmp_spc.sOutPath +  path +'new_goal_pos.txt', np.hstack((dmp_p_new_goal, angle_axis)), delimiter=',', fmt='%1.4f')
            
        if dmp_spc.bPLOT:
            if dmp_spc.bTimeDifferece:
                # Position DMP 3D    
                fig1 = plt.figure(1)
                fig1.suptitle('Position DMP', fontsize=16)
                ax = plt.axes(projection='3d')
                ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], '--', label=DEMO_LABEL, color='red')
                ax.plot3D(dmp_p[:, 0], dmp_p[:, 1],dmp_p[:, 2], label=DMP_LABEL, color='blue')
                ax.plot3D(dmp_p_new_goal[:, 0], dmp_p_new_goal[:, 1],dmp_p_new_goal[:, 2], label=DMP_LABEL_NEW_POS, color='green')
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z') 
                ax.legend()
                
                
                # 2D plot the DMP against the original demonstration X, y, Z dir
                fig2, axs = plt.subplots(3, 1, sharex=True)
                fig2.suptitle('Position DMP', fontsize=16)
                axs[0].plot(t_train, demo_p[:, 0], '--',label=DEMO_LABEL, color='red')
                axs[0].plot(t, dmp_p[:, 0], label=DMP_LABEL, color='blue')
                axs[0].plot(t, dmp_p_new_goal[:, 0], label=DMP_LABEL_NEW_POS, color='green')
                
                axs[0].set_xlabel(xlabel)
                axs[0].set_ylabel('X')

                
                axs[1].plot(t_train, demo_p[:, 1], '--', label=DEMO_LABEL, color='red')
                axs[1].plot(t, dmp_p[:, 1], label=DMP_LABEL, color='blue')
                axs[1].plot(t, dmp_p_new_goal[:, 1], label=DMP_LABEL_NEW_POS, color='green')
                axs[1].set_xlabel(xlabel)
                axs[1].set_ylabel('Y')

                axs[2].plot(t_train, demo_p[:, 2], '--', label=DEMO_LABEL, color='red')
                axs[2].plot(t, dmp_p[:, 2], label=DMP_LABEL, color='blue')
                axs[2].plot(t, dmp_p_new_goal[:, 2], label=DMP_LABEL_NEW_POS, color='green')
                axs[2].set_xlabel(xlabel)
                axs[2].set_ylabel('Z')
                axs[2].legend()

                #------------------------------------------------------------------------------------------#
                # PLOT QUATERNION IN 3D

                if dmp_spc.bOriention:
                    # Rotation DMP 3D    
                    fig4 = plt.figure(4)
                    fig4.suptitle('Rotation DMP (Quaternion)', fontsize=16)
                    ax = plt.axes(projection='3d')
                    ax.plot3D(demo_quat_array[:, 1], demo_quat_array[:, 2], demo_quat_array[:, 3], '--', label=DEMO_LABEL, color='red')
                    ax.plot3D(result_quat_array[:, 1], result_quat_array[:, 2],result_quat_array[:, 3], label=DMP_LABEL, color='blue')
                    ax.plot3D(result_quat_array_new_goal[:, 1], result_quat_array_new_goal[:, 2],result_quat_array_new_goal[:, 3], label=DMP_LABEL_NEW_POS, color='green')
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.legend()


                    # 2D plot the DMP against the original demonstration X, y, Z dir
                    fig5, axs = plt.subplots(4, 1, sharex=True)
                    fig5.suptitle('Rotation DMP (Quaternion) ', fontsize=16)
                    axs[0].plot(t_train, demo_quat_array[:, 0], '--', label=DEMO_LABEL, color='red')
                    axs[0].plot(t, result_quat_array[:, 0], label=DMP_LABEL, color='blue')
                    axs[0].plot(t, result_quat_array_new_goal[:, 0], label=DMP_LABEL_NEW_POS, color='green')
                    axs[0].set_xlabel(xlabel)
                    axs[0].set_ylabel('Real')
                    
                    axs[1].plot(t_train, demo_quat_array[:, 1], '--', label=DEMO_LABEL, color='red')
                    axs[1].plot(t, result_quat_array[:, 1], label=DMP_LABEL, color='blue')
                    axs[1].plot(t, result_quat_array_new_goal[:, 1], label=DMP_LABEL_NEW_POS, color='green')
                    axs[1].set_xlabel(xlabel)
                    axs[1].set_ylabel('Img 1')

                    axs[2].plot(t_train, demo_quat_array[:, 2], '--', label=DEMO_LABEL, color='red')
                    axs[2].plot(t, result_quat_array[:, 2], label=DMP_LABEL, color='blue')
                    axs[2].plot(t, result_quat_array_new_goal[:, 2], label=DMP_LABEL_NEW_POS, color='green')
                    axs[2].set_xlabel(xlabel)
                    axs[2].set_ylabel('Img 2')

                    axs[3].plot(t_train, demo_quat_array[:, 3], '--', label=DEMO_LABEL, color='red')
                    axs[3].plot(t, result_quat_array[:,3], label=DMP_LABEL, color='blue')
                    axs[3].plot(t, result_quat_array_new_goal[:, 3], label=DMP_LABEL_NEW_POS, color='green')
                    axs[3].set_xlabel(xlabel)
                    axs[3].set_ylabel('Img 3')
                    axs[3].legend()

        

            plt.show()
            env.hold()