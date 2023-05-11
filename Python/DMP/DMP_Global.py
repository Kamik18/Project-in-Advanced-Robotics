import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3
from spatialmath.base import q2r   
import spatialgeometry as sg
import math

class DMP_SPC:

    def __init__(self):
        self.bSimulation = True
        self.bPLOT = True
        self.bTimeDifferece = True
        self.bSMOTHER   = True
        self.bSaveFiles = True
        self.bOriention = True
        self.bDifferntTime = True

        self.DEOM_COLOR = 'orange'
        self.DMP_COLOR = 'blue'
        self.DMP_COLOR_NEW_POS = 'green'

        self.DMP_J  = True
        self.DMP_TCP = False
        self.DMP_NEW_POS = True

        self.RUN_DOWN_A = False
        self.RUN_DOWN_B = True
        self.RUN_UP_A = False
        self.RUN_UP_B = False
        

        self.J_GOAL_POS_UP_A= np.array([-79.52, -30.42, 83.56, -56.42, -56.13, 2.97])
        self.J_GOAL_POS_UP_A = np.deg2rad(self.J_GOAL_POS_UP_A)

        self.J_GOAL_POS_UP_B = np.array([4.49, -18.11, 18.78, -94.43, -88.82, 107.88])
        self.J_GOAL_POS_UP_B = np.deg2rad(self.J_GOAL_POS_UP_B)

        self.J_GOAL_POS_DOWN_A = np.array([])
        self.J_GOAL_POS_DOWN_B = np.array([0.300169,-1.672241,1.561409,-1.428667,-1.595302,-2.834139])


        
        if self.RUN_DOWN_A:
            self.TRAINING_TIME = 10.0
            self.DMP_TIME = 10.0
            self.FileName = 'Records/DOWN_A/20/'
            if self.DMP_NEW_POS:
                self.J_GOAL_POS  = self.J_GOAL_POS_DOWN_A
                
        
        if self.RUN_DOWN_B:
            self.TRAINING_TIME = 5.0
            self.DMP_TIME = 5.0
            self.FileName = 'Records/DOWN_B/20/'
            if self.DMP_NEW_POS:
                self.J_GOAL_POS  = self.J_GOAL_POS_DOWN_B

        if self.RUN_UP_A:
            self.TRAINING_TIME = 10.0
            self.DMP_TIME = 10.0
            self.FileName = 'Records/UP_A/20/'
            if self.DMP_NEW_POS:
                self.J_GOAL_POS  = self.J_GOAL_POS_UP_A

        if self.RUN_UP_B:
            self.TRAINING_TIME = 5.0
            self.DMP_TIME = 5.0
            self.FileName = 'Records/UP_B/20/'
            if self.DMP_NEW_POS:
                self.J_GOAL_POS  = self.J_GOAL_POS_UP_B
        
        self.sOutPath = 'Python/DMP/Out/'

    

    def euler_from_quaternion(self, x, y, z, w):
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

    def quat_to_angle_axis(self, q):
        """Converts quaternion to angle axis.

        Args:
            q (_type_): quaternion

        Returns:
            np.array: angle axis np.array([x,y,z])
        """
        angle_axis = np.empty((len(q),3))
        for i in range(len(q)-1):
            q_ = q[i]
            qw= q_[3]
            angle = 2 * np.arccos(qw)
            axis_x = q_[0] / np.sqrt(1 - qw**2)
            axis_y = q_[1] / np.sqrt(1 - qw**2)
            axis_z = q_[2] / np.sqrt(1 - qw**2)
            an_axis = [axis_x * angle, axis_y * angle, axis_z * angle]

            angle_axis[i] = an_axis

        return angle_axis
        
        angle_axis = np.empty((len(q),3))
        for i in range(len(q)-1):
            r = R.from_quat(q[i])
            angle_axis[i] = r.as_rotvec()

        return angle_axis

    def trans_from_pos_quat(self, pos_list, quat_list, bList):
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

    def get_q_from_trans(self, HomgenTrans,UR5,q_init):
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

    def add_marker(self, TransformationMatrix, color,env,bSinglePOint=False):
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

    def read_demo_files(self, filename, skip_lines=4):
        
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
                    values = tuple(map(float, line.split(',')))
                    tuples.append(values)
        demo = np.array(tuples)

        tuples_joints =[]
        with open(filename + "record_j.txt", "r") as f:
            for i, line in enumerate(f):
                if i % skip_lines == skip_lines-1:
                    values = tuple(map(float, line.split(',')))
                    tuples_joints.append(values)

        demo_joint = np.array(tuples_joints)

        return demo, demo_joint

    def convert_angleaxis_to_quat(self, demo_o):
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

    def quaternion_to_np_array(self, dmp_r):
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

    def read_out_file(self,skip_lines=5):
        """Read output files.

        Returns:
            np.array: down_a, down_b, up_a, up_b
        """
        skip_lines=2
        tuples =[]
        with open("Python/DMP/Out/DOWN_A_smoothing.txt", "r") as f:
            for i, line in enumerate(f):
                # Check if the line number is a multiple of skip_lines-1
                if i % skip_lines == skip_lines-1:
                    values = tuple(map(float, line.split(',')))
                    tuples.append(values)
        down_a = np.array(tuples)
        skip_lines=1
        tuples =[]
        with open("Python/DMP/Out/DOWN_B_smoothing.txt", "r") as f:
            for i, line in enumerate(f):
                # Check if the line number is a multiple of skip_lines-1
                if i % skip_lines == skip_lines-1:
                    values = tuple(map(float, line.split(',')))
                    tuples.append(values)
        down_b = np.array(tuples)
        skip_lines=2
        tuples =[]
        with open("Python/DMP/Out/UP_A_smoothing.txt", "r") as f:
            for i, line in enumerate(f):
                # Check if the line number is a multiple of skip_lines-1
                if i % skip_lines == skip_lines-1:
                    values = tuple(map(float, line.split(',')))
                    tuples.append(values)
        up_a = np.array(tuples)
        skip_lines=1
        tuples =[]
        with open("Python/DMP/Out/UP_B_smoothing.txt", "r") as f:
            for i, line in enumerate(f):
                # Check if the line number is a multiple of skip_lines-1
                if i % skip_lines == skip_lines-1:
                    values = tuple(map(float, line.split(',')))
                    tuples.append(values)
        up_b = np.array(tuples)


        return down_a, down_b, up_a, up_b

    def getcolor(self,color):

        if color  == 'red':
            return [1, 0, 0]
        elif color == 'blue':
            return [0, 0, 1]
        elif color == 'green':
            return [0, 1, 0]
        elif color == 'orange':
            return [1, 0.5, 0]
        
    def plot_traj_profile(self, joint_list_demo, dmp_joint_q):
        title = 'Position Profile-Joint Space'
        DEMO_LABEL = 'Demo'
        DMP_LABEL = 'DMP'
        DMP_COLOR = self.getcolor(self.DMP_COLOR)
        DEOM_COLOR = self.getcolor(self.DEOM_COLOR)
        xlabel = 'Time [$s$]'

        Tq = np.gradient(joint_list_demo)
        Tq_acc = np.gradient(Tq[0])
        joint_list_demo = Tq[1]
        joint_demo_dq = Tq[0]
        joint_demo_ddq = Tq_acc[0]


        Tq_dmp = np.gradient(dmp_joint_q)
        Tq_acc_dmp = np.gradient(Tq_dmp[0])
        dmp_joint_q = Tq_dmp[1]
        dmp_joint_dq = Tq_dmp[0]
        dmp_joint_ddq = Tq_acc_dmp[0]

        

        fig1, axs = plt.subplots(6, 1, sharex=True)
        fig1.suptitle(title, fontsize=16)
        axs[0].plot(joint_list_demo[:, 0], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[0].plot(dmp_joint_q[:, 0], label=DMP_LABEL, color=DMP_COLOR)
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel('$q_1$ [$rad$]')

        axs[1].plot(joint_list_demo[:, 1], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[1].plot( dmp_joint_q[:, 1], label=DMP_LABEL, color=DMP_COLOR)
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel('$q_2$ [$rad$]')

        axs[2].plot(joint_list_demo[:, 2], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[2].plot(dmp_joint_q[:, 2], label=DMP_LABEL, color=DMP_COLOR)
        axs[2].set_xlabel(xlabel)
        axs[2].set_ylabel('$q_3$ [$rad$]')

        axs[3].plot(joint_list_demo[:, 3], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[3].plot(dmp_joint_q[:, 3], label=DMP_LABEL, color=DMP_COLOR)
        axs[3].set_xlabel(xlabel)
        axs[3].set_ylabel('$q_4$ [$rad$]')

        axs[4].plot(joint_list_demo[:, 4], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[4].plot(dmp_joint_q[:, 4], label=DMP_LABEL, color=DMP_COLOR)
        axs[4].set_ylabel('$q_5$ [$rad$]')

        axs[5].plot(joint_list_demo[:, 5], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[5].plot(dmp_joint_q[:, 5], label=DMP_LABEL, color=DMP_COLOR)
        axs[5].set_xlabel(xlabel)
        axs[5].set_ylabel('$q_6$ [$rad$]')
        axs[5].legend()

        title = 'Velocity Profile-Joint Space'
        fig2, axs = plt.subplots(6, 1, sharex=True)
        fig2.suptitle(title, fontsize=16)
        axs[0].plot(joint_demo_dq[:, 0], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[0].plot(dmp_joint_dq[:, 0], label=DMP_LABEL, color=DMP_COLOR)
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel('$q_1$ [$rad/s$]')

        axs[1].plot(joint_demo_dq[:, 1], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[1].plot(dmp_joint_dq[:, 1], label=DMP_LABEL, color=DMP_COLOR)
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel('$q_2$ [$rad/s$]')
        
        axs[2].plot(joint_demo_dq[:, 2], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[2].plot(dmp_joint_dq[:, 2], label=DMP_LABEL, color=DMP_COLOR)
        axs[2].set_xlabel(xlabel)
        axs[2].set_ylabel('$q_3$ [$rad/s$]')

        axs[3].plot(joint_demo_dq[:, 3], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[3].plot(dmp_joint_dq[:, 3], label=DMP_LABEL, color=DMP_COLOR)
        axs[3].set_xlabel(xlabel)
        axs[3].set_ylabel('$q_4$ [$rad/s$]')

        axs[4].plot(joint_demo_dq[:, 4], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[4].plot(dmp_joint_dq[:, 4], label=DMP_LABEL, color=DMP_COLOR)
        axs[4].set_ylabel('$q_5$ [$rad/s$]')
        axs[4].set_xlabel(xlabel)

        axs[5].plot(joint_demo_dq[:, 5], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[5].plot(dmp_joint_dq[:, 5], label=DMP_LABEL, color=DMP_COLOR)
        axs[5].set_xlabel(xlabel)
        axs[5].set_ylabel('$q_6$ [$rad/s$]')
        axs[5].legend()

        title = 'Acceleration Profile-Joint Space'
        fig3, axs = plt.subplots(6, 1, sharex=True)
        fig3.suptitle(title, fontsize=16)
        axs[0].plot(joint_demo_ddq[:, 0], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[0].plot(dmp_joint_ddq[:, 0], label=DMP_LABEL, color=DMP_COLOR)
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel('$q_1$ [$rad/s^2$]')

        axs[1].plot(joint_demo_ddq[:, 1], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[1].plot(dmp_joint_ddq[:, 1], label=DMP_LABEL, color=DMP_COLOR)
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel('$q_2$ [$rad/s^2$]')
        
        axs[2].plot(joint_demo_ddq[:, 2], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[2].plot(dmp_joint_ddq[:, 2], label=DMP_LABEL, color=DMP_COLOR)
        axs[2].set_xlabel(xlabel)
        axs[2].set_ylabel('$q_3$ [$rad/s^2$]')

        axs[3].plot(joint_demo_ddq[:, 3], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[3].plot(dmp_joint_ddq[:, 3], label=DMP_LABEL, color=DMP_COLOR)
        axs[3].set_xlabel(xlabel)
        axs[3].set_ylabel('$q_4$ [$rad/s^2$]')

        axs[4].plot(joint_demo_ddq[:, 4], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[4].plot(dmp_joint_ddq[:, 4], label=DMP_LABEL, color=DMP_COLOR)
        axs[4].set_ylabel('$q_5$ [$rad/s^2$]')
        axs[4].set_xlabel(xlabel)

        axs[5].plot(joint_demo_ddq[:, 5], '--', label=DEMO_LABEL, color=DEOM_COLOR)
        axs[5].plot(dmp_joint_ddq[:, 5], label=DMP_LABEL, color=DMP_COLOR)
        axs[5].set_xlabel(xlabel)
        axs[5].set_ylabel('$q_6$ [$rad/s^2$]')
        axs[5].legend()
       
        plt.show()

