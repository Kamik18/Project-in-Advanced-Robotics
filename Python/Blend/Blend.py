import roboticstoolbox as rtb
from roboticstoolbox.tools import trajectory
import numpy as np
import swift
import time
from spatialmath import SE3
from spatialmath.base import *
import spatialgeometry as sg
from cmath import pi, sqrt
#import transforms3d.quaternions as txq
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
#import winsound


class Trajectory():
    def __init__(self, UR5: rtb.ERobot, box=None) -> None:
        #self.env = env
        self.UR5 = UR5
        self.box = box
        self.q0 = [-np.pi / 2, -np.pi / 2, 0, -np.pi / 2, np.pi / 2, 0]
        # Steps per second - 20 steps is equal to a step each 0.05 seconds
        self.sps: int = 20
        
    def inverse_kinematics(self, trans, q0):
        '''
        Function to calculate the inverse kinematic and find alternative solutions

        Args:
        - trans (SE3): desired position
        - q0 (ndarray[6]): array of joint values with 6 joints

        Returns:
        - joints (ndarray[6]): numpy array containing 6 joint angles
        '''
        # Define the distance threshold
        dist_threshold = 0.01

        # Calc inverse kinematic
        joints = self.UR5.ik_lm_wampler(trans, ilimit=500, slimit=500, q0=q0)[0]

        # Validate the initial guess
        if (not self.UR5.iscollided(joints, self.box)):
            dist = self.distance_to_point(
                start_pos=self.UR5.fkine(joints), end_pos=trans)
            if (dist < dist_threshold):
                return joints

        # Check alternative solutions
        for attempt in range(len(self.UR5.q)):
            print("")
            # Try flip one joint 180 degree, as the new initial guess
            joints[attempt] += pi
            joints = self.UR5.ik_lm_wampler(
                trans, ilimit=500, slimit=500, q0=joints)[0]

            # Chack if a solution is found
            if (not self.UR5.iscollided(joints, self.box)):
                dist = self.distance_to_point(
                    start_pos=self.UR5.fkine(joints), end_pos=trans)
                if (dist < dist_threshold):
                    return joints

        # Return the error
        return -1
    
    def distance_to_point(self, start_pos, end_pos):
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

    def makeTraj(self, start_pos, end_pos, duration : int = 2):
        """
        Args:
        - start_pos (SE3): Transformation matrix with starting position
        - end_pos (SE3): Transformation matrix with ending position

        Returns:
        - trajectory (Trajectory instance): The return value is an object that contains position, velocity and acceleration data.
        """

        time_vec = np.linspace(0, duration, self.sps*duration)

        if isinstance(start_pos, SE3) and isinstance(end_pos, SE3):
            print("Cartesian space")
            joint_pos_start = self.inverse_kinematics(start_pos, self.q0)
            joint_pos_end = self.inverse_kinematics(end_pos, joint_pos_start)
            return rtb.jtraj(joint_pos_start, joint_pos_end, time_vec)
            # Catersian space doesnt work problerly with the parabolic blend
            c_traj = rtb.ctraj(start_pos, end_pos, time_vec)
            # Calculate the joint positions
            joint_pos = []
            joint_pos.append(self.inverse_kinematics(c_traj[0], self.q0))
            for i in range(len(c_traj)-1):
                joint_pos.append(self.inverse_kinematics(c_traj[i+1], joint_pos[i]))

            return trajectory.Trajectory("jtraj", time_vec, np.asarray(joint_pos), istime=True)

        if (isinstance(start_pos, list) and isinstance(end_pos, list)) or (isinstance(start_pos, np.ndarray) and isinstance(end_pos, np.ndarray)):
            joint_pos_start = start_pos
            joint_pos_end = end_pos
            return rtb.jtraj(joint_pos_start, joint_pos_end, time_vec)

        print('error in types of start_pos and end_pos')
        return -1  
    
    def blendTraj(self, traj1, traj2, duration, printpath=False):
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
        T1 = len(traj1)/self.sps
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
        time_vec = np.linspace(0, tau, int(tau*self.sps))
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
                q = self.UR5.fkine(joint_pos)
                mark = sg.Sphere(0.01, pose=q, color=(1.0,0.0,0.0))
                self.env.add(mark)
                
            for joint_pos in blended_traj.q:
                q = self.UR5.fkine(joint_pos)
                mark = sg.Sphere(0.01, pose=q, color=(0.0,1.0,0.0))
                self.env.add(mark)

            for joint_pos in traj2.q:
                q = self.UR5.fkine(joint_pos)
                mark = sg.Sphere(0.01, pose=q, color=(1.0,0.0,0.0))
                self.env.add(mark)
        # Combine trajectories
        # Make trajectory of given size
        entireduration = T1+tau
        time_vec = np.linspace(0, entireduration, int(entireduration*self.sps))
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
    
    def blendTwoPointsTraj(self, traj1, traj2, duration, printpath=False):
        """
        
        """        
        
        # End of traj1 has to be same location as start of traj2, within 0.1
        if not np.allclose(traj1[-1], traj2[0], atol=0.1):
            print('traj not same location')
            return -1
        
        tau = duration
        
        #  duration of interpolation is calculated as length of traj1 divided by step_pr_sec
        T1 = len(traj1)/self.sps
        # Define midpoint
        T_mid = traj1[-1]
        
        # Calculate the velocity of the two trajectories
        v1 = (traj1[0] - traj1[-1]) / (0 - T1)
        v2 = (traj2[-1] - traj1[-1]) / T1
        
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
        time_vec = np.linspace(0, tau, int(tau*self.sps))
        
        # Start
        t = T1 - tau
        blended_traj = np.zeros((len(time_vec), 6))
        for i in range(len(time_vec)):
            blended_traj[i] = a * t * t + b * t + c 
            t += 0.1

        return blended_traj
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
                q = self.UR5.fkine(joint_pos)
                mark = sg.Sphere(0.01, pose=q, color=(1.0,0.0,0.0))
                self.env.add(mark)
                
            for joint_pos in blended_traj.q:
                q = self.UR5.fkine(joint_pos)
                mark = sg.Sphere(0.01, pose=q, color=(0.0,1.0,0.0))
                self.env.add(mark)

            for joint_pos in traj2.q:
                q = self.UR5.fkine(joint_pos)
                mark = sg.Sphere(0.01, pose=q, color=(1.0,0.0,0.0))
                self.env.add(mark)
        # Combine trajectories
        # Make trajectory of given size
        entireduration = T1+tau
        time_vec = np.linspace(0, entireduration, int(entireduration*self.sps))
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
    
    def traj_poly(self, s0,stf,sd0,sdtf,sdd0,sddtf,t):
        t0=t[0] # Note! t0 must always 0
        tf=t[-1]
        if t0 != 0:
            print('Error: t0 =', t0)
            return 0
        #solving for equation
        coef = np.zeros((6,1)) #we are looking for this
        param = np.asarray([[s0],[stf],[sd0],[sdtf],[sdd0],[sddtf]])
        mat = np.asarray([[0,0,0,0,0,1],
                [tf**5,tf**4,tf**3,tf**2,tf,1],
                [0,0,0,0,1,0],
                [5*tf**4,4*tf**3,3*tf**2,2*tf,1,0],
                [0,0,0,2,0,0],
                [20*tf**3,12*tf**2,6*tf,2,0,0]])
        mat_i = np.linalg.inv(mat) #inverse
        coef = np.matmul(mat_i,param) #acquiring A B C D E F

        #using equation
        zeros = np.zeros(t.shape)
        ones = np.ones(t.shape)
        twos = ones*2
        mat = np.asarray([ #the original equation
            [t**5,t**4,t**3,t**2,t,ones],
            [5*t**4,4*t**3,3*t**2,2*t,ones,zeros],
            [20*t**3,12*t**2,6*t,twos,zeros,zeros]
        ])
        coef_tensor=(np.repeat(coef,t.size,axis=1))
        coef_tensor=np.reshape(coef_tensor,(coef_tensor.shape[0],1,coef_tensor.shape[1]))
        # d = np.tensordot(mat,coef_tensor,axes=[1, 0]).diagonal(axis1=1, axis2=3) #alternative way
        res = np.einsum('mnr,ndr->mdr', mat, coef_tensor)
        return res

    def jointPositions(self):
        return [
            # Pick up
            [0.7866979241371155, -1.2816428703120728, 1.972131077443258, -2.2549549541869105, -1.5587285200702112, 1.2535892724990845],
            # Start move
            [0.7379084229469299, -1.668586870233053, 1.5848916212665003, -1.552712408160307, -1.547173802052633, 1.2541890144348145],
            # Move end
            [-0.5928247610675257, -1.3727052968791504, 1.3051970640765589, -1.5188871336034317, -1.5476864019977015, 1.254164695739746],
            # Drop off
            [-0.5876477400409144, -1.2064689558795472, 1.8478692213641565, -2.192582746545309, -1.5472973028766077, 1.254176378250122]
        ]

    def transformationPositions(self):
        return [
            # Step 0:
            SE3(np.array([[0, -1, 0, -0.2],    
                        [0, 0, 1, 0.5],    
                        [-1, 0, 0, 0.5],    
                        [0, 0, 0, 1]]), check=False),
            # Step 1:
            SE3(np.array([[0, -1, 0, 0.2],    
                        [0, 0, 1, 0.5],    
                        [-1, 0, 0, 0.5],    
                        [0, 0, 0, 1]]), check=False),
            # Step 2:
            SE3(np.array([[0, -1, 0, 0.2],    
                        [0, 0, 1, 0.5],    
                        [-1, 0, 0, 0.3],    
                        [0, 0, 0, 1]]), check=False),
            # Step 3:
            SE3(np.array([[0, -1, 0, -0.2],    
                        [0, 0, 1, 0.5],    
                        [-1, 0, 0, 0.3],    
                        [0, 0, 0, 1]]), check=False),
            # Step 4:
            SE3(np.array([[0, -1, 0, -0.2],    
                        [0, 0, 1, 0.5],    
                        [-1, 0, 0, 0.1],    
                        [0, 0, 0, 1]]), check=False),
            # Step 5:
            SE3(np.array([[0, -1, 0, 0.2],    
                        [0, 0, 1, 0.5],    
                        [-1, 0, 0, 0.1],    
                        [0, 0, 0, 1]]), check=False)
        ]
    
    def adddots(self, traj, colorref=(1.0,0.0,0.0)):
        for joint_pos in traj.q:
            q = self.UR5.fkine(joint_pos)
            mark = sg.Sphere(0.01, pose=q, color=colorref)
            self.env.add(mark)

    def moveTraj(self, traj):
        for joint_pos in traj.q:
            self.UR5.q = joint_pos
            self.env.step()

    def quad(self, joint_pos1, joint_pos2):
        import cvxopt
        from qpsolvers import solve_qp
        #https://github.com/qpsolvers/qpsolvers
        # Define the objective function coefficients (quadratic and linear terms) for 6 joint angles
        M = np.eye(6)
        P = M.T @ M  # this is a positive definite matrix
        q = np.array([3.0, 2.0, 3.0, 3.0, 2.0, 3.0]) @ M # Linear Term
        G = np.zeros((6,6))
        h = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Inequality constraint vector
        A = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # Equality constraint matrix
        b = np.array([1.0]) # Equality constraint vector

        print('M', M)
        print('P', P)
        print('q', q)
        print('G', G)
        print('h', h)
        print('A', A)
        print('b', b)
        x = solve_qp(P, q, G, h, A, b, solver="cvxopt")
        print('x',x)
        exit(1)
        P = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # Quadratic term
    
        #q = cvxopt.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Linear term
        
        q = np.array(-2.0 * np.array(joint_pos1) + 2.0 * np.array(joint_pos2))  # Linear term
        # Define the constraint matrix and right-hand side vector
        # In this example, we are assuming no constraints for simplicity
        G = cvxopt.matrix(0.0, (1, 6))  # Empty constraint matrix
        h = cvxopt.matrix(0.0, (0, 1))  # Empty constraint right-hand side
        
        # Define the equality constraint matrix and right-hand side vector
        # In this example, we are assuming no equality constraints for simplicity
        A = cvxopt.matrix(0.0, (0, 6))  # Empty equality constraint matrix
        b = cvxopt.matrix(0.0, (0, 1))  # Empty equality constraint right-hand side
        
        # Call the quadratic program solver
        x = solve_qp(P, q, G, h, A, b, solver="cvxopt")
        #sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Extract the optimal solution
        #optimal_solution = sol['x']
        print('optimal solution: ', x)
        print('optimal solution:', [f'{val:.5f}' for val in x])

        # The optimal solution contains the optimized joint angles (or velocities) for controlling the robot arm between the two poses
        # You can use the values in the optimal solution to command the robot arm to move accordingly
