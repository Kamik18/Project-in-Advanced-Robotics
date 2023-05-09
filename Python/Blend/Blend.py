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


class Blend():
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
            return rtb.ctraj(start_pos, end_pos, time_vec)
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
    
    def blendTrajJointSpace(self, traj1, traj2, duration, printpath=False):
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
    
    def traj_poly(self, s0, stf, sd0, sdtf,sdd0  ,sddtf, t):
        """
        This is a polynomial trajectory generator. 
        
        Args:
        - s0 (int): Initial position
        - stf (int): Final position
        - sd0 (int): Initial velocity
        - sdtf (int): Final velocity
        - sdd0 (int): Initial acceleration
        - sddtf (int): Final acceleration
        - t (np.ndarray): Time vector

        Returns:
        - np.ndarray: Data from the records. 
        - np.ndarray: Returns a an array of a polynomial trajectory of the given order. Shape: (both position, velocity, acceleration)
        """
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

    def linearInterpolation(self, point0, v0, t0, t1, step=1):
        """
        function for linear interpolation

        Returns:
            - np.ndarray: (time vector, position, velocity, acceleration)
        """
        # Generate a series of timestep
        t = np.arange(t0, t1+step,step)#makit one column
        # Calculate velocity
        v = v0
        #time shift
        Ti = t0
        #equation
        s = point0 + v*(t-Ti)
        v = np.ones(t.size)*v
        a = np.zeros(t.size)
        return (t,s,v,a)

    def parab(self, p0, v0, v1, t0, t1, step=1):
        # Generate a series of timestep
        t = np.arange(t0, t1+step,step)        
        #calculate acceleration
        a = (v1-v0)/(t1-t0)
        #time shift
        Ti=t0
        # equation
        s = p0  +v0*(t-Ti) +0.5*a*(t-Ti)**2
        v = v0 + a*(t-Ti)
        a = np.ones(t.size)*a
        return (t,s,v,a)

    def lspb(self, via,dur,tb):
        """
        https://github.com/novice1011/trajectory-planning
        1. It must start and end at the first and last waypoint respectively with zero velocity
        2. Note that during the linear phase acceleration is zero, velocity is constant and position is linear in time
        Args:
         - via (np.ndarray): array of via points
         - dur (np.ndarray): array of duration for each segment
         - tb (np.ndarray): array of acceleration for each segment
        
        """
        STEP = 0.1
        
        # if acc.min < 0 :
        #     print('acc must bigger than 0')
        #     return 0
        if ((via.size-1) != dur.size):
            print('duration must equal to number of segment which is via-1')
            return 0
        if (via.size <2):
            print('minimum of via is 2')
            return 0
        if (via.size != (tb.size)):
            print('acc must equal to number of via')
            return 0
        
        #=====CALCULATE-VELOCITY-EACH-SEGMENT=====
        v_seg=np.zeros(dur.size)
        for i in range(0,len(via)-1):
            v_seg[i]=(via[i+1]-via[i])/dur[i]

        #=====CALCULATE-ACCELERATION-EACH-VIA=====
        a_via=np.zeros(via.size)
        a_via[0]=(v_seg[0]-0)/tb[0]
        for i in range(1,len(via)-1):
            a_via[i]=(v_seg[i]-v_seg[i-1])/tb[i]
        a_via[-1]=(0-v_seg[-1])/tb[-1]

        #=====CALCULATE-TIMING-EACH-VIA=====
        T_via=np.zeros(via.size)
        T_via[0]=0.5*tb[0]
        for i in range(1,len(via)-1):
            T_via[i]=T_via[i-1]+dur[i-1]
        T_via[-1]=T_via[-2]+dur[-1]

        #=====GENERATING-CHART/GRAPH/FIGURE=====
        # q(t) = q_i + v_{i-1}(t-T_i) + \frac{1}{2}a(t-T_i+\frac{t_i^b}{2})^2  #parabolic phase
        # q(t) = q_i + v_i*(t-T_i)                 #linear phase
        #parabolic
        
        t,s,v,a = self.parab(via[0], 0, v_seg[0], T_via[0]-0.5*tb[0], T_via[0]+0.5*tb[0], step=STEP)
        time    = t
        pos     = s
        speed   = v
        accel   = a
        
        for i in range(1,len(via)-1):
            # linear
            t,s,v,a = self.linearInterpolation(pos[-1],v_seg[i-1],T_via[i-1]+0.5*tb[i],T_via[i]-0.5*tb[i+1],STEP)
            time    = np.concatenate((time,t))
            pos     = np.concatenate((pos,s))
            speed   = np.concatenate((speed,v))
            accel   = np.concatenate((accel,a))

            #parabolic
            t,s,v,a = self.parab(pos[-1], v_seg[i-1], v_seg[i], T_via[i]-0.5*tb[i+1], T_via[i]+0.5*tb[i+1], STEP)
            time    = np.concatenate((time,t))
            pos     = np.concatenate((pos,s))
            speed   = np.concatenate((speed,v))
            accel   = np.concatenate((accel,a))

        # linear
        t,s,v,a = self.linearInterpolation(pos[-1],v_seg[-1],T_via[-2]+0.5*tb[-2],T_via[-1]-0.5*tb[-1],STEP)
        time    = np.concatenate((time,t))
        pos     = np.concatenate((pos,s))
        speed   = np.concatenate((speed,v))
        accel   = np.concatenate((accel,a))
        
        #parabolic
        t,s,v,a = self.parab(pos[-1], v_seg[-1], 0, T_via[-1]-0.5*tb[-1],  T_via[-1]+0.5*tb[-1], STEP)
        time    = np.concatenate((time,t))
        pos     = np.concatenate((pos,s))
        speed   = np.concatenate((speed,v))
        accel   = np.concatenate((accel,a))
        
        """
        print('v seg = ',v_seg,
        '\na via = ',a_via,
        '\nT via = ',T_via,
        '\ntime = ',time,
        '\npos = ',pos)
        """
        return(v_seg,a_via,T_via,time,pos,speed,accel)

    def blendTraj(self, traj1, traj2, dur: int=2, bsize1: int =20, bsize2: int = 20, plot: bool=True):       
        """
        Args:
        - traj1 (np.ndarray): array with x,y,z and rx,ry,rz
        - traj2 (np.ndarray): array with x,y,z and rx,ry,rz
        - dur (int): duration length
        - bsize1 (int): size of the blend for traj 1
        - bsize2 (int): size of the blend for traj 2
        - plot (bool): determines if plots are shown
        """
        if type(traj1) == SE3:
            traj1 = traj1.t
            
        if type(traj2) == SE3:
            traj2 = traj2.t
    
        p1 = traj1[-bsize1][0:3]
        p2 = traj2[0][0:3]
        p3 = traj2[bsize2][0:3]
        print('p1', p1)
        print('p2', p2)
        print('p3', p3)

        o1 = traj1[-bsize1][3:6]
        o2 = traj2[0][3:6]
        o3 = traj2[bsize2][3:6]

        ACC = 10
        # Translation
        via_x = np.asarray([p1[0],p2[0],p3[0]])
        dur_x = np.asarray([dur,dur])
        tb_x = np.asarray([1,1,1])*ACC
        res_x = self.lspb(via_x, dur_x, tb_x)

        via_y = np.asarray([p1[1],p2[1],p3[1]])
        dur_y = np.asarray([dur,dur])
        tb_y = np.asarray([1,1,1])*ACC
        res_y = self.lspb(via_y, dur_y, tb_y)

        via_z = np.asarray([p1[2],p2[2],p3[2]])
        dur_z = np.asarray([dur,dur])
        tb_z = np.asarray([1,1,1])*ACC
        res_z = self.lspb(via_z, dur_z, tb_z)

        # Orientation
        via_ox = np.asarray([o1[0],o2[0],o3[0]])
        dur_ox = np.asarray([dur,dur])
        tb_ox = np.asarray([1,1,1])*ACC
        res_ox = self.lspb(via_ox, dur_ox, tb_ox)

        via_oy = np.asarray([o1[1],o2[1],o3[1]])
        dur_oy = np.asarray([dur,dur])
        tb_oy = np.asarray([1,1,1])*ACC
        res_oy = self.lspb(via_oy, dur_oy, tb_oy)

        via_oz = np.asarray([o1[2],o2[2],o3[2]])
        dur_oz = np.asarray([dur,dur])
        tb_oz = np.asarray([1,1,1])*ACC
        res_oz = self.lspb(via_oz, dur_oz, tb_oz)
        
        # Combine all three axis translations into one
        trans = np.ndarray(shape=(len(res_x[3]),6))

        count = 0
        for i in range(len(res_x[4])):
            trans[i] = np.array([res_x[4][i],res_y[4][i],res_z[4][i], res_ox[4][i],res_oy[4][i],res_oz[4][i]])
            if count < 10:
                print('trans: ', trans[i,2], ', res_z: ', res_z[4][i], ', time: ', res_z[3][i])
            count = count + 1
        # Reduce size
        
        #trans = trans[::100]
        
        traj1r = traj1[:-bsize1,:]
        traj2r = traj2[bsize2:,:]
        #trans = np.concatenate([traj1r, trans, traj2r])
        #trans = np.concatenate([traj1[:,:3], traj2[:,:3]])
        
        if plot:
            fig, (axx, axy, axz, axox, axoy, axoz) = plt.subplots(nrows=6, ncols=1,figsize=(10,8))
            # Plot points and generated line for each axis
            axx.plot(res_x[2],via_x,'*',res_x[3],trans[:,0],'.', label='x')
            axx.legend()
            axy.plot(res_y[2],via_y,'*',res_y[3],trans[:,1],'.', label='y')
            axy.legend()
            axz.plot(res_z[2],via_z,'*',res_z[3],trans[:,2],'.', label='z')
            axz.legend()
            axox.plot(res_z[2],via_z,'*',res_ox[3],trans[:,3],'.', label='z')
            axox.legend()
            axoy.plot(res_oy[2],via_oy,'*',res_oy[3],trans[:,4],'.', label='ry')
            axoy.legend()
            axoz.plot(res_oz[2],via_oz,'*',res_oz[3],trans[:,5],'.', label='rz')
            axoz.legend()
            
            
            fig2 = plt.figure(figsize=(10,5))
            ax2 = fig2.add_subplot(121, projection='3d')
            ax2.plot(trans[:,0], trans[:,1], trans[:,2],c='r')            
            ax2.scatter(traj1[0,0], traj1[0,1], traj1[0,2],c='cyan',marker='o')
            ax2.scatter(p1[0], p1[1], p1[2],c='r',marker='o')
            ax2.scatter(p2[0], p2[1], p2[2],c='b',marker='o')
            ax2.scatter(p3[0], p3[1], p3[2],c='g',marker='o')
            ax2.scatter(traj2[-1,0], traj2[-1,1], traj2[-1,2],c='yellow',marker='o')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_title('Line through Points')
            ax2.set_xlim(-1,1)
            ax2.set_ylim(-1,1)
            ax2.set_zlim(-1,1)

            ax3 = fig2.add_subplot(122, projection='3d')
            ax3.scatter(trans[:,0], trans[:,1], trans[:,2],c='r')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            ax3.set_title('Plot 3D')
            ax3.set_xlim(-1,1)
            ax3.set_ylim(-1,1)
            ax3.set_zlim(-1,1)
            plt.show()
        
        return trans
    
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