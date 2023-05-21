import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

class AdmittanceControl():
    def __init__(self, Kp: float = 10, Kd: float = 25, tr: float = 0.1, sample_time: float = 0.001) -> None:
        """Admittance control of the robot

        Args:
            - KP (float, optional): Kp is the proportional gain of the admittance control. Defaults to 10. 
            - KD (float, optional): Kd is the derivative gain of the admittance control.. Defaults to 20. 
            - tr (float, optional): Tr is the time it takes for the system to reach 63% of its final value. Defaults to 0.1.
            - sample_time (float, optional): Sample time in seconds. Defaults to 0.001.
        """
        # Set the gains
        self.kp: np.ndarray = np.diag([Kp, Kp, Kp])
        self.kd: np.ndarray = np.diag([Kd, Kd, Kd])

        # Calculate the mass matrix gain
        Natural_freq: float = 1 / tr
        mdx: float = Kp/(Natural_freq*Natural_freq)
        mdy: float = Kp/(Natural_freq*Natural_freq)
        mdz: float = Kp/(Natural_freq*Natural_freq)
        self.Mp: np.ndarray = np.diag([mdx, mdy, mdz])

        # Initial position, velocity and acceleration
        self.p: np.ndarray = np.zeros((3, 1))
        self.dp: np.ndarray = np.zeros((3, 1))
        self.ddp: np.ndarray = np.zeros((3, 1))

        # Initial angle, angular velocity and angular acceleration
        self.theta: np.ndarray = np.zeros((3, 1))
        self.dtheta: np.ndarray = np.zeros((3, 1))
        self.ddtheta: np.ndarray = np.zeros((3, 1))

        self.sample_time: float = sample_time

    def Translation(self, wrench: np.ndarray, p_ref: np.ndarray) -> tuple:
        """Translation control of the robot

        Args:
            wrench (np.ndarray): 3x1 wrench vector (Fx, Fy, Fz)
            p_desired (np.ndarray): 3x1 desired position vector (x, y, z)

        Returns:
            tuple: p_c, p, dp, ddp. p_c is the compliance frame position, p is the current position, dp is the current velocity, ddp is the current acceleration
        """
        # Reshape the input vectors
        wrench: np.ndarray = np.array(wrench).reshape((3, 1))
        p_ref: np.ndarray = np.array(p_ref).reshape((3, 1))

        # Calculate the inputs
        sum_block: np.ndarray = wrench + \
            (-self.kd @ self.dp) # + (-self.kp @ self.p)
        # Calculate the acceleration, velocity and position
        self.ddp: np.ndarray = np.linalg.inv(self.Mp) @ sum_block
        self.dp: np.ndarray = self.dp + (self.ddp * self.sample_time)
        self.p: np.ndarray = self.p + (self.dp * self.sample_time)

        # Add the desired position to the current position
        p_c: np.ndarray = self.p + p_ref

        # Return the position, velocity, acceleration, and compliance frame position
        return list(p_c), list(self.p), list(self.dp), list(self.ddp)

    def Rotation_euler(self, wrench: np.ndarray, theta_ref: np.ndarray) -> tuple:
        """Rotation control of the robot in euler angles

        Args:
            wrench (np.ndarray): 3x1 wrench vector (Fx, Fy, Fz)
            theta_desired (np.ndarray): 3x1 desired position vector (x, y, z)

        Returns:
            tuple: theta_c, theta, dtheta, ddtheta. theta_c is the compliance frame position, theta is the current position, dtheta is the current velocity, ddtheta is the current acceleration
        """
        # Calculate the inputs
        sum_block: np.ndarray = wrench + \
            (-self.kd @ self.dtheta) + (-self.kp @ self.theta)
        # Calculate the angular acceleration, angular velocity and angle
        self.ddtheta: np.ndarray = np.linalg.inv(self.Mp) @ sum_block
        self.dtheta: np.ndarray = self.dtheta + self.ddtheta * self.sample_time
        self.theta: np.ndarray = self.theta + self.dtheta * self.sample_time

        # Add the desired angle to the current angle
        theta_c: np.ndarray = self.theta + theta_ref

        # Return the angle, angular velocity, angular acceleration, and compliance frame angle
        return theta_c, self.theta, self.dtheta, self.ddtheta
    


class AdmittanceControlQuaternion():
    def __init__(self, Kp: float = 10, Kd: float = 25, tr: float = 0.1, sample_time: float = 0.001) -> None:      
        # Initialize the variables
        self.sample_time: float = sample_time
       
        # Set the gains
        self.kp: np.ndarray = np.diag([Kp, Kp, Kp])
        self.kd: np.ndarray = np.diag([Kd, Kd, Kd])

        # Calculate the mass matrix gain
        Natural_freq: float = 1 / tr
        mdx: float = self.kp[0, 0]/(Natural_freq*Natural_freq)
        mdy: float = self.kp[1, 1]/(Natural_freq*Natural_freq)
        mdz: float = self.kp[2, 2]/(Natural_freq*Natural_freq)
        self.Mp: np.ndarray = np.diag([mdx, mdy, mdz])

        self.q: np.ndarray = np.zeros((4), dtype=np.float64)
        self.w: np.ndarray = np.zeros((3), dtype=np.float64)
        self.dw: np.ndarray = np.zeros((3))

    def Rotation_Quaternion(self, wrench, q_ref) -> tuple:
        y = self.fcn(self.q, self.kp)
        
        sum_block = wrench - (self.kd @ self.w) # - y

        self.dw = np.linalg.inv(self.Mp) @ sum_block
        self.w = self.w + self.dw * self.sample_time

        self.q = self.w2q(self.w,self.sample_time, self.q)
    
        # Add the desired angle to the current angle
        theta_c = np.zeros((4))
        '''
        ref_q = Quaternion(q_ref)
        qq = Quaternion(self.q)
        qqq = qq * ref_q
        theta_c[0] = qqq[0]
        theta_c[1] = qqq[1]
        theta_c[2] = qqq[2]
        theta_c[3] = qqq[3]
        '''
        
        # Return the angle, angular velocity, angular acceleration, and compliance frame angle
        return theta_c, self.w, self.dw
    

    def skew_symmetric_matrix(self,v):
        
        aa = float(v[2])
        bb = float(v[1])
        cc = float(v[0])
        yy = np.array([[0, -aa, bb], 
                       [aa, 0, -cc],
                       [-bb, cc, 0],
                       ])
        return yy
    
    def w2q(self, w, T_s, Q_prev):
        # if Q_pre is 0, then the initial quaternion is [1,0,0,0]
        if 0 == Q_prev[0] and 0 == Q_prev[1] and 0 == Q_prev[2] and 0 == Q_prev[3]:
            Q_prev = np.array([1,0,0,0])
            
        r = (T_s/2) * w
        q0 = Quaternion(0, r[0], r[1], r[2])
        q_new = Quaternion.exp(q0) * Quaternion(Q_prev)
        y = np.array([q_new.w, q_new.x, q_new.y, q_new.z])
        
        return y

    def fcn(self, q, K_0):
        Eta = q[0]
        Epsilon = q[1:4]
        
        S = np.array([[0, -Epsilon[2], Epsilon[1]],
                    [Epsilon[2], 0, -Epsilon[0]],
                    [-Epsilon[1], Epsilon[0], 0]])
        E = Eta * np.eye(3) - S
        K = 2 * np.transpose(E) @ K_0
        y = K @ np.transpose(Epsilon)
        
        return y


def SimulateQuaternion():
    
    time: np.ndarray = np.linspace(0, 30, 30 * 1000 + 1)
    dt: float = np.gradient(time)[0]

    admittance_control_q = AdmittanceControlQuaternion(sample_time=dt,kp=11.11,kd=4.667, tr=0.1)
    q_ref = np.array([1, 0, 0, 0])

    temp_p = []
    temp_dp = []
    temp_ddp = []
    
    # Run the simulation
    for i in range(len(time)):
        if (15000 < i < 20000):
            wrench_ = np.array([1, 0.5, 1])
        else:
            wrench_ = np.array([0,0,0])
        
        p, dp, ddp = admittance_control_q.Rotation_Quaternion(
           wrench = wrench_, q_ref=q_ref)
        temp_p.append(p)
        temp_dp.append(dp)
        temp_ddp.append(ddp)

    temp_p = np.array(temp_p)
    temp_dp = np.array(temp_dp)
    temp_ddp = np.array(temp_ddp)

    print("temp_p shape ", temp_p.shape)
    print("temp_dp shape ", temp_dp.shape)
    print("temp_ddp shape ", temp_ddp.shape)


    fig, axs = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(
        10, 10), constrained_layout=True, dpi=100)
    fig.suptitle('Admittance Control Oriention', fontsize=16)
    fig.set_constrained_layout_pads(
        w_pad=0.1, h_pad=0.1, hspace=0.1, wspace=0.1)
    axs[0].plot(time, temp_p, label=["x", "y", "z", "w"])
    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('Quaternion, [rad]')
    axs[0].legend(loc='upper right', ncol=3, fontsize=8)
    axs[0].set_title('q')
    
    axs[1].plot(time, temp_dp, label=["x", "y", "z"])
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Omega, [rad/s]')
    axs[1].legend(loc='upper right', ncol=3, fontsize=8)
    axs[1].set_title('Omega')
    
    axs[2].plot(time, temp_ddp, label=["x", "y", "z"])
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('DOmega, [rad/s^2]')
    axs[2].legend(loc='upper right', ncol=3, fontsize=8)
    axs[2].set_title('omega_dot')
    plt.show()
    


def create_position_reference(time: np.ndarray) -> np.ndarray:
    """Create a position reference vector

    Args:
        time (np.ndarray): Time vector

    Returns:
        np.ndarray: Position reference vector
    """
    # Create the position reference vector
    ref_x = 0.3 * np.sin(time / 3)
    ref_y = 0.3 * np.sin(time / 3) * np.cos(time / 3)
    ref_z = 0.1 * np.sin(time)
    ref_p: np.ndarray = np.array([ref_x, ref_y, ref_z])

    # Return the position reference vector
    return ref_p

def create_orientation_reference(time: np.ndarray) -> np.ndarray:
    """Create a orientation reference vector

    Args:
        time (np.ndarray): Time vector

    Returns:
        np.ndarray: Orientation reference vector
    """
    # Create the orientation reference vector
    ref_x = np.zeros((len(time)))
    ref_y = np.zeros((len(time)))
    ref_z = 0.1 * np.sin(time)
    ref: np.ndarray = np.array([ref_x, ref_y, ref_z])

    # Return the orientation reference vector
    return ref

def create_wrench(time: np.ndarray) -> np.ndarray:
    """Create a wrench vector

    Args:
        time (np.ndarray): Time vector

    Returns:
        np.ndarray: Wrench vector
    """
    # Create the wrench vector
    wrench_x: np.ndarray = np.ones((len(time))) * 0.5
    wrench_y: np.ndarray = np.ones((len(time)))
    wrench_z: np.ndarray = np.ones((len(time))) * -1.0
    wrench: np.ndarray = np.array([wrench_x, wrench_y, wrench_z])

    # Return the wrench vector
    return wrench


def simulate_translation(time: np.ndarray, ref_p: np.ndarray, wrench: np.ndarray) -> None:
    # Create the admittance controller
    admittance_control: AdmittanceControl = AdmittanceControl(
        Kp=11.11, Kd=4.667, tr=0.1, sample_time=dt)

    # Create the lists to store the results
    p_list: np.ndarray = np.zeros((3, len(time)))
    dp_list: np.ndarray = np.zeros((3, len(time)))
    ddp_list: np.ndarray = np.zeros((3, len(time)))
    p_c_list: np.ndarray = np.zeros((3, len(time)))

    # Run the simulation
    for i in range(len(time)):
        p_c, p, dp, ddp = admittance_control.Translation(
            wrench=wrench[:, [i]], p_ref=ref_p[:, [i]])
        p_c_list[:, [i]] = p_c
        p_list[:, [i]] = p
        dp_list[:, [i]] = dp
        ddp_list[:, [i]] = ddp

    # Plot the results
    fig, axs = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(
        10, 10), constrained_layout=True, dpi=100)
    fig.suptitle('Admittance Control', fontsize=16)
    fig.set_constrained_layout_pads(
        w_pad=0.1, h_pad=0.1, hspace=0.1, wspace=0.1)
    axs[0].plot(time, p_c_list.T, label=["x", "y", "z"])
    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('pos, [m]')
    axs[0].legend(loc='upper right', ncol=3, fontsize=8)
    axs[0].set_title('Position Compliance Frame (p_c)')

    axs[1].plot(time, p_list.T, label=["x", "y", "z"])
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('pos, [m]')
    axs[1].legend(loc='upper right', ncol=3, fontsize=8)
    axs[1].set_title('Position (p)')

    axs[2].plot(time, dp_list.T, label=["x", "y", "z"])
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('vel, [m/s]')
    axs[2].legend(loc='upper right', ncol=3, fontsize=8)
    axs[2].set_title('Velocity (dp)')

    axs[3].plot(time, ddp_list.T, label=["x", "y", "z"])
    axs[3].set_xlabel('t (s)')
    axs[3].set_ylabel('acc, [m/s^2]')
    axs[2].legend(loc='upper right', ncol=3, fontsize=8)
    axs[3].set_title('Acceleration (ddp)')
    plt.show()

def simulate_rotation(time: np.ndarray, ref_theta: np.ndarray, wrench: np.ndarray) -> None:
    # Create the admittance controller
    admittance_control: AdmittanceControl = AdmittanceControl(
        Kp=11.11, Kd=4.667, tr=0.1, sample_time=dt)

    # Create the lists to store the results
    p_list: np.ndarray = np.zeros((3, len(time)))
    dp_list: np.ndarray = np.zeros((3, len(time)))
    ddp_list: np.ndarray = np.zeros((3, len(time)))
    p_c_list: np.ndarray = np.zeros((3, len(time)))

    # Run the simulation
    for i in range(len(time)):
        p_c, p, dp, ddp = admittance_control.Rotation_euler(
            wrench=wrench[:, [i]], theta_ref=ref_theta[:, [i]])
        p_c_list[:, [i]] = p_c
        p_list[:, [i]] = p
        dp_list[:, [i]] = dp
        ddp_list[:, [i]] = ddp

    # Plot the results
    fig, axs = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(
        10, 10), constrained_layout=True, dpi=100)
    fig.suptitle('Admittance Control', fontsize=16)
    fig.set_constrained_layout_pads(
        w_pad=0.1, h_pad=0.1, hspace=0.1, wspace=0.1)
    axs[0].plot(time, p_c_list.T, label=["x", "y", "z"])
    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('Theta, [rad]')
    axs[0].legend(loc='upper right', ncol=3, fontsize=8)
    axs[0].set_title('theta Compliance Frame (theta_c)')

    axs[1].plot(time, p_list.T, label=["x", "y", "z"])
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Theta, [rad]')
    axs[1].legend(loc='upper right', ncol=3, fontsize=8)
    axs[1].set_title('theta (theta)')

    axs[2].plot(time, dp_list.T, label=["x", "y", "z"])
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Omega, [rad/s]')
    axs[2].legend(loc='upper right', ncol=3, fontsize=8)
    axs[2].set_title('Omega (dtheta)')

    axs[3].plot(time, ddp_list.T, label=["x", "y", "z"])
    axs[3].set_xlabel('t (s)')
    axs[3].set_ylabel('omega, [rad/s^2]')
    axs[2].legend(loc='upper right', ncol=3, fontsize=8)
    axs[3].set_title('omega_dot (ddtheta)')
    plt.show()


if __name__ == "__main__":
    # Time vector
    #time: np.ndarray = np.linspace(0, 30, 30 * 1000 + 1)
    dt: float = 0.001

    # # Create the reference trajectory and the wrench vector
    # ref_p: np.ndarray = create_position_reference(time=time)
    # ref_theta: np.ndarray = create_orientation_reference(time=time)
    # wrench: np.ndarray = create_wrench(time=time)

    # # Run the simulations
    # simulate_translation(time=time, ref_p=ref_p, wrench=wrench)
    # simulate_rotation(time=time, ref_theta=ref_theta, wrench=wrench)

    # SimulateQuaternion()


    # Simulate the translation

    # admittance_control: AdmittanceControl = AdmittanceControl(
    #     Kp=10, Kd=20, tr=0.1, sample_time=dt)

    # # Create the lists with unknown size to store the results
    # rtde_r = RTDEReceive("192.168.1.131")
    
    # p_list: np.ndarray = np.array()

    # # Run the simulation
    # while True:
        
    #     force_q = rtde_r.getActualTCPForce()
    #     force = np.array([force_q[0], force_q[1], force_q[2]])
    #     ref_p = np.array([0.5, 0.5, 0.5])

    #     p_c,_ ,_, _ = admittance_control.Translation(
    #         wrench=force, p_ref=ref_p[:, [i]])
        
    #     p_list = np.append(p_list, p_c)
    #     print("pos: ",p_c)



