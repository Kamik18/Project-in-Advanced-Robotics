import numpy as np
import matplotlib.pyplot as plt


class AdmittancePosition():

    def __init__(self, KP=10, KD=20, tr=0.1):

        self.kp = np.diag([KP, KP, KP])
        self.kd = np.diag([KD, KD, KD])

        Natural_freq = 1 / tr
        mdx = self.kp[0, 0]/(Natural_freq*Natural_freq)
        mdy = self.kp[1, 1]/(Natural_freq*Natural_freq)
        mdz = self.kp[2, 2]/(Natural_freq*Natural_freq)

        self.Mp = np.diag([1.9, 1, 1])

        self.p = np.zeros((3, 1))
        self.dp = np.zeros((3, 1))
        self.ddp = np.zeros((3, 1))

        self.ts = 0.001

    # , p_des, p_dot_des, p_ddot_des, dt, Wrench):
    def AdmittanceControl(self, p_input: np.ndarray['float64'], p_desired: np.ndarray['float64']) -> np.ndarray['float64']:
        sum_block = p_input + (-self.kd @ self.dp) + (-self.kp @ self.p)
        self.ddp = np.linalg.inv(self.Mp) @ sum_block
        self.dp = self.dp + self.ddp * self.ts
        self.p = self.p + self.dp * self.ts

        # Return the position, velocity and acceleration
        return self.p, self.dp, self.ddp


if __name__ == "__main__":
    Ad_pos = AdmittancePosition(KP=11.11, KD=4.667)

    # line space
    t = np.linspace(0, 30, 1000)
    dt = np.gradient(t)

    ref_x = 0.3 * np.sin(t/3)
    ref_y = 0.3 * np.sin(t/3) * np.cos(t/3)
    ref_z = 0.1 * np.sin(t)

    ref_p = np.array([ref_x, ref_y, ref_z])
    ref_p_dot = np.zeros((3, 1))
    ref_p_dot_dot = np.zeros((3, 1))

    wrench = np.ones((3, 1))

    p_desired = np.zeros((3, 1))
    p_desired[0] = 491
    p_desired[1] = 492
    p_desired[2] = 493
    p_input = np.zeros((3, 1))
    p_input[0] = 991
    p_input[1] = 992
    p_input[2] = 993

    time = np.linspace(0, 30, 30 * 1000)
    p_list = np.array((3, len(time)))
    dp_list = np.array((3, len(time)))
    ddp_list = np.array((3, len(time)))
    

    for it in range(len(time)):
        p, dp, ddp = Ad_pos.AdmittanceControl(p_input=p_input, p_desired=p_desired)
        p_list[it] = p
        dp_list[it] = dp
        ddp_list[it] = ddp
    
    
    
    # p, dp, ddp = Ad_pos.AdmittanceControl(p_des= ref_p, p_dot_des=ref_p_dot, p_ddot_des= ref_p_dot_dot , dt=0.01, Wrench=wrench)

    fig1, axs = plt.subplots(3, 1, sharex=True)
    fig1.suptitle('Admittance Control', fontsize=16)

    axs[0].plot(time, np.array(p_list), label=["x", "y", "z"])
    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('pos')

    # axs[1].plot(time, np.array(dp_list).T, label=["x", "y", "z"])
    # axs[1].set_xlabel('t (s)')
    # axs[1].set_ylabel('vel')

    # axs[2].plot(time, np.array(ddp_list).T, label=["x", "y", "z"])
    # axs[2].set_xlabel('t (s)')
    # axs[2].set_ylabel('acc')
    # axs[2].legend()
    plt.show()
