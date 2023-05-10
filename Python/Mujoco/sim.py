"""Starts Simulation for UR5e Robot System model for MuJoCo."""

import time
from threading import Thread

import glfw
import mujoco
import numpy as np
from pathlib import Path
from spatialmath.base import r2q
from spatialmath import SO3, SE3, UnitQuaternion
import roboticstoolbox as rtb


class Demo:
    qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    K = [600.0, 600.0, 600.0, 30.0, 30.0, 30.0]
    height, width = 1000, 1000  # Rendering window resolution.
    fps = 30  # Rendering framerate.

    def __init__(self) -> None:
        # filename = Path("franka_emika_panda/scene.xml.xml")
        self.model = mujoco.MjModel.from_xml_path("./Python/Mujoco/universal_robots_ur5e/scene.xml")
        self.data = mujoco.MjData(self.model)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[0] = 0
        self.cam.lookat[1] = 0
        self.cam.lookat[2] = 0
        self.cam.elevation = -30
        self.cam.azimuth = 300
        self.cam.distance = 3
        #self.cam.fixedcamid = 0
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.run = True
        self.gripper(False)
        #for i in range(1, 6):
        #    self.data.joint(f"joint{i}").qpos = self.qpos0[i - 1]
        #mujoco.mj_forward(self.model, self.data)
        
        q0 =  np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
        up_b_j: np.ndarray = np.loadtxt("./Records/Up_B/1/record_j.txt", delimiter=',', skiprows=0)
        down_b_j: np.ndarray = np.loadtxt("./Records/Down_B/1/record_j.txt", delimiter=',', skiprows=0)
        up_a_j: np.ndarray = np.loadtxt("./Records/Up_A/1/record_j.txt", delimiter=',', skiprows=0)
        down_a_j: np.ndarray = np.loadtxt("./Records/Down_A/1/record_j.txt", delimiter=',', skiprows=0)
        #downsample
        up_b_j = up_b_j[::50]
        down_b_j = down_b_j[::50]
        up_a_j = up_a_j[::50]
        down_a_j = down_a_j[::50]
        home_to_start = self.makeTraj(q0, up_b_j[0])
        connectionTraj = self.makeTraj(down_b_j[-1], up_a_j[0])
        return_to_start = self.makeTraj(down_a_j[-1], q0)
        self.combined_traj = np.concatenate([home_to_start.q, up_b_j, down_b_j, connectionTraj.q, up_a_j, down_a_j, return_to_start.q])

    def gripper(self, open=True):
        pass
        #self.data.actuator("actuator8").ctrl = (0.04, 0)[not open]
        #self.data.actuator("actuator8").ctrl = (0.04, 0)[not open]
    
    def makeTraj(self, start_pos, end_pos, duration : int = 2):
        sps = 20
        time_vec = np.linspace(0, duration, sps*duration)

        joint_pos_start = start_pos
        joint_pos_end = end_pos
        return rtb.jtraj(joint_pos_start, joint_pos_end, time_vec)    

    def moveq(self, arr):
        self.data.joint('shoulder_pan_joint').qpos = arr[0]
        self.data.joint('shoulder_lift_joint').qpos = arr[1]
        self.data.joint('elbow_joint').qpos = arr[2]
        self.data.joint('wrist_1_joint').qpos = arr[3]
        self.data.joint('wrist_2_joint').qpos = arr[4]
        self.data.joint('wrist_3_joint').qpos = arr[5]
        

    def control(self, xpos_d, xquat_d):
        ## send to simulator 
        #up_b_j: np.ndarray = np.loadtxt("./Records/Up_B/1/record_j.txt", delimiter=',', skiprows=0)

        for qs in self.combined_traj:
            self.moveq(qs)
            mujoco.mj_step(self.model, self.data)
            time.sleep(0.05)#1e-3)
        

    def step(self) -> None:
        xpos0 = self.data.body("wrist_3_link").xpos.copy()
        xpos_d = xpos0
        xquat0 = self.data.body("wrist_3_link").xquat.copy()
        
        self.control(xpos_d, xquat0)
        mujoco.mj_step(self.model, self.data)
        time.sleep(1e-3)

    def render(self) -> None:
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 8)
        window = glfw.create_window(self.width, self.height, "Demo", None, None)
        glfw.make_context_current(window)
        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_100
        )
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        while glfw.get_key(window,glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(window):
            w, h = glfw.get_framebuffer_size(window)
            viewport.width = w
            viewport.height = h
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                opt,
                pert,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene,
            )
            mujoco.mjr_render(viewport, self.scene, self.context)
            time.sleep(1.0 / self.fps)
            glfw.swap_buffers(window)
            glfw.poll_events()
        self.run = False
        glfw.terminate()

    def start(self) -> None:
        step_thread = Thread(target=self.step)
        step_thread.start()
        self.render()


if __name__ == "__main__":
    Demo().start()
