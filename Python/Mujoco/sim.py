"""Starts Simulation for UR5e Robot System model for MuJoCo."""

import time
from threading import Thread

import glfw
import mujoco
import numpy as np
from pathlib import Path


class Demo:

    qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    K = [600.0, 600.0, 600.0, 30.0, 30.0, 30.0]
    height, width = 1000, 1000  # Rendering window resolution.
    fps = 30  # Rendering framerate.

    def __init__(self) -> None:
        # filename = Path("franka_emika_panda/scene.xml.xml")
        self.model = mujoco.MjModel.from_xml_path("./Example_exercise/universal_robots_ur5e/scene.xml")
        self.data = mujoco.MjData(self.model)
        self.cam = mujoco.MjvCamera()
        #self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = 0
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.run = True
        self.gripper(False)
        #for i in range(1, 6):
        #    self.data.joint(f"joint{i}").qpos = self.qpos0[i - 1]
        #mujoco.mj_forward(self.model, self.data)

    def gripper(self, open=True):
        pass
        #self.data.actuator("actuator8").ctrl = (0.04, 0)[not open]
        #self.data.actuator("actuator8").ctrl = (0.04, 0)[not open]

    def control(self, xpos_d, xquat_d):
        xpos = self.data.body("wrist_3_link").xpos
        xquat = self.data.body("wrist_3_link").xquat
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        #bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ur5_hand")
        #mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)

        res = np.zeros(3)
        mujoco.mju_subQuat(res, xquat, xquat_d)
        mujoco.mju_rotVecQuat(res, res, xquat)

        J = np.concatenate((jacp, jacr))
        v = J @ self.data.qvel
        #self.somecoolmethod(J, v, error,'shoulder_pan_joint', 'shoulder_pan')
        #self.somecoolmethod(J, v, error,'shoulder_lift_joint', 'shoulder_lift')
        #self.somecoolmethod(J, v, error,'elbow_joint', 'elbow')
        #self.somecoolmethod(J, v, error,'wrist_1_joint', 'wrist_1')
        #self.somecoolmethod(J, v, error,'wrist_2_joint', 'wrist_2')
        #self.somecoolmethod(J, v, error,'wrist_3_joint', 'wrist_3')

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
        while not glfw.window_should_close(window):
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
