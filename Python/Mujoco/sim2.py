"""Starts Simulation for UR5e Robot System model for MuJoCo."""
import mujoco_py
import math
import os

print(os.getcwd())

model = mujoco_py.load_model_from_xml("./Python/Mujoco/universal_robots_ur5e/scene.xml")
sim = mujoco_py.MjSim(model)

# Set camera position
camera_pos = [1.0, 1.0, 1.0]  # x, y, z
sim.viewer.cam.lookat[0] = camera_pos[0]
sim.viewer.cam.lookat[1] = camera_pos[1]
sim.viewer.cam.lookat[2] = camera_pos[2]

# Set camera orientation
camera_yaw = math.radians(45)  # yaw angle in radians
camera_pitch = math.radians(-30)  # pitch angle in radians
sim.viewer.cam.euler[0] = camera_pitch
sim.viewer.cam.euler[1] = camera_yaw
sim.viewer.cam.euler[2] = 0

# Render the scene with the new camera settings
sim.render()

# Close the MuJoCo viewer
sim.viewer.close()

