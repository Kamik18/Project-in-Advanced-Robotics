import swift
import spatialgeometry as sg
from spatialmath import SE3
from Blend.Blend import Trajectory
import roboticstoolbox as rtb
import numpy as np

# init environtment 
env = swift.Swift()
env.launch(realtime=True)

# npzfile = np.load('Python/Blend/sequencing-blending-main/examples/GripperExperiment/demos/pickplace_demo.npz')
#print(npzfile.files)
#print(len(npzfile['qt']))


# Create an obstacles
box = sg.Cuboid([2,2,-0.1], pose=SE3(0,0,0))
env.add(box)

UR5 = rtb.models.UR5()
env.add(UR5)

trajclass = Trajectory(box, UR5=UR5, box=box)

joint_pos = trajclass.jointPositions()
trans_pos = trajclass.transformationPositions()
UR5.q = joint_pos[3]
T = UR5.fkine(joint_pos[3])
print('T: \n', T)
env.step()
exit(1)
#traj1 = trajclass.makeTraj(joint_pos[0], joint_pos[1], 2)
#traj2 = trajclass.makeTraj(joint_pos[1], joint_pos[2], 2)
#traj3 = trajclass.makeTraj(joint_pos[2], joint_pos[3], 2)

print('joint_pos[0]: ', joint_pos[0])
print('joint_pos[1]: ', joint_pos[1])
trajclass.quad(joint_pos[0], joint_pos[1])

"""
traj1 = trajclass.makeTraj(trans_pos[0], trans_pos[1], 2)
traj2 = trajclass.makeTraj(trans_pos[1], trans_pos[2], 2)
traj3 = trajclass.makeTraj(trans_pos[2], trans_pos[3], 2)
traj4 = trajclass.makeTraj(trans_pos[3], trans_pos[4], 2)
traj5 = trajclass.makeTraj(trans_pos[4], trans_pos[5], 2)

trajclass.adddots(traj1, (1,0,0))
trajclass.adddots(traj2, (0,1,0))
trajclass.adddots(traj3, (0,0,1))
trajclass.adddots(traj4, (1,1,0))
trajclass.adddots(traj5, (1,0,1))
env.hold()
"""
#blend1 = trajclass.blendTraj(traj1, traj2, 1, printpath=True)
#blend2 = trajclass.blendTraj(traj2, traj3, 1, printpath=True)


