
import numpy as np

import time

import matplotlib.pyplot as plt

import winsound

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

def read_demo_files(filename, skip_lines=4):
    
    """Read demo files.

    Args:
        filename (string): folder directory
        skip_lines (int, optional): Skip every x lines. Defaults to 4.

    Returns:
        np.array: demo data and demo joint angles data
    """
    tuples =[]
    with open(filename , "r") as f:
        for i, line in enumerate(f):
            # Check if the line number is a multiple of skip_lines-1
            if i % skip_lines == skip_lines-1:
                values = tuple(map(float, line.split(',')))
                tuples.append(values)
    demo = np.array(tuples)


    return demo

#print('init_q: ', init_q)


import numpy as np

# Open file for reading
def read_exp_dat(file):
    with open(file, 'r') as f:
        # Read lines from file
        lines = f.readlines()

    # Remove newline characters from each line
    lines = [line.strip() for line in lines]

    # Convert lines to numpy array
    data = np.array([np.fromstring(line[1:-1], sep=', ') for line in lines])
    return data


TIME:int = 0.02
# Read data from file
data_dmp= read_exp_dat('Records\experiments\GMM_B_DMP_A/acc.txt')
data_gmm = read_exp_dat('Records\experiments/DMP_B_GMM_A/acc.txt')
num_points = data_dmp.shape[0]
time = np.arange(num_points) * 0.02


fig, axs = plt.subplots(6, 1,figsize=(6.4, 8.6))
# plot data
for i in range(6):
    axs[i].plot(time, data_dmp[:, i], label='No-blend', color='red' , linewidth=1)
    if data_gmm.shape[0] > 0:
         axs[i].plot(time[:data_gmm.shape[0]], data_gmm[:, i], label='blend', color='blue', linewidth=1)
    axs[i].set_ylabel(f'$q_{i+1} [rad/s^2]$')
    
    # Hide the x-axis tick labels for the first five subplots
    if i < 5:
        axs[i].set_xticklabels([])

# Set the x-axis label and legend for the last subplot
axs[5].set_xlabel('Time $[s]$')
axs[1].legend(loc='upper right')



for i in range(6):
    axs[i].grid(False)
    axs[i].set_facecolor('white')

    # Set the color, linewidth, and edgecolor of the spines to draw a square box around each subplot
    for spine in axs[i].spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.5)
        spine.set_edgecolor('black')

fig.align_ylabels()
#plt.savefig('Records\experiments\Fig/blend.pdf', bbox_inches='tight')
plt.show()







exit(1)
rtde_c = RTDEControl("192.168.1.131")
rtde_r = RTDEReceive("192.168.1.131")


print("Starting RTDE test script...")
# Target in the robot base
frequency = 440  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
time.sleep(3)
VELOCITY = 0.2
ACCELERATION = 0.2
BLEND = 0

file_name = 'Python/DMP/Out/DMP_Joint_Up_B_3_new_goal_pos.txt'
data = read_demo_files(file_name, skip_lines=3)


speed = 1
for q in data: #combined_blend:
    rtde_c.servoJ(q, 0,0, speed, 0.2, 100)
    time.sleep(0.2)
    print('q: ', q)
    

    


rtde_c.speedStop()
time.sleep(0.2)
# Stop the movement before it reaches new_q
#rtde_c.stopL(0.5)
print("Stopped movement")
rtde_c.stopScript()
rtde_c.disconnect()
print("Disconnected from RTDE")
exit(1)

# Target in the Z-Axis of the TCP
target = rtde_r.getActualTCPPose()
print("target: " ,target)
target[2] += 0.10

# Move asynchronously in cartesian space to target, we specify asynchronous behavior by setting the async parameter to
# 'True'. Try to set the async parameter to 'False' to observe a default synchronous movement, which cannot be stopped
# by the stopL function due to the blocking behaviour.
rtde_c.moveL(target, 0.25, 0.5, True)
time.sleep(0.2)
# Stop the movement before it reaches target
rtde_c.stopL(0.5)

# Move back to initial joint configuration
rtde_c.moveL(init_q)

# Stop the RTDE control script
rtde_c.stopScript()
