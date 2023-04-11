from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import time

import matplotlib.pyplot as plt
import numpy as np

#pip3 install ur_rtde

#rtde_c = RTDEControl("192.168.1.131")
#rtde_r = RTDEReceive("192.168.1.131")
#init_q = rtde_r.getActualQ()

#force_q = rtde_r.getActualTCPForce()
#print("force_q: " ,force_q)

# Create the figure and axis
fig, ax = plt.subplots()
# Define the x and y data
x_data = np.linspace(0, 100, 100)
y_data = np.linspace(0, 0, 100)

lines = [ax.plot(x_data, np.sin(x_data))[0] for i in range(6)]

forces, = ax.plot(x_data, y_data)

for i in range(100):
    # Get the current TCP force
    #force_q = rtde_r.getActualTCPForce()
    
    # Shift the data to the left
    y_data[:-1] = y_data[1:]
    y_data[-1] = np.sin(2*np.pi*i/100)
    
    # Update the line data
    forces.set_ydata(y_data)

    plt.draw()
    plt.pause(0.001)

exit()


# convert rad to deg
print("in_q: Rad " ,init_q)
init_q_deg = [x * 180 / 3.141592653589793 for x in init_q]


print("Starting RTDE test script...")
print("init_q_deg: " ,init_q_deg)
# Target in the robot base
new_q = init_q[:]
new_q[0] += 0.10

# Move asynchronously in joint space to new_q, we specify asynchronous behavior by setting the async parameter to
# 'True'. Try to set the async parameter to 'False' to observe a default synchronous movement, which cannot be stopped
# by the stopJ function due to the blocking behaviour.
rtde_c.moveL(new_q, 1.05, 1.4, True)

time.sleep(0.2)
# Stop the movement before it reaches new_q
rtde_c.stopL(0.5)

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
