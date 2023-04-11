from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
import numpy as np

import random

NUM_ITTERATIONS = 10
newton_thres = 0.0
torque_thres = 0.0

# Function handler for button and slider
def on_button_click(event):
    exit()

def slider_newton(val):
    global newton_thres
    newton_thres = val

def slider_torque(val):
    global torque_thres
    torque_thres = val

if __name__ == "__main__":
    rtde_c = RTDEControl("192.168.1.131")
    rtde_r = RTDEReceive("192.168.1.131")

    # Create the figure and axis
    fig = plt.figure(constrained_layout=False)
    
    # Scale the plot to the size of the screen
    fig.set_size_inches(plt.figaspect(0.5))
    
    # Add white space between subplots
    fig.subplots_adjust(hspace=0.5)

    # Create subplots for translation and rotation
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    # Create the button and add it to the plot
    button_ax = plt.axes([0.05, 0.9, 0.03, 0.05])
    button = Button(button_ax, "Stop")
    button.on_clicked(on_button_click)

    # Define the slider
    axcolor = "lightgoldenrodyellow"
    newton_ax_slider = plt.axes([0.9, 0.58, 0.05, 0.3], facecolor=axcolor, label="Newton thresshold")
    newton_slider = Slider(newton_ax_slider, "Newton thresshold", 0.0, 10.0, valinit=newton_thres, orientation="vertical")
    newton_slider.on_changed(slider_newton)

    torque_ax_slider = plt.axes([0.9, 0.12, 0.05, 0.3], facecolor=axcolor)
    torque_slider = Slider(torque_ax_slider, "Torque thresshold", 0.0, 1.0, valinit=torque_thres, orientation="vertical")
    torque_slider.on_changed(slider_torque)

    # Define the x and y data
    x_data = np.linspace(0, NUM_ITTERATIONS, NUM_ITTERATIONS)
    y_data = np.linspace(0, 0, NUM_ITTERATIONS)

    # Set the x and y limits
    ax1.set_xlim(x_data[0], x_data[-1])
    ax2.set_xlim(x_data[0], x_data[-1])

    # Set the x and y labels
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Newton (N)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Torque (Nm)")

    # Set the title
    ax1.set_title("Newton")
    ax2.set_title("Torque")

    # Create the line objects
    legend = ["x", "y", "z"]
    lines_newton = [ax1.plot(x_data, y_data, label=legend[i])[0] for i in range(len(legend))]
    lines_torque = [ax2.plot(x_data, y_data, label=legend[i])[0] for i in range(len(legend))]

    # Set the legend
    ax1.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax2.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

    # Create a list of newton and torque data
    newton_data = [np.linspace(0, 0, NUM_ITTERATIONS) for _ in range(len(legend))]
    torque_data = [np.linspace(0, 0, NUM_ITTERATIONS) for _ in range(len(legend))]
    for i in range(NUM_ITTERATIONS):
        # Read the force data from the robot
        force_q = rtde_r.getActualTCPForce()
        #force_q = [1,2,3,0.1,0.2,0.3]

        # Copy the data to the newton and torque data lists
        for axis in range(len(legend)):
            newton_data[axis][i] = force_q[axis]
            torque_data[axis][i] = force_q[axis + 3]

        # Wait for 0.1 seconds
        time.sleep(0.1)

    # Get the average force and torque for each axis
    newton_avg = [sum(newton) / len(newton) for newton in newton_data]
    torque_avg = [sum(torque) / len(torque) for torque in torque_data]

    # Update the plot continuously with 10 hz
    i = 0
    while True:
        # Increment the x-data by one element and remove the first element
        x_data = np.append(x_data[1:], x_data[-1] + 1)

        # Set the x-axis limits
        ax1.set_xlim(x_data[0], x_data[-1])
        ax2.set_xlim(x_data[0], x_data[-1])

        # Get the current TCP force
        force_q = rtde_r.getActualTCPForce()
        #force_q = [1,2,3,0.1,0.2,0.3]
        #for i in range(len(force_q)):
        #    force_q[i] = force_q[i] * random.uniform(0.5, 1.5)
        
        # Remove the average force and torque from the data with a threshold
        for axis in range(len(legend)):
            # Remove the average force and torque
            force_q[axis] = force_q[axis] - newton_avg[axis]
            force_q[axis + 3] = force_q[axis + 3] - torque_avg[axis]
            print(force_q)
            # Check if the force and torque is within the threshold
            if -newton_thres <= force_q[axis] <= newton_thres:
                force_q[axis] = 0
            if -torque_thres <= force_q[axis + 3] <= torque_thres:
                force_q[axis + 3] = 0
        
        # Update the y-data of each newton line object
        for axis, line in enumerate(lines_newton):
            # Add the new data to the end of the y-data and remove the first element
            newton_data[axis] = np.append(newton_data[axis][1:], force_q[axis])

            # Update the line data
            line.set_ydata(newton_data[axis])
            line.set_xdata(x_data)
        
        # Update the y-data of each torque line object
        for axis, line in enumerate(lines_torque):
            # Add the new data to the end of the y-data and remove the first element
            torque_data[axis] = np.append(torque_data[axis][1:], force_q[axis + 3])

            # Update the line data
            line.set_ydata(torque_data[axis])
            line.set_xdata(x_data)

        # Set the y-axis limits
        ax1.set_ylim(min([min(newton) for newton in newton_data]) - 1, max([max(newton) for newton in newton_data]) + 1)
        ax2.set_ylim(min([min(torque) for torque in torque_data]) - 1, max([max(torque) for torque in torque_data]) + 1)

        plt.draw()
        plt.pause(0.1)

    plt.show()

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
    # "True". Try to set the async parameter to "False" to observe a default synchronous movement, which cannot be stopped
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
    # "True". Try to set the async parameter to "False" to observe a default synchronous movement, which cannot be stopped
    # by the stopL function due to the blocking behaviour.
    rtde_c.moveL(target, 0.25, 0.5, True)
    time.sleep(0.2)
    # Stop the movement before it reaches target
    rtde_c.stopL(0.5)

    # Move back to initial joint configuration
    rtde_c.moveL(init_q)

    # Stop the RTDE control script
    rtde_c.stopScript()
