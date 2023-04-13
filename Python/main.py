from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_io import RTDEIOInterface as RTDEIO
from rtde_receive import RTDEReceiveInterface as RTDEReceive

from Gripper.RobotiqGripper import RobotiqGripper
from Admittance.Admittance_Control_position import AdmittanceControl

import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
import numpy as np
from scipy.signal import butter, lfilter
import os

import atexit



NUM_ITTERATIONS:int = 10
newton_thres:float = 0.5
torque_thres = 0.0

ACCELERATION:float = 1.0

IP = "192.168.1.131"


TIME:float = 0.002
FILTER:str = "MEAN" # THRESHOLD, BUTTERWORTH, MEAN

# Define the sampling rate and cutoff frequency
fs = 1 / TIME  # sampling rate (Hz)
fc = 100   # cutoff frequency (Hz)

# Define the order of the filter
order = 4

# Design the Butterworth filter coefficients
nyquist_freq = 0.5 * fs
cutoff_norm = fc / nyquist_freq
b, a = butter(order, cutoff_norm, btype='low')

# Function handler for button and slider
def on_button_click(event):
    rtde_c.stopScript()
    exit()

def slider_newton(val):
    global newton_thres
    newton_thres = val

def slider_torque(val):
    global torque_thres
    torque_thres = val

def goodbye(rtde_c:RTDEControl, rtde_r:RTDEReceive, gripper:RobotiqGripper):
    # Robot
    try:
        # Stop the robot controller
        rtde_c.speedStop()
        rtde_c.stopScript()
        rtde_c.disconnect()

        # Disconnect the receiver
        rtde_r.disconnect()
    except:
        print("Robot failed to terminate")

    # Gripper
    try:
        position:int = 0 # 0-255 - Low value is open, high value is closed 
        speed:int = 50 # 0-255
        force:int = 10 # 0-255
        gripper.move_and_wait_for_pos(position=position, speed=speed, force=force)
        gripper.disconnect()
    except:
        print("Gripper failed to terminate")

    print("Program terminated")

def initialize_plot():
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
    
    # Return the lines
    return lines_newton, lines_torque


def Grippper_example():
    position:int = 0 # 0-255 - Low value is open, high value is closed 
    speed:int = 50 # 0-255
    force:int = 10 # 0-255
    gripper.move_and_wait_for_pos(position=position, speed=speed, force=force)
    print(f"Pos: {str(gripper.get_current_position()): >3}  "
        f"Open: {gripper.is_open(): <2}  "
        f"Closed: {gripper.is_closed(): <2}  ")


if __name__ == "__main__":
    # Create control and receive interface for the robot
    rtde_c = RTDEControl(IP)
    rtde_r = RTDEReceive(IP)

    # Create a Robotiq gripper
    gripper = RobotiqGripper()
    gripper.connect(IP, 63352)
    gripper.activate()

    # Add exit handler
    atexit.register(goodbye, rtde_c, rtde_r, gripper)

    # Zero Ft sensor
    rtde_c.zeroFtSensor()
    time.sleep(0.2)

    # Plot the output
    #lines_newton, lines_torque = initialize_plot()

    # Admintance control
    admittance_control: AdmittanceControl = AdmittanceControl(
        Kp=10, Kd=25, tr=0.3, sample_time=TIME)
    
    # Update the plot continuously with 10 hz
    i = 0
    duration = 0.5 # Sec
    freq = 440 # Hz
    os.system(f"play -nq -t alsa synth {duration} sine {freq}")

    # Store the newton measurements
    newton_data = [np.zeros((NUM_ITTERATIONS, 1)) for _ in range(3)]
    
    exit() 
    # Main loop
    while True:
        t_start = rtde_c.initPeriod()
        
        # Get the current TCP force
        force_q = rtde_r.getActualTCPForce()
        
        # Remove the average force and torque from the data with a threshold
        for axis in range(3):
            # Add the new data to the end of the y-data and remove the first element
            newton_data[axis] = np.append(newton_data[axis][1:], force_q[axis])
            
            if FILTER == "THRESHOLD":
                # Check if the force and torque is within the threshold
                if -newton_thres <= force_q[axis] <= newton_thres:
                    force_q[axis] = 0
                if -torque_thres <= force_q[axis + 3] <= torque_thres:
                    force_q[axis + 3] = 0
            elif FILTER == "BUTTERWORTH":                
                # Apply the Butterworth filter to the input signal
                y = lfilter(b, a, newton_data[axis])
                force_q[axis] = y[-1]
            elif FILTER == "MEAN":
                # Find the avverage of the measurements
                force_q[axis] = np.mean(newton_data[axis])

                # Check if the force and torque is within the threshold
                if -newton_thres <= force_q[axis] <= newton_thres:
                    force_q[axis] = 0

                #running_mean = np.convolve(newton_data[axis], np.ones(NUM_ITTERATIONS) / NUM_ITTERATIONS, mode='valid')
                #force_q[axis] = running_mean[-1]

        # Increment the x-data by one element and remove the first element
        """
        x_data = np.append(x_data[1:], x_data[-1] + 1)

        # Set the x-axis limits
        ax1.set_xlim(x_data[0], x_data[-1])
        ax2.set_xlim(x_data[0], x_data[-1])
        
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
        """

        # Find the translational velocity with the andmittance control
        _, p, dp, ddp = admittance_control.Translation(
           wrench=force_q[0:3], p_ref=[0, 0, 0])
        
        # Set the translational velocity of the robot
        rtde_c.speedL([dp[0], dp[1], dp[2], 0, 0, 0], ACCELERATION, TIME)
        
        #plt.draw()
        #plt.pause(TIME/2)

        # Wait for next timestep
        rtde_c.waitPeriod(t_start)

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
