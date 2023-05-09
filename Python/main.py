from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_io import RTDEIOInterface as RTDEIO
from rtde_receive import RTDEReceiveInterface as RTDEReceive

from Gripper.RobotiqGripper import RobotiqGripper
from Admittance.Admittance_Control_position import AdmittanceControl, AdmittanceControlQuaternion
from Admittance.Filter import Filter

import threading
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.spatial.transform import Rotation   

import threading
import sys
import atexit
import math

import uuid

ACCELERATION:float = 150 # 50

IP = "192.168.1.131"

THREAD:threading.Thread = None

TIME:float = 0.002

RUNNING: bool = True


force_measurement:list = []
speed_measurement:list = []


def goodbye(rtde_c:RTDEControl, rtde_r:RTDEReceive, gripper:RobotiqGripper):
    global RUNNING
    RUNNING = False

    # Robot
    try:
        # Stop the robot controller
        rtde_c.speedStop()
        rtde_c.stopScript()
        rtde_c.disconnect()
    except:
        print("Robot failed to terminate")

    try:
        # Disconnect the receiver
        rtde_r.disconnect()
    except:
        print("Robot failed to terminate")

    # Gripper
    try:
        if gripper is not None:
            position:int = 0 # 0-255 - Low value is open, high value is closed 
            speed:int = 50 # 0-255
            force:int = 10 # 0-255
            gripper.move_and_wait_for_pos(position=position, speed=speed, force=force)
            gripper.disconnect()
    except:
        print("Gripper failed to terminate")
    print("Program terminated")

def Grippper_example():
    position:int = 0 # 0-255 - Low value is open, high value is closed 
    speed:int = 50 # 0-255
    force:int = 10 # 0-255
    gripper.move_and_wait_for_pos(position=position, speed=speed, force=force)
    print(f"Pos: {str(gripper.get_current_position()): >3}  "
        f"Open: {gripper.is_open(): <2}  "
        f"Closed: {gripper.is_closed(): <2}  ")

# Define a function to run in the thread
def update_plot():
    global RUNNING

    # Function handler for button and slider
    def on_button_click(event):
        import signal
        print("Stop button clicked")
        os.kill(os.getpid(),  signal.SIGINT)
        exit()

    # Create the figure and axis
    fig = plt.figure(constrained_layout=False)

    # Scale the plot to the size of the screen
    fig.set_size_inches(plt.figaspect(1))

    # Add white space between subplots
    fig.subplots_adjust(hspace=0.75)

    # Create subplots for translation and rotation
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)

    # Create the button and add it to the plot
    button_ax = plt.axes([0.05, 0.9, 0.03, 0.05])
    button = Button(button_ax, "Stop")
    button.on_clicked(on_button_click)

    # Initialize data arrays
    xdata_forces = np.array([0])
    xdata_velocity = np.array([0])
    ax1_data:dict = {
        "x": np.array([0]),
        "y": np.array([0]),
        "z": np.array([0])
    }
    ax2_data:dict = {
        "x": np.array([0]),
        "y": np.array([0]),
        "z": np.array([0])
    }
    ax3_data:dict = {
        "x": np.array([0]),
        "y": np.array([0]),
        "z": np.array([0])
    }
    ax4_data:dict = {
        "x": np.array([0]),
        "y": np.array([0]),
        "z": np.array([0])
    }
    keys = ["x", "y", "z"]

    # Plot the initial empty data
    for i in range(len(keys)):
        ax1.plot(xdata_forces, ax1_data[keys[i]], label=str(keys[i] + '[N]'))
        ax2.plot(xdata_forces, ax2_data[keys[i]], label=str(keys[i] + '[Nm]'))
        ax3.plot(xdata_velocity, ax3_data[keys[i]], label=str(keys[i] + '[m/s]'))
        ax4.plot(xdata_velocity, ax4_data[keys[i]], label=str(keys[i] + '[rad/s]'))

    # Set the legend and x labels
    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        ax.set_xlabel("Time (ms)")

    # Set the y labels
    ax1.set_ylabel("Newton (N)")
    ax2.set_ylabel("Torque (Nm)")
    ax3.set_ylabel("Velocity (m/s)")
    ax4.set_ylabel("Velocity (rad/s)")

    # Set the title
    ax1.set_title("Newton")
    ax2.set_title("Torque")
    ax3.set_title("Velocity (m/s)")
    ax4.set_title("Velocity (rad/s)")

    # Open the text file for reading
    file_force = open("forces.txt", "r")
    file_velocity = open("velocity.txt", "r")
    while RUNNING:
        # Read the lineS of data from the file
        lines = file_force.readlines()

        for line in lines:
            # Split the line into x and y values
            values = [float(val) for val in line.strip().split(',')]
            x = xdata_forces[-1] + TIME
            # Add the new data to the arrays
            xdata_forces = np.append(xdata_forces, x)
            for i in range(len(keys)):
                ax1_data[keys[i]] = np.append(ax1_data[keys[i]], values[i])
                ax2_data[keys[i]] = np.append(ax2_data[keys[i]], values[i+3])

        # Read the lineS of data from the file
        lines = file_velocity.readlines()

        for line in lines:
            # Split the line into x and y values
            values = [float(val) for val in line.strip().split(',')]
            x = xdata_velocity[-1] + TIME
            # Add the new data to the arrays
            xdata_velocity = np.append(xdata_velocity, x)
            for i in range(len(keys)):
                ax3_data[keys[i]] = np.append(ax3_data[keys[i]], values[i])
                ax4_data[keys[i]] = np.append(ax4_data[keys[i]], values[i+3])

        # Update the plot with the new data
        for i in range(len(keys)):
            ax1.lines[i].set_data(xdata_forces, ax1_data[keys[i]])
            ax2.lines[i].set_data(xdata_forces, ax2_data[keys[i]])
            ax3.lines[i].set_data(xdata_velocity, ax3_data[keys[i]])
            ax4.lines[i].set_data(xdata_velocity, ax4_data[keys[i]])
       
        for ax in (ax1, ax2, ax3, ax4):
            ax.relim()
            ax.autoscale_view()

        # Pause to allow the plot to update
        plt.pause(0.01)

    plt.savefig('Output.png')

    file_force.close()
    file_velocity.close()

def angleAxis_to_RotationMatrix(angle_axis):
    # Extract the angle and axis from the angle-axis representation
    angle = np.linalg.norm(angle_axis)
    axis = angle_axis / angle

    # Calculate the rotation matrix
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    cross_matrix = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation_matrix = cos_theta * np.eye(3) + (1 - cos_theta) * np.outer(axis, axis) + sin_theta * cross_matrix

    fake = Rotation.from_rotvec(angle*axis)
   
    return rotation_matrix

def rotvec_to_R(rotvec):
    theta = np.linalg.norm(rotvec)
    if theta < sys.float_info.epsilon:
        rotational_mat = np.eye(3,dtype=float)
    else:
        r = rotvec/theta
        I = np.eye(3,dtype=float)
        r_rT = np.array(
            [[r[0]*r[0], r[0]*r[1], r[0]*r[2]],
            [r[1]*r[0], r[1]*r[1], r[1]*r[2]],
            [r[2]*r[0], r[2]*r[1], r[2]*r[2]]])
        
        r_cross = np.array([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]])
        
        rotational_mat = math.cos(theta) * I + (1 - math.cos(theta)) * r_rT + math.sin(theta) * r_cross
    
    return rotational_mat


def wrench_transformation(tcp, tau, f) -> tuple:
    
    R = angleAxis_to_RotationMatrix(tcp[3:6])
    #R = np.linalg.inv(R)

    #F  = R @ f
    #Tau = R @ tau 
    
    P = tcp[0:3] + [0, 0, 0.057]
    S = np.array([[0, -P[2], P[1]],
                  [P[2], 0, -P[0]],
                  [-P[1], P[0], 0]])
        
    F_ext = -np.dot(R.T, np.dot(S, tau)) + np.dot(R.T, f)
    Tau_ext = np.dot(R.T, tau)

    print("Tau_ext: ", Tau_ext)
    print("F inpur: ", f)

    print("F_ext: ", F_ext)
    print("Tau input: ", tau)

    exit()
    return np.array(f), np.array(Tau_ext)




if __name__ == "__main__":

    

    # Thread for force plot
    Force_thread = threading.Thread(target=update_plot)
    #Force_thread.start()
    
    
    # Create control and receive interface for the robot
    try:
        rtde_c = RTDEControl(IP)
        rtde_r = RTDEReceive(IP)
        rtde_r.stopFileRecording()
    except:
        time.sleep(1.0)
        rtde_c = RTDEControl(IP)
        rtde_r = RTDEReceive(IP)
        rtde_r.stopFileRecording()

    # Set the payload
    rtde_c.setPayload(1.39, [0,0,0.057])

    # Create a Robotiq gripper
    gripper = None
    '''
    gripper = RobotiqGripper()
    gripper.connect(IP, 63352)
    gripper.activate()

    # Close the gripper
    os.system(f"play -nq -t alsa synth {0.5} sine {220}")
    time.sleep(1.0)
    gripper.move_and_wait_for_pos(position=255, speed=5, force=25)
    time.sleep(1.0)
    '''
    
    # Add exit handler
    os.system(f"play -nq -t alsa synth {0.5} sine {440}") 
    atexit.register(goodbye, rtde_c, rtde_r, gripper)

    # Zero Ft sensor
    rtde_c.zeroFtSensor()
    time.sleep(0.2)

    # Admittance control
    admittance_control: AdmittanceControl = AdmittanceControl(
        Kp=10, Kd=25, tr=1.0, sample_time=TIME)
    admittance_control_quarternion: AdmittanceControlQuaternion = AdmittanceControlQuaternion(
        Kp=0.1, Kd=0.5, tr=0.9, sample_time=TIME)
    # Kd Lower damping -> motion is more smooth
    # Kd higher damping -> motion is more stiff

    
 
    # Create the filters for the newton and torque measurements
    newton_filters = [Filter(iterations=1, input="NEWTON") for _ in range(3)]
    torque_filters = [Filter(iterations=1, input="TORQUE") for _ in range(3)] 

    file_data:dict = {
        "tcp": [],
        "forces": [],
        "velocity": [],
        "joint": []
    }
    
    #joint_init =  [-1.5207427183734339, -1.7672444782652796, 2.0703089872943323, -1.8949862919249476, -1.6235335508929651, 0.47969508171081543]
    #tcp_init = [-0.11393864957703922, 0.4495333526836168, -0.07185608921857356, 1.5553759729297227, -0.45710594525512366, -0.14085995620748915]

    numere  = 0
    
    # Main loo
    for i_run in range(1):  
        
        rtde_c.zeroFtSensor()
        time.sleep(1.0)   
        os.system(f"play -nq -t alsa synth {0.5} sine {440}") 
        
        folder_name = "Records/TEST/"+ str(numere)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        
        filename = folder_name + '/force.txt'
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, 'w') as f:
            f.write(','.join(map(str, [0,0,0,0,0,0])) + '\n')  

        filename_vel = folder_name + '/velocity.txt'
        if os.path.exists(filename_vel):
            os.remove(filename_vel)
        with open(filename_vel, 'w') as f:
            f.write(','.join(map(str, [0,0,0,0,0,0])) + '\n')

        filename_record = folder_name + '/record_tcp.txt'
        if os.path.exists(filename_vel):
            os.remove(filename_vel)

        filename_record_j = folder_name + '/record_j.txt'
        if os.path.exists(filename_record_j):
            os.remove(filename_record_j)


        for i in range(50 * int(1/TIME)):
            
            # Detect if robot is operational
            if(rtde_r.getRobotStatus() != 3):
                print("Robot is not operational")
                exit()

            t_start = rtde_c.initPeriod()
            
            # Get the current TCP force
            force_tcp = rtde_r.getActualTCPForce()
            for axis in range(3):
                # Add the newton and torque measurement to the filter
                newton_filters[axis].add_data(force_tcp[axis])
                torque_filters[axis].add_data(force_tcp[axis + 3])

            # Get the filtered measurement
            newton_force = np.array([newton_filters[axis].filter() for axis in range(3)])
            torque_force = np.array([torque_filters[axis].filter() for axis in range(3)]) 

            tcp = rtde_r.getActualTCPPose()
            joint = rtde_r.getActualQ()
            #newton, tau = wrench_transformation(tcp, tau=torque_force, f=newton_force)
            
            # Find the translational velocity with the and amittance control
            _, p, dp, ddp = admittance_control.Translation(wrench=newton_force, p_ref=[0, 0, 0])
            _, w, dw = admittance_control_quarternion.Rotation_Quaternion(wrench=torque_force, q_ref=[1, 0, 0, 0])

            # Set the translational velocity of the robot
            rtde_c.speedL([dp[0], dp[1], dp[2], w[0], w[1], w[2]], ACCELERATION, TIME)
            

            
           
            # Save the data            
            # file_data["tcp"].append(tcp)
            # file_data["forces"].append(np.append(newton_force, torque_force))
            # file_data["velocity"].append([dp[0][0], dp[1][0], dp[2][0], w[0], w[1], w[2]])
            # file_data["joint"].append(joint)
            

            # Wait for next timestep            
            rtde_c.waitPeriod(t_start)
            

        RUNNING = False

        # Save the data to files
        # with open(filename, 'a') as f:
        #     for forces in file_data["forces"]:
        #         f.write(','.join(map(str, forces)) + '\n')

        # with open(filename_vel, 'a') as f:
        #     for velocity in file_data["velocity"]:
        #         f.write(','.join(map(str, velocity)) + '\n')
        
        # with open(filename_record, 'a') as f:
        #     for tcp in file_data["tcp"]:
        #         f.write(','.join(map(str, tcp)) + '\n')

        # with open(filename_record_j, 'a') as f:
        #     for tcp in file_data["joint"]:
        #         f.write(','.join(map(str, tcp)) + '\n')

        # # clear file_data
        file_data['tcp'].clear()
        file_data['joint'].clear()
        file_data['velocity'].clear()
        file_data['forces'].clear()
        os.system(f"play -nq -t alsa synth {0.5} sine {440}") 




