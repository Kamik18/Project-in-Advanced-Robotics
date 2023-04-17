from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_io import RTDEIOInterface as RTDEIO
from rtde_receive import RTDEReceiveInterface as RTDEReceive

from Gripper.RobotiqGripper import RobotiqGripper
from Admittance.Admittance_Control_position import AdmittanceControl, AdmittanceControlQuaternion
from Admittance.Filter import Filter

import time
import numpy as np
import os

import atexit


ACCELERATION:float = 1.0

IP = "192.168.1.131"


TIME:float = 0.002

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

    # Admittance control
    admittance_control: AdmittanceControl = AdmittanceControl(
        Kp=10, Kd=25, tr=0.3, sample_time=TIME)
    admittance_control_quarternion: AdmittanceControlQuaternion = AdmittanceControlQuaternion(
        Kp=10, Kd=25, tr=0.3, sample_time=TIME)

    # Update the plot continuously with 10 hz
    i = 0
    duration = 0.5 # Sec
    freq = 440 # Hz
    os.system(f"play -nq -t alsa synth {duration} sine {freq}")

    # Create the filters for the newton and torque measurements
    newton_filters = [Filter(iterations=3, input="NEWTON") for _ in range(3)]
    torque_filters = [Filter(iterations=3, input="TORQUE") for _ in range(3)]

    # Main loop
    while True:
        t_start = rtde_c.initPeriod()
        
        # Get the current TCP force
        force_q = rtde_r.getActualTCPForce()
        
        for axis in range(3):
            # Add the newton and torque measurement to the filter
            newton_filters[axis].add_data(force_q[axis])
            torque_filters[axis].add_data(force_q[axis+3])

        # Get the filtered measurement
        newton_force = np.array([newton_filters[axis].filter() for axis in range(3)])
        torque_force = np.array([torque_filters[axis].filter() for axis in range(3)])

        # Find the translational velocity with the and amittance control
        _, p, dp, ddp = admittance_control.Translation(
           wrench=newton_force, p_ref=np.array([0, 0, 0]))
        
        # TODO 
        #_, w, dw = admittance_control_quarternion.Rotation_Quaternion(wrench=torque_force, q_ref=[0, 0, 0])
        
           
        # Set the translational velocity of the robot
        rtde_c.speedL([dp[0], dp[1], dp[2], 0, 0, 0], ACCELERATION, TIME)
        
        # Wait for next timestep
        rtde_c.waitPeriod(t_start)

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
