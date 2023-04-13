from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from rtde_io import RTDEIOInterface as RTDEIO
from rtde_receive import RTDEReceiveInterface as RTDEReceive


"""Module to control Robotiq's grippers - tested with HAND-E"""

import socket
import threading
import time
from enum import Enum
from typing import Union, Tuple, OrderedDict

class RobotiqGripper:
    """
    Communicates with the gripper directly, via socket with string commands, leveraging string names for variables.
    """
    # WRITE VARIABLES (CAN ALSO READ)
    ACT = 'ACT'  # act : activate (1 while activated, can be reset to clear fault status)
    GTO = 'GTO'  # gto : go to (will perform go to with the actions set in pos, for, spe)
    ATR = 'ATR'  # atr : auto-release (emergency slow move)
    ADR = 'ADR'  # adr : auto-release direction (open(1) or close(0) during auto-release)
    FOR = 'FOR'  # for : force (0-255)
    SPE = 'SPE'  # spe : speed (0-255)
    POS = 'POS'  # pos : position (0-255), 0 = open
    # READ VARIABLES
    STA = 'STA'  # status (0 = is reset, 1 = activating, 3 = active)
    PRE = 'PRE'  # position request (echo of last commanded position)
    OBJ = 'OBJ'  # object detection (0 = moving, 1 = outer grip, 2 = inner grip, 3 = no object at rest)
    FLT = 'FLT'  # fault (0=ok, see manual for errors if not zero)

    ENCODING = 'UTF-8'  # ASCII and UTF-8 both seem to work

    class GripperStatus(Enum):
        """Gripper status reported by the gripper. The integer values have to match what the gripper sends."""
        RESET = 0
        ACTIVATING = 1
        # UNUSED = 2  # This value is currently not used by the gripper firmware
        ACTIVE = 3

    class ObjectStatus(Enum):
        """Object status reported by the gripper. The integer values have to match what the gripper sends."""
        MOVING = 0
        STOPPED_OUTER_OBJECT = 1
        STOPPED_INNER_OBJECT = 2
        AT_DEST = 3

    def __init__(self):
        """Constructor."""
        self.socket = None
        self.command_lock = threading.Lock()
        self._min_position = 0
        self._max_position = 255
        self._min_speed = 0
        self._max_speed = 255
        self._min_force = 0
        self._max_force = 255

    def connect(self, hostname: str, port: int, socket_timeout: float = 2.0) -> None:
        """Connects to a gripper at the given address.
        :param hostname: Hostname or ip.
        :param port: Port.
        :param socket_timeout: Timeout for blocking socket operations.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((hostname, port))
        self.socket.settimeout(socket_timeout)

    def disconnect(self) -> None:
        """Closes the connection with the gripper."""
        self.socket.close()

    def _set_vars(self, var_dict: OrderedDict[str, Union[int, float]]):
        """Sends the appropriate command via socket to set the value of n variables, and waits for its 'ack' response.
        :param var_dict: Dictionary of variables to set (variable_name, value).
        :return: True on successful reception of ack, false if no ack was received, indicating the set may not
        have been effective.
        """
        # construct unique command
        cmd = "SET"
        for variable, value in var_dict.items():
            cmd += f" {variable} {str(value)}"
        cmd += '\n'  # new line is required for the command to finish
        # atomic commands send/rcv
        with self.command_lock:
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)
        return self._is_ack(data)

    def _set_var(self, variable: str, value: Union[int, float]):
        """Sends the appropriate command via socket to set the value of a variable, and waits for its 'ack' response.
        :param variable: Variable to set.
        :param value: Value to set for the variable.
        :return: True on successful reception of ack, false if no ack was received, indicating the set may not
        have been effective.
        """
        return self._set_vars(OrderedDict([(variable, value)]))

    def _get_var(self, variable: str):
        """Sends the appropriate command to retrieve the value of a variable from the gripper, blocking until the
        response is received or the socket times out.
        :param variable: Name of the variable to retrieve.
        :return: Value of the variable as integer.
        """
        # atomic commands send/rcv
        with self.command_lock:
            cmd = f"GET {variable}\n"
            self.socket.sendall(cmd.encode(self.ENCODING))
            data = self.socket.recv(1024)

        # expect data of the form 'VAR x', where VAR is an echo of the variable name, and X the value
        # note some special variables (like FLT) may send 2 bytes, instead of an integer. We assume integer here
        var_name, value_str = data.decode(self.ENCODING).split()
        if var_name != variable:
            raise ValueError(f"Unexpected response {data} ({data.decode(self.ENCODING)}): does not match '{variable}'")
            
        value = int(value_str)
        return value

    @staticmethod
    def _is_ack(data: str):
        return data == b'ack'

    def _reset(self):
        """
        Reset the gripper.
        The following code is executed in the corresponding script function
        def rq_reset(gripper_socket="1"):
            rq_set_var("ACT", 0, gripper_socket)
            rq_set_var("ATR", 0, gripper_socket)

            while(not rq_get_var("ACT", 1, gripper_socket) == 0 or not rq_get_var("STA", 1, gripper_socket) == 0):
                rq_set_var("ACT", 0, gripper_socket)
                rq_set_var("ATR", 0, gripper_socket)
                sync()
            end

            sleep(0.5)
        end
        """
        self._set_var(self.ACT, 0)
        self._set_var(self.ATR, 0)
        while (not self._get_var(self.ACT) == 0 or not self._get_var(self.STA) == 0):
            self._set_var(self.ACT, 0)
            self._set_var(self.ATR, 0)
        time.sleep(0.5)


    def activate(self, auto_calibrate: bool = True):
        """Resets the activation flag in the gripper, and sets it back to one, clearing previous fault flags.
        :param auto_calibrate: Whether to calibrate the minimum and maximum positions based on actual motion.
        The following code is executed in the corresponding script function
        def rq_activate(gripper_socket="1"):
            if (not rq_is_gripper_activated(gripper_socket)):
                rq_reset(gripper_socket)

                while(not rq_get_var("ACT", 1, gripper_socket) == 0 or not rq_get_var("STA", 1, gripper_socket) == 0):
                    rq_reset(gripper_socket)
                    sync()
                end

                rq_set_var("ACT",1, gripper_socket)
            end
        end
        def rq_activate_and_wait(gripper_socket="1"):
            if (not rq_is_gripper_activated(gripper_socket)):
                rq_activate(gripper_socket)
                sleep(1.0)

                while(not rq_get_var("ACT", 1, gripper_socket) == 1 or not rq_get_var("STA", 1, gripper_socket) == 3):
                    sleep(0.1)
                end

                sleep(0.5)
            end
        end
        """
        if not self.is_active():
            self._reset()
            while (not self._get_var(self.ACT) == 0 or not self._get_var(self.STA) == 0):
                time.sleep(0.01)

            self._set_var(self.ACT, 1)
            time.sleep(1.0)
            while (not self._get_var(self.ACT) == 1 or not self._get_var(self.STA) == 3):
                time.sleep(0.01)

        # auto-calibrate position range if desired
        if auto_calibrate:
            self.auto_calibrate()

    def is_active(self):
        """Returns whether the gripper is active."""
        status = self._get_var(self.STA)
        return RobotiqGripper.GripperStatus(status) == RobotiqGripper.GripperStatus.ACTIVE

    def get_min_position(self) -> int:
        """Returns the minimum position the gripper can reach (open position)."""
        return self._min_position

    def get_max_position(self) -> int:
        """Returns the maximum position the gripper can reach (closed position)."""
        return self._max_position

    def get_open_position(self) -> int:
        """Returns what is considered the open position for gripper (minimum position value)."""
        return self.get_min_position()

    def get_closed_position(self) -> int:
        """Returns what is considered the closed position for gripper (maximum position value)."""
        return self.get_max_position()

    def is_open(self):
        """Returns whether the current position is considered as being fully open."""
        return self.get_current_position() <= self.get_open_position()

    def is_closed(self):
        """Returns whether the current position is considered as being fully closed."""
        return self.get_current_position() >= self.get_closed_position()

    def get_current_position(self) -> int:
        """Returns the current position as returned by the physical hardware."""
        return self._get_var(self.POS)

    def auto_calibrate(self, log: bool = True) -> None:
        """Attempts to calibrate the open and closed positions, by slowly closing and opening the gripper.
        :param log: Whether to print the results to log.
        """
        # first try to open in case we are holding an object
        (position, status) = self.move_and_wait_for_pos(self.get_open_position(), 64, 1)
        if RobotiqGripper.ObjectStatus(status) != RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed opening to start: {str(status)}")

        # try to close as far as possible, and record the number
        (position, status) = self.move_and_wait_for_pos(self.get_closed_position(), 64, 1)
        if RobotiqGripper.ObjectStatus(status) != RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed because of an object: {str(status)}")
        assert position <= self._max_position
        self._max_position = position

        # try to open as far as possible, and record the number
        (position, status) = self.move_and_wait_for_pos(self.get_open_position(), 64, 1)
        if RobotiqGripper.ObjectStatus(status) != RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed because of an object: {str(status)}")
        assert position >= self._min_position
        self._min_position = position

        if log:
            print(f"Gripper auto-calibrated to [{self.get_min_position()}, {self.get_max_position()}]")

    def move(self, position: int, speed: int, force: int) -> Tuple[bool, int]:
        """Sends commands to start moving towards the given position, with the specified speed and force.
        :param position: Position to move to [min_position, max_position]
        :param speed: Speed to move at [min_speed, max_speed]
        :param force: Force to use [min_force, max_force]
        :return: A tuple with a bool indicating whether the action it was successfully sent, and an integer with
        the actual position that was requested, after being adjusted to the min/max calibrated range.
        """

        def clip_val(min_val, val, max_val):
            return max(min_val, min(val, max_val))

        clip_pos = clip_val(self._min_position, position, self._max_position)
        clip_spe = clip_val(self._min_speed, speed, self._max_speed)
        clip_for = clip_val(self._min_force, force, self._max_force)

        # moves to the given position with the given speed and force
        var_dict = OrderedDict([(self.POS, clip_pos), (self.SPE, clip_spe), (self.FOR, clip_for), (self.GTO, 1)])
        return self._set_vars(var_dict), clip_pos

    def move_and_wait_for_pos(self, position: int, speed: int, force: int) -> Tuple[int, ObjectStatus]:  # noqa
        """Sends commands to start moving towards the given position, with the specified speed and force, and
        then waits for the move to complete.
        :param position: Position to move to [min_position, max_position]
        :param speed: Speed to move at [min_speed, max_speed]
        :param force: Force to use [min_force, max_force]
        :return: A tuple with an integer representing the last position returned by the gripper after it notified
        that the move had completed, a status indicating how the move ended (see ObjectStatus enum for details). Note
        that it is possible that the position was not reached, if an object was detected during motion.
        """
        set_ok, cmd_pos = self.move(position, speed, force)
        if not set_ok:
            raise RuntimeError("Failed to set variables for move.")

        # wait until the gripper acknowledges that it will try to go to the requested position
        while self._get_var(self.PRE) != cmd_pos:
            time.sleep(0.001)

        # wait until not moving
        cur_obj = self._get_var(self.OBJ)
        while RobotiqGripper.ObjectStatus(cur_obj) == RobotiqGripper.ObjectStatus.MOVING:
            cur_obj = self._get_var(self.OBJ)

        # report the actual position and the object status
        final_pos = self._get_var(self.POS)
        final_obj = cur_obj
        return final_pos, RobotiqGripper.ObjectStatus(final_obj)
    
import time

IP = "192.168.1.131"

def log_info(gripper):
    print(f"Pos: {str(gripper.get_current_position()): >3}  "
          f"Open: {gripper.is_open(): <2}  "
          f"Closed: {gripper.is_closed(): <2}  ")

gripper = RobotiqGripper()
gripper.connect(IP, 63352)
log_info(gripper)
gripper.activate()

# Low value is open, high value is closed (0-255)
gripper.move_and_wait_for_pos(60, 255, 10)
time.sleep(2)
gripper.move_and_wait_for_pos(150, 255, 10)
log_info(gripper)
exit()

import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
import numpy as np
from scipy.signal import butter, lfilter
import random
import os

import atexit

from Admittance.Admittance_Control_position import AdmittanceControl


NUM_ITTERATIONS:int = 10
newton_thres:float = 0.5
torque_thres = 0.0

ACCELERATION:float = 1.0
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

def goodbye(rtde_c:RTDEControl):
    rtde_c.speedStop()
    rtde_c.stopScript()
    print("System terminated succesfully")

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

if __name__ == "__main__":
    # Create control and receive interface for the robot
    rtde_c = RTDEControl(IP)
    rtde_r = RTDEReceive(IP)

    # Create a Robotiq gripper
    gripper = RobotiqGripper(IP, 63352)

    # Read the position of the gripper
    pos = gripper.getCurrentPosition()
    print(pos)

    # Add exit handler
    atexit.register(goodbye, rtde_c)

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
