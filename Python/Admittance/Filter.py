import numpy as np

class Filter(object):
    data: np.ndarray
    input: str

    def __init__(self, iterations: int, input: str) -> None:
        """Initialize the filter

        Args:
            iterations (int): Number of iterations to store
            input (str): Type of filter, 'NEWTON' or 'TORQUE'
        """
        # Create the data array to store the measurements
        self.data = np.zeros((iterations, 1))
        # Set the input type
        self.input = input

    def add_data(self, data: float) -> None:
        """Add data point to the filter

        Args:
            data (float): Data point
        """
        # Add the data to the array
        if self.input == "NEWTON":
            if abs(data) < 0.0:
                data = 0
            else:
                # Offset the force
                data = data * 0.35

        elif self.input == "TORQUE":
            if abs(data) < 0.0:
                data = 0
            else:
                # Offset the torque
                data = data * 10

        # Add the data to the array and remove the oldest data
        self.data = np.append(self.data[1:], data)

    def filter(self) -> float:
        """Get the filtered measurement

        Returns:
            float: A filtered measurement
        """
        data = self.data

        # Return the mean of the filtered data
        mean = np.mean(data)
        if mean == 0:
            return 0

        # Check if the force and torque is within the threshold
        res = np.log2(abs(mean)) * mean / 5

        # Remove the offset
        if self.input == "NEWTON":
            res /= 0.25
        elif self.input == "TORQUE":
            res /= 10

        # Return the result
        return res

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    import TCP_data

    # Filters
    newton_filters = [Filter(iterations=1, input="NEWTON") for _ in range(3)]
    torque_filters = [Filter(iterations=1, input="TORQUE") for _ in range(3)]

    # Recorded data
    tcp_forces = TCP_data.tcp_forces

    # Function handler for button and slider
    def on_button_click(event):
        print("Stop button clicked")
        exit()

    def slider_newton(val):
        global newton_thres
        newton_thres = val

    def slider_torque(val):
        global torque_thres
        torque_thres = val

    # Create the figure and axis
    fig = plt.figure(constrained_layout=False)

    # Scale the plot to the size of the screen
    fig.set_size_inches(plt.figaspect(0.5))

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

    # Define the x and y data
    raw_tcp_forces = np.array(tcp_forces)
    raw_tcp_forces = raw_tcp_forces.transpose()

    x_data = [0]
    for i in range(len(raw_tcp_forces[0]) - 1):
        x_data.append(x_data[-1] + 2)
    y_data = np.linspace(0, 0, len(x_data))

    # Set the x and y limits
    ax1.set_xlim(x_data[0], x_data[-1])
    ax2.set_xlim(x_data[0], x_data[-1])
    ax3.set_xlim(x_data[0], x_data[-1])
    ax4.set_xlim(x_data[0], x_data[-1])

    # Set the x and y labels
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Newton (N)")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Newton (N)")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Torque (Nm)")
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("Torque (Nm)")

    # Set the title
    ax1.set_title("Newton")
    ax2.set_title("Newton raw")
    ax3.set_title("Torque")
    ax4.set_title("Torque raw")

    # Create the line objects
    legend = ["x", "y", "z"]
    lines_newton = [ax1.plot(x_data, y_data, label=legend[i])[0]
                    for i in range(3)]
    lines_newton_raw = [ax2.plot(x_data, raw_tcp_forces[i], label=legend[i])[
        0] for i in range(3)]
    lines_torque = [ax3.plot(x_data, y_data, label=legend[i])[0]
                    for i in range(3)]
    lines_torque_raw = [
        ax4.plot(x_data, raw_tcp_forces[i + 3], label=legend[i])[0] for i in range(3)]

    # Set the legend
    ax1.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax2.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax3.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax4.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

    # Store the newton measurements
    newton_data = [np.zeros((len(x_data), 1)) for _ in range(3)]
    torque_data = [np.zeros((len(x_data), 1)) for _ in range(3)]

    # Main loop
    for index, tcp_force in enumerate(tcp_forces):
        # Remove the average force and torque from the data with a threshold
        for axis in range(3):
            # Add data point to the filter
            newton_filters[axis].add_data(tcp_force[axis])
            torque_filters[axis].add_data(tcp_force[axis + 3])

        # Update the y-data of each newton line object
        for axis, line in enumerate(lines_newton):
            # Get the filtered measurement
            newton_data[axis][index] = newton_filters[axis].filter()

            # Update the line data
            line.set_ydata(newton_data[axis])

        # Update the y-data of each torque line object
        for axis, line in enumerate(lines_torque):
            # Add the new data to the end of the y-data and remove the first element
            torque_data[axis][index] = torque_filters[axis].filter()

            # Update the line data
            line.set_ydata(torque_data[axis])

    # Set the y-axis limits
    ax1.set_ylim(-25, 25)
    ax2.set_ylim(-25, 25)
    ax3.set_ylim(-1, 1)
    ax4.set_ylim(-1, 1)

    plt.show()
