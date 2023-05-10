import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.spatial.transform import Rotation   

import numpy as np

PATH :str = "Records/experiments"
TIME:int = 0.02

# Function handler for button and slider
def on_button_click(event):
    import signal
    print("Stop button clicked")
    exit()

# Create the figure and axis
fig = plt.figure(constrained_layout=False)

# Scale the plot to the size of the screen
fig.set_size_inches(plt.figaspect(1))

# Add white space between subplots
fig.subplots_adjust(hspace=0.75)

# Create subplots for translation and rotation
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

# Create the button and add it to the plot
button_ax = plt.axes([0.05, 0.9, 0.03, 0.05])
button = Button(button_ax, "Stop")
button.on_clicked(on_button_click)

keys = ["j1", "j2", "j3", "j4", "j5", "j6"]

# Plot the initial empty data
for i in range(len(keys)):
    ax1.plot(np.array([0]), np.array([0]), label=str(keys[i] + '[rad/s²]'))
    ax2.plot(np.array([0]), np.array([0]), label=str(keys[i] + '[rad/s]'))
    
# Set the legend and x labels
for ax in [ax1, ax2]:
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.set_xlabel("Time (ms)")

# Set the y labels
ax1.set_ylabel("Acceleration (rad/s²)")
ax2.set_ylabel("Velocity (rad/s)")

# Set the title
ax1.set_title("Acceleration")
ax2.set_title("Velocity")

# Open the text file for reading
with open(f"{PATH}/acc.txt", "r") as file:
    # Read the lines of data from the file
    lines = file.readlines()
    x_data = np.array([0])
    ax_data:dict = {
        "j1": np.array([0]),
        "j2": np.array([0]),
        "j3": np.array([0]),
        "j4": np.array([0]),
        "j5": np.array([0]),
        "j6": np.array([0])
    }

    for line in lines:
        # Remove '[' and ']' from the line
        line = line.replace("[", "").replace("]", "")
        # Split the line into x and y values
        values = [float(val) for val in line.split(',')]
        x = x_data[-1] + TIME
        # Add the new data to the arrays
        x_data = np.append(x_data, x)
        for i in range(len(keys)):
            ax_data[keys[i]] = np.append(ax_data[keys[i]], values[i])
    for i in range(len(keys)):
        # Differentiate the data
        #ax_data[keys[i]] = np.diff(ax_data[keys[i]]) / TIME
        # Remove the last element from the x data
        #x_data = np.linspace(0, len(ax_data[keys[i]]) * TIME, len(ax_data[keys[i]]))
        ax1.lines[i].set_data(x_data, ax_data[keys[i]])

# Open the text file for reading
with open(f"{PATH}/vel.txt", "r") as file:
    # Read the lines of data from the file
    lines = file.readlines()
    x_data = np.array([0])
    ax_data:dict = {
        "j1": np.array([0]),
        "j2": np.array([0]),
        "j3": np.array([0]),
        "j4": np.array([0]),
        "j5": np.array([0]),
        "j6": np.array([0])
    }

    for line in lines:
        # Remove '[' and ']' from the line
        line = line.replace("[", "").replace("]", "")
        # Split the line into x and y values
        values = [float(val) for val in line.split(',')]
        x = x_data[-1] + TIME
        # Add the new data to the arrays
        x_data = np.append(x_data, x)
        for i in range(len(keys)):
            ax_data[keys[i]] = np.append(ax_data[keys[i]], values[i])
    for i in range(len(keys)):
        ax2.lines[i].set_data(x_data, ax_data[keys[i]])


for ax in (ax1, ax2):
    ax.relim()
    ax.autoscale_view()

plt.show()