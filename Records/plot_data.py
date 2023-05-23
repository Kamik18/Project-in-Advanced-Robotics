import numpy as np
import time
import matplotlib.pyplot as plt

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

folder1 = "moveJ"
folder2 = "No_blend"
data= read_exp_dat(f'Records/experiments/{folder1}/tcp_speed.txt')
data_blend= read_exp_dat(f'Records/experiments/{folder2}/tcp_speed.txt')

folder1 = "Filtered"
folder2 = "Raw"

data= np.loadtxt(f'Records/forces_filter.txt', delimiter=',')
data_blend= np.loadtxt(f'Records/forces_raw.txt', delimiter=',')

#data_gmm = read_exp_dat('Records\experiments/DMP_B_GMM_A/acc.txt')
num_points = data_blend.shape[0]
time = np.arange(num_points) * 0.02


fig, axs = plt.subplots(6, 1,figsize=(6.4, 8.6))
# plot data
for i in range(6):
    axs[i].plot(time, data_blend[:, i], label=folder2, color='red' , linewidth=1)
    if data.shape[0] > 0:
         axs[i].plot(time[:data.shape[0]], data[:, i], label=folder1, color='blue', linewidth=1)

    axs[i].set_ylabel(f'$q_{i+1} [rad/s^2]$')
    
    # Hide the x-axis tick labels for the first five subplots
    if i < 5:
        axs[i].set_xticklabels([])

# Set the x-axis label and legend for the last subplot
axs[5].set_xlabel('Time $[s]$')
axs[1].legend(loc='upper right')

plt.show()