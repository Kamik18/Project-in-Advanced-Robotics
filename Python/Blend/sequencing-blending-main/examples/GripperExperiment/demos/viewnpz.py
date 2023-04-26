import numpy as np

# Load the .npz file
data = np.load('./pickplace_demo.npz')

# Access the individual arrays stored in the .npz file by their keys
# The keys are the names of the arrays used when the .npz file was saved
demo_log = dict(data)
for key in demo_log.keys():
    print(key)
print(demo_log['xpt'])
exit(1)
print(demo_log['qt'])
print(demo_log['xot'])
print(demo_log['K'])
print(demo_log['dt'])
print(demo_log['wtraj'])
print(demo_log['q0'])
print(demo_log['timesteps'])

array1 = data['array1']
array2 = data['array2']

# View the data in the arrays
print("Array1:")
print(array1)
print("Array2:")
print(array2)