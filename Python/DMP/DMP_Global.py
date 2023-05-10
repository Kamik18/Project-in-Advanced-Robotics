import numpy as np
class DMP:

    def __init__():
    
    def read_out_file(self, new_pos: bool = False):
        if new_pos == True:
            down_a = np.loadtxt("Python\DMP\Out\DMP_Joint_DOWN_A_new_goal.txt", delimiter=",")
            down_b = np.loadtxt("Python\DMP\Out\DMP_Joint_DOWN_B_new_goal.txt", delimiter=",")
            up_a = np.loadtxt("Python\DMP\Out\DMP_Joint_UP_A_new_goal.txt", delimiter=",")
            up_b = np.loadtxt("Python\DMP\Out\DMP_Joint_UP_B_new_goal.txt", delimiter=",")

        else:
            down_a = np.loadtxt("Python\DMP\Out\DMP_Joint_DOWN_A_new_goal.txt", delimiter=",")
            down_b = np.loadtxt("Python\DMP\Out\DMP_Joint_DOWN_B_new_goal.txt", delimiter=",")
            up_a = np.loadtxt("Python\DMP\Out\DMP_Joint_UP_A_new_goal.txt", delimiter=",")
            up_b = np.loadtxt("Python\DMP\Out\DMP_Joint_UP_B_new_goal.txt", delimiter=",")

        return down_a, down_b, up_a, up_b
    

     