

"""
####################### Test Inv kin ##############################
#up_b: np.ndarray = np.loadtxt("./Records/Up_B/1/record_tcp.txt", delimiter=',', skiprows=0)


inv_kin_traj = np.ndarray([len(up_b),6])
for i in range(len(up_b)):
    inv_kin_traj[i] = traj.inverse_kinematics(create_4x4_matrix(up_b[i]), up_b_j[i])
    #inv_kin_traj[i] = UR5.ik_lm_wampler(create_4x4_matrix(up_b[i]), ilimit=500, slimit=500, q0=up_b_j[i])[0]
    #inv_kin_traj_corrected[i] = [float(x)% (2*np.pi) for x in inv_kin_traj[i]] 


fig1 = plt.figure()
ax = fig1.add_subplot(111)
line_objects = ax.plot(up_b_j[:,0],'r',up_b_j[:,1],'g',up_b_j[:,2],'b', up_b_j[:,3],'c', up_b_j[:,4],'m', up_b_j[:,5],'y')
ax.legend(line_objects, ('q0', 'q1', 'q2', 'q3', 'q4', 'q5'))
ax.set_title('Joint space')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
line_objects2 = ax2.plot(up_b[:,0],'r',up_b[:,1],'g',up_b[:,2],'b', up_b[:,3],'c', up_b[:,4],'m', up_b[:,5],'y')
ax2.legend(line_objects2, ('x', 'y', 'z', 'rx', 'ry', 'rz'))
ax2.set_title('TCP')


fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
line_objects3 = ax3.plot(inv_kin_traj[:,0],'r',inv_kin_traj[:,1],'g',inv_kin_traj[:,2],'b', inv_kin_traj[:,3],'c', inv_kin_traj[:,4],'m', inv_kin_traj[:,5],'y')
ax3.legend(line_objects3, ('q0', 'q1', 'q2', 'q3', 'q4', 'q5'))
ax3.set_title('TCP inv Kin')

"""


"""
####################### Test cubic spline ##############################
def test():
    # Define two Cartesian space trajectories as lists of positions and orientations
    traj1_pos = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
    traj1_ori = [[1, 0, 0, 0], [0.707, 0.0, 0.707, 0.0], [0.0, 1.0, 0.0, 0.0]]
    traj2_pos = [[3, 4, 5], [4, 5, 6], [5, 6, 7]]
    traj2_ori = [[0.0, 1.0, 0.0, 0.0], [0.707, 0.0, 0.707, 0.0], [1, 0, 0, 0]]

    # Define the duration of the interpolation
    duration = 2.0

    # Calculate the step size for the interpolation
    num_steps = len(traj1_pos)
    step_size = duration / (num_steps - 1)

    # Initialize the blended trajectory as empty lists
    blended_traj_pos = []
    blended_traj_ori = []

    # Perform linear interpolation between the two trajectories
    for i in range(num_steps):
        # Calculate the blend factor (between 0 and 1)
        t = i * step_size / duration
        
        # Linearly interpolate the position and orientation
        blended_pos = (1 - t) * np.array(traj1_pos[i]) + t * np.array(traj2_pos[i])
        blended_ori = txq.slerp(traj1_ori[i], traj2_ori[i], t)

        # Append the blended position and orientation to the blended trajectory
        blended_traj_pos.append(blended_pos)
        blended_traj_ori.append(blended_ori)

def test2():
    t = np.array([0, 1, 2, 3, 4, 5])
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 2], [1, 2]])
    x = points[:, 0]
    y = points[:, 1]
    x = x.reshape(-1,1)
    y = x.reshape(-1,1)
    print('1.x: ',x.shape)
    print('2.y: ',y.shape)
    
    theta = 2 * np.pi * np.linspace(0, 1, 5)
    y = np.c_[np.cos(theta), np.sin(theta)]
    print('len(y)', len(theta))
    print('y', y.shape)
    
    plt.figure()
    plt.plot(t, x, t, x, 'o')
    plt.title('x vs. t')
    plt.axis([-0.5, 5.5, -0.5, 1.5])
    
    tq = np.linspace(0, 5, 8)#np.arange(0, 5.1, 0.1)
    tq2 = np.linspace(0, 5, 51)
    slope0 = 0
    slopeF = 0
    print('3.x', x)
    print('4.slope0', slope0)
    print('5.slopeF', slopeF)
    #tes = np.concatenate(([slope0], x, [slopeF]))
    tes = np.array([0, 0, 1, 1, 0, 0, 1,0]).transpose()
    tes = tes.reshape(-1,1)
    print('6.tes', tes.shape)
    cs_x = CubicSpline(tq, tes)#np.concatenate(([slope0], x, [slopeF])))
    #cs_y = CubicSpline(t, np.concatenate(([slope0], y, [slopeF])))
    xq = cs_x(tq2)
    #yq = cs_y(tq)
    
    plt.figure()
    plt.plot(t, x, t, x, 'o', tq, xq, ':.')
    plt.axis([-0.5, 5.5, -0.5, 1.5])
    plt.show()
#def f(x):  
#        return 1/(1 + (x**2))  
def test3():
    a = -1      
    b = 1
    n = 6   
    xArray = np.linspace(a,b,n)
    yArray = [0, 1, 1, 0, 0,1]#f(xArray)
    plt.figure()
    plt.plot(xArray, yArray, xArray, yArray, 'o', label="Lines, " + str(n) + " points")
    
    
    x = np.linspace(a,b,101)
    cs = CubicSpline(xArray, yArray, True)  # fit a cubic spline
    y = cs(x) # interpolate/extrapolate
    plt.figure()
    plt.plot(xArray, yArray,x, y, label="Interpolation, " + str(n) + " points")
    plt.show()
"""

"""
######################Convert to trajectory###########################
#up_b = trajectory.Trajectory('jtraj', 2, up_b)
#down_b = trajectory.Trajectory('jtraj', 2, down_b)
#down_a = trajectory.Trajectory('jtraj', 2, down_a)
#up_a = trajectory.Trajectory('jtraj', 2, up_a)
"""