# calculate the generalised definition of MSD
def msd_sim(MAX_DIV, traj_num, trajectories, data_point):
    import numpy as np
    from cmath import log10

    # initialise the msd matrix
    msd_traj = np.zeros((MAX_DIV, traj_num))
    cdef list lagtime = []
    cdef int j
    # calculate logarithmically the timestapms
    # to speed up the calculation (fewer number 
    # points)
    for j in range(0, MAX_DIV):
        dt = 1.0*(j)*(log10(data_point/2.0).real-log10(1).real)/MAX_DIV
        lagtime.append(int(10**dt))
    index = [ int(data_point/2 + i) for i in lagtime]
    # calculate MSD
    msd_traj[:, :] = ((trajectories[index, :] - 
        trajectories[int(data_point/2), :]) -
        np.average(trajectories[index, :] -
            trajectories[int(data_point/2), :], axis=1)[:, np.newaxis])**2
    return msd_traj

