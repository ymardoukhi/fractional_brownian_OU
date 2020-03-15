# Calculate the generalised time averaged MSD
def tamsd_sim(T, data_point, MAX_DIV, traj_num, trajectories):                
    import numpy as np
    from cmath import log10

    # calculate the unit time step
    delta_t = T/data_point
    # initiate the time averaged MSD trajectory
    tamsd_traj = np.zeros((MAX_DIV, traj_num))
    cdef int j
    # calculate the logirithmic timestapms to 
    # calculate for a fewer points.
    for j in range(0, MAX_DIV):
        dt = 1.0*(j)*(log10(data_point/2.0).real-log10(1).real)/MAX_DIV
        lagtime = int(10**dt)
        # calculate the time averaged MSD
        tamsd_traj[j,:] = np.sum( ((trajectories[lagtime:data_point, :] -
            trajectories[0:data_point-lagtime, :]) -
            (np.average(trajectories[lagtime:data_point, :] - 
                trajectories[0:data_point-lagtime, :], axis=1)[:, np.newaxis] )
            )**2*delta_t, axis=0)
        tamsd_traj[j,:] = tamsd_traj[j,:]/(T-delta_t*lagtime)
    return tamsd_traj

