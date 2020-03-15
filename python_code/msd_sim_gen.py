# calculate the generalised MSD
def msd_sim_gen(x_traj,traj_num):
    import numpy as np

    msd_traj = np.zeros((len(x_traj), traj_num))
    msd_traj[0:len(x_traj)-1, :] =\
            (x_traj[0:len(x_traj)-1, :] - x_traj[0, :])**2
    return msd_traj

