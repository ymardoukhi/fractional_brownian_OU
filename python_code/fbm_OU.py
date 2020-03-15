import errno
import multiprocessing as mp
import numpy as np
from functools import partial
from mpmath import gamma
from cmath import  sqrt
from os import chdir, makedirs
from random import normalvariate
from fou_trajectory import fou_trajectory
from tamsd_sim import tamsd_sim
from tamsd_sim_gen import tamsd_sim_gen
from msd_sim import msd_sim
from msd_sim_gen import msd_sim_gen

########################## Parameteres ######################################
# Global Variables: 
# trajectories, var_x0, MAX_DIV, 
# lambd, H, sigma, data_point, 
# traj_num, T, t, x0

# Reading the global variables from 
# the input file
input_file = open('inputs.txt','r')
inputs = input_file.readlines()
input_file.close()
# Divide the trajectory into MAX_DIV
# for logarithmic performance usage
MAX_DIV = int(inputs[0].split()[1])
# lambda is the positive definite 
# parameter of the fractional 
# Ornstein-Uhlenbeck process
lambd = float(inputs[1].split()[1])
# Hurst exponent
H = float(inputs[2].split()[1].split('/')[0])/\
        float(inputs[2].split()[1].split('/')[1])
# sigma is the positive definite 
# parameter of the fractional 
# Ornstein-Uhlenbeck process
sigma = float(inputs[3].split()[1])
# Total number of data points to be 
# generated for a single trajectory 
# of fractional Ornstein-Uhlenbeck 
# process
data_point = int(inputs[4].split()[1])
# total number of trajectories to be 
# generated
traj_num = int(inputs[5].split()[1])
# Total time observation (trajectory 
# length)
T = float(inputs[6].split()[1])
# either "eqDist" on "noneqDist"
# indicacting equilibrium initial 
# condition or otherwise
eqDist = inputs[7].split()[1]
# initial condition for "noneqDist"
noneqDist_x0 = float(inputs[8].split()[1])
# equilibrium stationary solution
x0 = sqrt(sigma**2/(2*(lambd)**(2*H))*\
        complex(gamma(2*H+1))).real

# navigate to the directory which 
# depending on the global variables
if eqDist == "eqDist":
	filepath = '{0},{1:.2f},{2},eqDist,{3},{4},{5}'.format(int(sigma), H, int(lambd), int(data_point), int(traj_num), int(T))
if eqDist == "noneqDist":
	filepath = '{0},{1:.2f},{2},noneqDist,{3},{4},{5}'.format(int(sigma), H, int(lambd), int(data_point), int(traj_num), int(T))
try:
	makedirs(filepath)
except OSError as e:
	if e.errno != errno.EEXIST:
		raise
chdir(filepath)
####################### End of Parameters ###################################
######################### Global Variables ##################################
CPU_n = mp.cpu_count() - 2
initial_pos = []
f = open('initial_pos','w')
if eqDist == "eqDist":
    for i in range(0,traj_num):
        rand_x0 = normalvariate(0,x0)
        initial_pos.append(rand_x0)
        f.write(str(rand_x0) + '\n')
if eqDist == "noneqDist":
    for i in range(0,traj_num):
        initial_pos.append(noneqDist_x0)
        f.write(str(noneqDist_x0) + '\n')
f.close()
var_x0 = np.var(initial_pos)


if __name__ == "__main__":
    ##################### Generating Trajectories ########################
    print("Generating Fractional Ornstein Uhlenbeck Processes started...")
    pool1 = mp.Pool(processes=CPU_n)
    par_fou_trajectory = partial(fou_trajectory, data_point, H, T, lambd, sigma)
    results = pool1.map(par_fou_trajectory, initial_pos)
    pool1.close()
    pool1.terminate()
    del pool1
    print("Trajectories Created")
    trajectories = np.reshape(results, ((traj_num,data_point+1)))
    trajectories = np.transpose(trajectories)
    del results
    print("Trajectories Reshaped")
    print("Saving the trajectories...")
    np.save('trajectories.npy', trajectories)
    print("Saving the trajectories finished")
    ######################################################################
    ####################### Generic Definition ###########################
    print("Generic MSD began...")
    msd_sim_res = msd_sim_gen(trajectories, traj_num)
    np.save('msd_sim_generic.npy', msd_sim_res)
    del msd_sim_res
    print("Generic MSD done!")
    print("Generic TAMSD began...")
    tamsd_sim_res = tamsd_sim_gen(MAX_DIV, traj_num, trajectories, 
            data_point, T)
    np.save('tamsd_sim_generic.npy', tamsd_sim_res)
    del tamsd_sim_res
    print("Generic TAMSD done!")
    ######################################################################
    ###################### Generalised Definition ########################
    print("Generalised MSD began...")
    msd_sim_res = msd_sim(MAX_DIV, traj_num, trajectories, data_point)
    np.save('msd_sim_generalised.npy', msd_sim_res)
    del msd_sim_res
    print("Generalised MSD done!")
    print("Generalised TAMSD began...")
    tamsd_sim_res = tamsd_sim(T, data_point, MAX_DIV, traj_num, trajectories)
    np.save('tamsd_sim_generalised.npy', tamsd_sim_res)
    del tamsd_sim_res
    print("Generalised TAMSD done!")
    del trajectories
    ######################################################################
