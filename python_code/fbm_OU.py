from fbm import FBM
import multiprocessing as mp
import numpy as np
from functools import partial
from mpmath import gammainc, inf, gamma
from cmath import exp, pi, log10, sqrt
from os import chdir, makedirs
import errno
from random import normalvariate

########################## Parameteres ######################################
#global trajectories, var_x0, MAX_DIV, lambd, H, sigma, data_point, traj_num, T, t, x0

input_file = open('inputs.txt','r')
inputs = input_file.readlines()
input_file.close()
MAX_DIV = int(inputs[0].split()[1])
lambd = float(inputs[1].split()[1])
H = float(inputs[2].split()[1].split('/')[0])/float(inputs[2].split()[1].split('/')[1])
sigma = float(inputs[3].split()[1])
data_point = int(inputs[4].split()[1])
traj_num = int(inputs[5].split()[1])
T = float(inputs[6].split()[1])
eqDist = inputs[7].split()[1]
noneqDist_x0 = float(inputs[8].split()[1])
x0 = sqrt(sigma**2/(2*(lambd)**(2*H))*complex(gamma(2*H+1))).real

if eqDist == "eqDist":
	filepath = '/home/Y-WORK/Fractional_Ornstein_Uhlenbeck/{0},{1:.2f},{2},eqDist,{3},{4},{5}'.format(int(sigma), H, int(lambd), int(data_point), int(traj_num), int(T))
if eqDist == "noneqDist":
	filepath = '/home/Y-WORK/Fractional_Ornstein_Uhlenbeck/{0},{1:.2f},{2},noneqDist,{3},{4},{5}'.format(int(sigma), H, int(lambd), int(data_point), int(traj_num), int(T))
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
#################### End of Global Variabels ################################
################### Fractional Ornstein-Uhlenbeck Trajectory#################
def fbm_trajectory(initial_x):
	f = FBM(n=data_point, hurst=H, length=T, method='hosking')
	dB = f.fgn()
	t = f.times()
	x = np.zeros((data_point+1,1))
	x[0,0] = initial_x 
	print('x0 = ' + str(x[0,0]))
	for i in range(1,len(t)):
		x[i,0] = np.exp(-lambd*t[i])*(x[0,0] +
				sigma*np.sum(np.exp(lambd*t[0:i])*dB[0:i]))
	return x
############## End of Fractional Ornstein-Uhlenbeck Trajectory###############
############################### MSD_Sim ######################################
def MSD_Sim():
	msd_traj = np.zeros((MAX_DIV, traj_num))
	lagtime = []
	for j in range(0, MAX_DIV):
		dt = 1.0*(j)*(log10(data_point/2.0).real-log10(1).real)/MAX_DIV
		lagtime.append(int(10**dt))
	index = [ int(data_point/2 + i) for i in lagtime]
	msd_traj[:, :] = ((trajectories[index, :] - 
			   trajectories[int(data_point/2), :]) -
			   np.average(trajectories[index, :] -
			   trajectories[int(data_point/2), :], axis=1)[:, np.newaxis])**2
	return msd_traj
############################ End of MSD_Sim ##################################
############################### TMSD_Sim #####################################
def TAMSD_Sim():                
	delta_t = T/data_point
	tamsd_traj = np.zeros((MAX_DIV, traj_num))
	for j in range(0,MAX_DIV):
		print("Generalised TAMSD: #%d" %(j))
		dt = 1.0*(j)*(log10(data_point/2.0).real-log10(1).real)/MAX_DIV
		lagtime = int(10**dt)
		tamsd_traj[j,:] = np.sum( ((trajectories[lagtime:data_point, :] -
					    trajectories[0:data_point-lagtime, :]) -
					   (np.average(trajectories[lagtime:data_point, :] - 
					               trajectories[0:data_point-lagtime, :], axis=1)[:, np.newaxis] )
					  )**2*delta_t, axis=0)
		tamsd_traj[j,:] = tamsd_traj[j,:]/(T-delta_t*lagtime)
	return tamsd_traj
############################ End of TMSD_Sim #################################
############################### MSD_Sim ######################################
def MSD_Sim_Gen(x_traj,traj_num):
	msd_traj = np.zeros((len(x_traj),traj_num))
	msd_traj[0:len(x_traj)-1, :] = (x_traj[0:len(x_traj)-1, :] - x_traj[0, :])**2
	return msd_traj
############################ End of MSD_Sim ##################################
############################### TMSD_Sim #####################################
def TAMSD_Sim_Gen():                
	delta_t = T/data_point
	tamsd_traj = np.zeros((MAX_DIV,traj_num))
	for j in range(0,MAX_DIV):
		print('Generic TAMSD: #%d' %(j))
		dt = 1.0*(j)*(log10(data_point/2.0).real-log10(1).real)/MAX_DIV
		lagtime = int(10**dt)
		tamsd_traj[j,:] = np.sum( ((trajectories[lagtime:data_point, :] -
					  trajectories[0:data_point-lagtime, :])**2)*delta_t, axis=0 )
		tamsd_traj[j,:] = tamsd_traj[j,:]/(T-delta_t*lagtime)
	return tamsd_traj
############################ End of TMSD_Sim #################################
##################### Generating Trajectories ################################
print("Generating Fractional Ornstein Uhlenbeck Processes started...")
pool1 = mp.Pool(processes=CPU_n)
results = pool1.map(fbm_trajectory,initial_pos)
pool1.close()
pool1.terminate()
del pool1
print("Trajectories Created")
trajectories = np.reshape(results,((traj_num,data_point+1)))
trajectories = np.transpose(trajectories)
del results
print("Trajectories Reshaped")
chdir(filepath)
print("Saving the trajectories...")
np.save('trajectories.npy',trajectories)
print("Saving the trajectories finished")
#############################################################################
####################### Generic Definition ##################################
print("Generic MSD began...")
msd_sim = MSD_Sim_Gen(trajectories,traj_num)
chdir(filepath)
np.save('msd_sim_generic.npy',msd_sim)
del msd_sim
print("Generic MSD done!")
print("Generic TAMSD began...")
tamsd_sim = TAMSD_Sim_Gen()
chdir(filepath)
np.save('tamsd_sim_generic.npy',tamsd_sim)
del tamsd_sim
print("Generic TAMSD done!")
#############################################################################
###################### Generalised Definition ###############################
print("Generalised MSD began...")
msd_sim = MSD_Sim()
chdir(filepath)
np.save('msd_sim_generalised.npy',msd_sim)
del msd_sim
print("Generalised MSD done!")
print("Generalised TAMSD began...")
tamsd_sim = TAMSD_Sim()
chdir(filepath)
np.save('tamsd_sim_generalised.npy',tamsd_sim)
del tamsd_sim
print("Generalised TAMSD done!")
del trajectories
#############################################################################
