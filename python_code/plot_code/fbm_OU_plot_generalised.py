import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import numpy as np
from mpmath import gammainc, inf, gamma
from cmath import exp, pi, log10, sqrt
from os import chdir

########################## Parameteres ######################################
input_file = open('../inputs.txt','r')
inputs = input_file.readlines()
input_file.close()
MAX_DIV = int(inputs[0].split()[1])
lambd = float(inputs[1].split()[1])
H = float(inputs[2].split()[1].split('/')[0])/\
    float(inputs[2].split()[1].split('/')[1])
sigma = float(inputs[3].split()[1])
data_point = int(inputs[4].split()[1])
traj_num = int(inputs[5].split()[1])
T = float(inputs[6].split()[1])
t = np.linspace(0,T,data_point+1)
eqDist = inputs[7].split()[1]
noneqDist_x0 = float(inputs[8].split()[1])
x0 = sqrt(sigma**2/(2*(lambd)**(2*H))*complex(gamma(2*H+1))).real

if eqDist == "eqDist":
	filepath = '/home/Y-WORK/Fractional_Ornstein_Uhlenbeck/{0},{1:.2f},{2},eqDist,{3},{4},{5}'.format(int(sigma), H, int(lambd), int(data_point), int(traj_num), int(T))
if eqDist == "noneqDist":
	filepath = '/home/Y-WORK/Fractional_Ornstein_Uhlenbeck/{0},{1:.2f},{2},noneqDist,{3},{4},{5}'.format(int(sigma), H, int(lambd), int(data_point), int(traj_num), int(T))
chdir(filepath)
####################### End of Parameters ###################################
################################ MSD ########################################
def MSD_generalised(x0, H, lambd, sigma, delta, t):
	y = np.zeros((len(delta),1))
	for i in range(0,len(delta)):
		y[i,0] = float((np.var(x0)*((1-exp(-lambd*delta[i]))**2)*exp(-2*lambd*t) + (sigma**2)*(delta[i]**(2*H)) + ((sigma**2)/(2*(lambd**(2*H))))*(exp(lambd*delta[i])*(gamma(2*H+1) - gammainc(2*H+1, lambd*delta[i], inf)) + exp(-2*1j*pi*H - lambd*delta[i])*(gamma(2*H+1) - gammainc(2*H+1, -lambd*delta[i], inf))) + (sigma**2)*(exp(-lambd*(t+delta[i]))*((t+delta[i])**(2*H)) + (1/(2*(lambd**(2*H))))*(gamma(2*H+1) - gammainc(2*H+1, lambd*(t+delta[i]), inf)) + ((exp(-2*1j*pi*H-2*lambd*(t+delta[i])))/(2*(lambd**(2*H))))*(gamma(2*H+1) - gammainc(2*H+1, -lambd*(t+delta[i]), inf)) )*(1-exp(lambd*delta[i])) + (sigma**2)*(exp(-lambd*t)*(t**(2*H)) + (1/(2*(lambd**(2*H))))*(gamma(2*H+1) - gammainc(2*H+1, lambd*t, inf)) + (exp(-2*1j*pi*H - 2*lambd*t)/(2*(lambd**(2*H))))*(gamma(2*H+1) - gammainc(2*H+1, -lambd*t)) )*(1-exp(-lambd*delta[i]))).real)
	return y
############################# End of MSD ####################################
################################ TMSD #######################################
def TAMSD_generalised(x0, H, lambd, sigma, delta, T): 
	y = np.zeros((len(delta),1))
	for i in range(0,len(delta)):
		y[i,0] = float(( (sigma**2)*(delta[i]**(2*H)) + (sigma**2)/(2*(lambd**(2*H)))*(exp(lambd*delta[i])*(gamma(2*H+1) - gammainc(2*H+1, lambd*delta[i], inf)) + exp(-2*1j*pi*H-lambd*delta[i])*(gamma(2*H+1) - gammainc(2*H+1, -lambd*delta[i], inf))) +\
			 (sigma**2)*(1-exp(lambd*delta[i]))/(4*(lambd**(2*H+1))*(T-delta[i]))*((3.+2.*lambd*T)*(gamma(2*H+1) - gammainc(2*H+1, lambd*T, inf)) - (3.+2.*lambd*delta[i])*(gamma(2*H+1) - gammainc(2*H+1, lambd*delta[i])) - 2.*((gamma(2*H+2) - gammainc(2*H+2, lambd*T, inf)) - (gamma(2*H+2) - gammainc(2*H+2, lambd*delta[i]))) - exp(-2.*1j*pi*H)*(exp(-2*lambd*T)*(gamma(2*H+1) - gammainc(2*H+1, -lambd*T, inf)) - exp(-2*lambd*delta[i])*(gamma(2*H+1) - gammainc(2*H+1, -lambd*delta[i], inf))) ) +\
			 (sigma**2)*(1-exp(-lambd*delta[i]))/(4*(lambd**(2*H+1))*(T-delta[i]))*((3. + 2.*lambd*(T-delta[i]))*(gamma(2*H+1) - gammainc(2*H+1, lambd*(T-delta[i]))) - 2.*(gamma(2*H+2) - gammainc(2*H+2, lambd*(T-delta[i]), inf)) - exp(-2*1j*pi*H - 2*lambd*(T-delta[i]))*(gamma(2*H+1) - gammainc(2*H+1, -lambd*(T-delta[i]), inf))) +\
			 np.var(x0)*((1-exp(-lambd*delta[i]))**2)*((1-exp(-2*lambd*(T-delta[i])))/(2*lambd*(T-delta[i])))).real)
	return y
############################# End of TMSD ###################################
################################ MSD ########################################
def MSD_generic(x0, H, lambd, sigma, t):
	y = np.zeros((len(t),1))
	for i in range(0,len(t)):
		y[i,0] = (x0**2)*((1-exp(-lambd*t[i]))**2).real + (sigma**2)*(t[i]**(2*H))*exp(-lambd*t[i]).real + (sigma**2)/(2*(lambd**(2*H)))*(complex(gamma(2*H+1)-gammainc(2*H+1,lambd*t[i],inf)) + exp(-2*1j*pi*H)*exp(-2*lambd*t[i])*complex(gamma(2*H+1)-gammainc(2*H+1,-lambd*t[i],inf))).real
	return y
############################# End of MSD ####################################
################################ TMSD #######################################
def TAMSD_generic(x0, H, lambd, sigma, delta, T):
	y = np.zeros((len(delta),1))
	for i in range(0,len(delta)):
		y[i,0] = (sigma**2)*(delta[i]**(2*H)) + (x0**2)*((1-exp(-lambd*delta[i]))**2)*(1-exp(-2*lambd*(T-delta[i])))/(2*lambd*(T-delta[i])).real + 3*(sigma**2)/(4*(lambd**(2*H+1)))*(1-exp(-lambd*delta[i]))/(T-delta[i])*complex(gamma(2*H+1)-gammainc(2*H+1,lambd*(T-delta[i]),inf)).real - ((sigma**2)*exp(-2*1j*pi*H)*exp(-2*lambd*(T-delta[i])))/(4*(lambd**(2*H+1)))*(1-exp(-lambd*delta[i]))/(T-delta[i])*complex(gamma(2*H+1)-gammainc(2*H+1,-lambd*(T-delta[i]),inf)).real + 3*(sigma**2)/(4*(lambd**(2*H+1)))*(1-exp(lambd*delta[i]))/(T-delta[i])*(complex(gamma(2*H+1)-gammainc(2*H+1,lambd*T,inf)) - complex(gamma(2*H+1)-gammainc(2*H+1,lambd*delta[i],inf))).real - ((sigma**2)*exp(-2*1j*pi*H))/(4*(lambd**(2*H+1)))*(1-exp(lambd*delta[i]))/(T-delta[i])*(exp(-2*lambd*T)*complex(gamma(2*H+1)-gammainc(2*H+1,-lambd*T,inf)) - exp(-2*lambd*delta[i])*complex(gamma(2*H+1)-gammainc(2*H+1,-lambd*delta[i],inf))).real + (sigma**2)/(2*(lambd**(2*H+1)))*(1-exp(-lambd*delta[i]))/(T-delta[i])*(lambd*(T-delta[i])*complex(gamma(2*H+1)-gammainc(2*H+1,lambd*(T-delta[i]),inf)) - complex(gamma(2*H+2)-gammainc(2*H+2,lambd*(T-delta[i]),inf))).real + (sigma**2)/(2*(lambd**(2*H+1)))*(1-exp(lambd*delta[i]))/(T-delta[i])*(lambd*T*complex(gamma(2*H+1)-gammainc(2*H+1,lambd*T,inf)) - complex(gamma(2*H+2)-gammainc(2*H+2,lambd*T,inf)) - lambd*delta[i]*complex(gamma(2*H+1)-gammainc(2*H+1,lambd*delta[i],inf)) + complex(gamma(2*H+2)-gammainc(2*H+2,lambd*delta[i],inf))).real + (sigma**2)/(2*(lambd**(2*H)))*(exp(lambd*delta[i])*complex(gamma(2*H+1)-gammainc(2*H+1,lambd*delta[i],inf)) + exp(-2*1j*pi*H)*exp(-lambd*delta[i])*complex(gamma(2*H+1)-gammainc(2*H+1,-lambd*delta[i],inf))).real
	return y
############################# End of TMSD ###################################
t = np.linspace(0,T,data_point+1)
delta_tamsd = []
for i in range(0,MAX_DIV):
	dt = 1.0*(i)*(log10(data_point/2.0).real - log10(1).real)/MAX_DIV
	lagtime = int(10**dt)*t[1]
	if lagtime < T/2.0:
		delta_tamsd.append(lagtime)
	else:
		break

delta_msd = []
for i in range(0,MAX_DIV):
	dt = 1.0*(i)*(log10(data_point/2.0).real - log10(1).real)/MAX_DIV
	lagtime = int(10**dt)*t[1]
	if lagtime < T/2.0:
		delta_msd.append(lagtime)
	else:
		break
################### Loading Generalised MSD & TAMSD  ########################
msd_sim_generalised = np.load(filepath + '/msd_sim_generalised.npy')
ensemble_ave_msd_generalised = np.average(msd_sim_generalised, axis=1)
if eqDist == "eqDist":
	ensemble_ave_msdT_generalised = MSD_generalised(x0, H, lambd,
                                                        sigma, delta_msd, T/2.0)
	print("Equilibrium MSD calculated.")
if eqDist == "noneqDist":
	ensemble_ave_msdT_generalised = MSD_generalised(noneqDist_x0, H,
                                                        lambd, sigma,
                                                        delta_msd, T/2.0)
	print("Non-Equilibrium MSD calculated.")

tamsd_sim_generalised = np.load(filepath + '/tamsd_sim_generalised.npy')
ensemble_ave_tamsd_generalised = np.mean(tamsd_sim_generalised,1)
if eqDist == "eqDist":
	ensemble_ave_tamsdT_generalised = TAMSD_generalised(x0, H, lambd,
                                                            sigma, delta_tamsd, T)
	print("Equilibrium TAMSD calculated.")
if eqDist == "noneqDist":
	ensemble_ave_tamsdT_generalised = TAMSD_generalised(noneqDist_x0,
                                                            H, lambd, sigma,
                                                            delta_tamsd, T)
	print("Non-Equilibrium TAMSD calculated.")
#############################################################################
##################### Loading Generic MSD & TAMSD  ##########################
msd_sim_generic = np.load(filepath + '/msd_sim_generic.npy')
ensemble_ave_msd_generic = np.average(msd_sim_generic, axis=1)
if eqDist == "eqDist":
    ensemble_ave_msdT_generic = MSD_generic(x0, H, lambd, sigma, t)
    print("Equilibrium MSD calculated.")
if eqDist == "noneqDist":
    ensemble_ave_msdT_generic = MSD_generic(noneqDist_x0, H, lambd, sigma, t)
    print("Non-Equilibrium MSD calculated.")

tamsd_sim_generic = np.load(filepath + '/tamsd_sim_generic.npy')
ensemble_ave_tamsd_generic = np.mean(tamsd_sim_generic,1)
if eqDist == "eqDist":
    ensemble_ave_tamsdT_generic = TAMSD_generic(x0, H, lambd, sigma,
                                                delta_tamsd, 10*T)
    print("Equilibrium TAMSD calculated.")
if eqDist == "noneqDist":
    ensemble_ave_tamsdT_generic = TAMSD_generic(noneqDist_x0, H, lambd,
                                                sigma, delta_tamsd, 10*T)
    print("Non-Equilibrium TAMSD calculated.")
#############################################################################
########################## Plotting Section #################################
fig_size= (14,6)
params = {'backend': 'svg',
          'font.size' : 12,
          'font.weight' : 'bold',
          'axes.labelsize' : 23,
          'axes.linewidth' : 2,
          #'font.weight' : 600,
          #'text.fontsize' : 11,
          'xtick.labelsize' : 20,
          'ytick.labelsize' : 20,
          'figure.figsize': fig_size,
          'xtick.major.pad': 8,
          'ytick.major.pad' :8,
          #'line.markersize' :6,
          'text.usetex': True,
          #'text.latex.preamble' : r'\usepackage{bm}',
          'font.family': 'sans-serif',
          'font.sans-serif': 'Helvetica'}
rcParams.update(params)
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

ax1.plot(delta_msd, ensemble_ave_msd_generalised[:len(delta_msd)],
         linestyle='None', marker="x", markeredgecolor='k',
         markerfacecolor='#ffffffff', markersize=7,
         label='Simulation $\\langle R^2(t + \\Delta, t) \\rangle$')
ax1.plot(delta_tamsd, ensemble_ave_tamsd_generalised[:len(delta_tamsd)],
	 linestyle='None', marker="o", markeredgecolor='k',
	 markerfacecolor='#ffffffff', markersize=7,
	 label='Simulation $\\langle \\overline{\\delta^2(\\Delta)} \\rangle$')
ax1.plot(delta_msd, ensemble_ave_msdT_generalised, color='#c63f3fff', linewidth=2,
	 label='Analytical $\\langle R^2(t + \\Delta, t) \\rangle$')
ax1.plot(delta_tamsd, ensemble_ave_tamsdT_generalised,
	 color='#c63f3fff', linestyle='--',
         label='Analytical $\\langle \\overline{\\delta^2(\\Delta)} \\rangle$')
ax1.set_xlabel('$t,\\Delta$',fontsize=28)
ax1.set_ylabel('$\\langle R^2(t + \\Delta, t) \\rangle$,$\\langle \\overline{\\delta^2(\\Delta)} \\rangle$',
           fontsize=28)
ax1.set_xticks([0.0, 5.0, 10.0, 15.0])
ax1.legend()


ax2.plot(t[0:int(len(t)/2-1):20], ensemble_ave_msd_generic[0:int(len(t)/2-1):20],
         linestyle='None', marker="x", markeredgecolor='k',
         markerfacecolor='#ffffffff', markersize=7,
         label='Simulation $\\langle R^2(t) \\rangle$')
ax2.plot(delta_tamsd, ensemble_ave_tamsd_generic[0:len(delta_tamsd)], linestyle='None',
         marker="o", markeredgecolor='k', markerfacecolor='#ffffffff',
         markersize=7,
         label='Simulation $\\langle \\overline{\\delta^2(\\Delta)} \\rangle$')
ax2.plot(t[0:int(len(t)/2-1)], ensemble_ave_msdT_generic[0:int(len(t)/2-1)], color='#c63f3fff',
         linewidth=2, label='Analytical $\\langle R^2(t) \\rangle$')
ax2.plot(delta_tamsd, ensemble_ave_tamsdT_generic, color='#c63f3fff', linestyle='--',
         label='Analytical $\\langle \\overline{\\delta^2(\\Delta)} \\rangle$')
ax2.set_xlabel('$t,\\Delta$',fontsize=28)
ax2.set_ylabel('$\\langle R^2(t) \\rangle$,$\\langle \\overline{\\delta^2(\\Delta)} \\rangle$',fontsize=28)
ax2.set_xticks([0.0, 5.0, 10.0, 15.0])
ax2.legend()

filename = "h_{}.svg".format(H)
chdir("/home/Y-WORK")
plt.savefig(filename)
