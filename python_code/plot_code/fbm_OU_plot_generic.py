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
def MSD(x0, H, lambd, sigma, t):
	y = np.zeros((len(t),1))
	for i in range(0,len(t)):
		y[i,0] = (x0**2)*((1-exp(-lambd*t[i]))**2).real + (sigma**2)*(t[i]**(2*H))*exp(-lambd*t[i]).real + (sigma**2)/(2*(lambd**(2*H)))*(complex(gamma(2*H+1)-gammainc(2*H+1,lambd*t[i],inf)) + exp(-2*1j*pi*H)*exp(-2*lambd*t[i])*complex(gamma(2*H+1)-gammainc(2*H+1,-lambd*t[i],inf))).real
	return y
############################# End of MSD ####################################
################################ TMSD #######################################
def TAMSD(x0, H, lambd, sigma, delta, T):                
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

msd_sim = np.load(filepath + '/msd_sim_generic.npy')
ensemble_ave_msd = np.average(msd_sim, axis=1)
if eqDist == "eqDist":
    ensemble_ave_msdT = MSD(x0, H, lambd, sigma, t)
    print("Equilibrium MSD calculated.")
if eqDist == "noneqDist":
    ensemble_ave_msdT = MSD(noneqDist_x0, H, lambd, sigma, t)
    print("None-Equilibrium MSD calculated.")

tamsd_sim = np.load(filepath + '/tamsd_sim_generic.npy')
ensemble_ave_tamsd = np.mean(tamsd_sim,1)
if eqDist == "eqDist":
    ensemble_ave_tamsdT = TAMSD(x0, H, lambd, sigma, delta_tamsd, T)
    print("Equilibrium TAMSD calculated.")
if eqDist == "noneqDist":
    ensemble_ave_tamsdT = TAMSD(noneqDist_x0, H, lambd, sigma, delta_tamsd, T)
    print("None-Equilibrium TAMSD calculated.")

#################################### Plotting Section##########################
fig_size= (12,8)
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

plt.plot(t[0:int(len(t)/2-1):20], ensemble_ave_msd[0:int(len(t)/2-1):20],
         linestyle='None', marker="x", markeredgecolor='k',
         markerfacecolor='#ffffffff', markersize=7,
         label='Simulation $\\langle R^2(t) \\rangle$')
plt.plot(delta_tamsd, ensemble_ave_tamsd[0:len(delta_tamsd)], linestyle='None',
         marker="o", markeredgecolor='k', markerfacecolor='#ffffffff',
         markersize=7,
         label='Simulation $\\langle \\overline{\\delta^2(\\Delta)} \\rangle$')
plt.plot(t[0:int(len(t)/2-1)], ensemble_ave_msdT[0:int(len(t)/2-1)], color='k',
         linewidth=2, label='Analytical $\\langle R^2(t) \\rangle$')
plt.plot(delta_tamsd, ensemble_ave_tamsdT, color='k', linestyle='--',
         label='Analytical $\\langle \\overline{\\delta^2(\\Delta)} \\rangle$')
plt.xlabel('$t,\\Delta$',fontsize=28)
plt.ylabel('$\\langle R^2(t) \\rangle$,$\\langle \\overline{\\delta^2(\\Delta)} \\rangle$',fontsize=28)
plt.xticks([0.0, 5.0, 10.0, 15.0])

#plt.legend()
#plt.show()
