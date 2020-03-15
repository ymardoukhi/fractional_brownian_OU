# Generate fractionaly Ornstoin-Ulenbeck 
# trajectories
def fou_trajectory(data_point, H, T, lambd, sigma, initial_x):
    from fbm import FBM 
    import numpy as np

    # generate the fractional Brownian Motion trajectories
    f=FBM(n=data_point, hurst=H, length=T, method='hosking')
    # store the fractional Gaussian noise
    dB=f.fgn()
    # storet he time vector
    t=f.times()
    del f
    # initiate the trajectory
    x=np.zeros((len(t),1))
    x[0,0]=initial_x 
    print('x0 = ' + str(x[0,0]))
    cdef int i
    # calculate the fractional Ornstein-Uhlenbeck 
    # integration
    for i in range(1,len(t)):
        x[i,0] = np.exp(-lambd*t[i])*(x[0,0] +\
                sigma*np.sum(np.exp(lambd*t[0:i])*dB[0:i]))
    return x

