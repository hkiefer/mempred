import numpy as np
import matplotlib.pyplot as plt
from memtools import * #install first https://github.com/jandaldrop/memtools
import memtools as mp
from mempred import *
from scipy import optimize
from scipy import integrate
from scipy import stats
from scipy import interpolate
import pandas as pd

def harm(x,k):
    return k/2*x**2

def extract_kernel(xvaf,trunc,kT=2.494,bins='kde',physical=False,free_energy=True,mori=True,verbose=False):
    
    dt = xvaf["t"].iloc[1] - xvaf["t"].iloc[0]
    if verbose:
        print('dt = ' + str(dt))
    
    corrv = correlation(xvaf['v'])
    corra = correlation(xvaf['a'])
    corrva = correlation(xvaf['v'],xvaf['a'])
    if physical:
        m = kT/corrv[0]
    else:
        m = 1
        kT = corrv[0]
    if free_energy:
        if bins == 'kde':
            pos,fe,force = compute_pmf_kde(xvaf['x'].values,dx=0.01,kT=kT)
            
            if verbose:
                plt.plot(pos,fe)
                plt.show()
            
            def dU(x,force=force,pos=pos):
                force_trj = np.zeros(len(x))
                for i in range(len(x)):
                    idx = bisection(pos,x[i])
                    force_trj[i] = force[idx]
                return force_trj
        else:
            #will not work if we don't have memtools installed: https://github.com/jandaldrop/memtools
            mem = mp.Igle(xvaf, kT = kT, trunc = trunc,verbose = False) 
            mem.compute_fe(bins=bins)
            #mem.compute_u_corr()
            dU=mem.dU
            
        if mori:
            fit,pcov = optimize.curve_fit(harm,pos,fe,p0=(1))
            if verbose:
                print(fit)
                plt.plot(pos,fe)
                plt.plot(pos,harm(pos,fit))
                plt.show()
            force_array=xvaf['x'].values*fit
        else:
            force_array=dU(xvaf['x'].values)
    else:
        
        dU=lambda x: 0
        force_array = np.zeros(len(xvaf))
    
    corrvU = correlation(xvaf['v'],force_array)
    corraU = correlation(xvaf['a'],force_array)
        

    tmax=int(trunc/dt)
    corrv = corrv[:tmax]
    corra = corra[:tmax]
    corrva = corrva[:tmax]
    corrvU = corrvU[:tmax]
    corraU = corraU[:tmax]
    kernel = np.zeros(len(corrv))

    if verbose:
        print("truncated after " + str(len(kernel)) + " steps")
    prefac = 1./corrv[0]
   
    if verbose:
        print('calculated mass = ' + str(m))
    kernel[0] = (m*corra[0] + corraU[0] )/corrv[0] #+corraU!!

    for i in range(1,len(kernel)-1):

        kernel[i] = 2*prefac*(- m*corrva[i]/dt - (kernel[0]/2)*corrv[i] - corrvU[i]/dt  -  np.sum(kernel[1:i+1]*corrv[:i][::-1]))

    ik = integrate.cumtrapz(kernel, dx = dt, initial = 0) #integrated kernel
    index_kernel = np.arange(0,len(kernel)*dt,dt)
    
    return corrv, corrva, corra, corrvU, corraU,index_kernel, kernel, ik,dU



def extract_kernel_G(xvaf,trunc,kT=2.494,bins='kde',physical=False,free_energy=True,mori=True,verbose=False,half_stepped=False):
    
    dt = xvaf["t"].iloc[1] - xvaf["t"].iloc[0]
    if verbose:
        print('dt = ' + str(dt))
    
    corrv = correlation(xvaf['v'])
    if physical:
        m = kT/corrv[0]
    else:
        m = 1
        kT = corrv[0]
    if verbose:
        print('calculated mass = ' + str(m))

    if free_energy:
        if bins == 'kde':
            pos,fe,force = compute_pmf_kde(xvaf['x'].values,dx=0.01,kT=kT)
            
            if verbose:
                plt.plot(pos,fe)
                plt.show()
            
            def dU(x,force=force,pos=pos):
                force_trj = np.zeros(len(x))
                for i in range(len(x)):
                    idx = bisection(pos,x[i])
                    force_trj[i] = force[idx]
                return force_trj
        else:
            mem = mp.Igle(xvaf, kT = kT, trunc = trunc,verbose = False) 
            mem.compute_fe(bins=bins)
            #mem.compute_u_corr()
            dU=mem.dU
        
        
        if mori:
            fit,pcov = optimize.curve_fit(harm,pos,fe,p0=(1))
            if verbose:
                print(fit)
                plt.plot(pos,fe)
                plt.plot(pos,harm(pos,fit))
                plt.show()
            force_array=xvaf['x'].values*fit
        else:
            force_array=dU(xvaf['x'].values)
            
    else:
        
        dU=lambda x: 0
        force_array=np.zeros(len(xvaf))
        
    corrxU = correlation(xvaf['x'],force_array)
        
    tmax=int(trunc/dt)
    corrv = corrv[:tmax]
    corrxU = corrxU[:tmax]
    kernel_i =  np.zeros(min(len(corrv),trunc))
    
    if verbose:
        print("truncated after " + str(len(kernel_i)) + " steps")
        
    if half_stepped:
        kernel_i_half = np.zeros(min(len(corrv),trunc))
        kernel = np.zeros(min(len(corrv),trunc))

        prefac = 1/(corrv[0] + corrv[1])
        kernel_i_half[0] = 0
        for i in range(1,len(kernel_i_half)-1):
            kernel_i_half[i] = (-2*m*(corrv[i]-corrv[0])/dt-np.sum(kernel_i_half[1:i+1][::-1]*(corrv[:i] + corrv[1:i+1])))*prefac
            kernel_i_half[i] -= 2*(corrxU[0]/dt - corrxU[i]/dt)*prefac

        kernel[0] = 2*kernel_i_half[1]/dt
        for i in range(1,len(kernel_i_half)-2):
            kernel[i] = (kernel_i_half[i+1] - kernel_i_half[i])/dt

        kernel_i = kernel_i_half                       


    else:

        prefac = 1/corrv[0]
        for i in range(1,len(kernel_i)):
            
            kernel_i[i] = -np.sum(kernel_i[:i]*(corrv[1:i+1])[::-1]) - m*(corrv[i]-corrv[0])/dt - (corrxU[0] -corrxU[i])/dt
            kernel_i[i] = kernel_i[i]*prefac*2
        

            kernel  = np.gradient(kernel_i,dt)  
    index_kernel = np.arange(0,len(kernel)*dt,dt)
        
    return corrv, corrxU,index_kernel, kernel, kernel_i,dU



def compute_pmf_kde(x,dx,kT):
    gauss_kernel = stats.gaussian_kde(x)
    pos = np.arange(np.min(x),np.max(x),dx)
    Z = np.reshape(gauss_kernel(pos),len(pos)) #histogram via kernel density estimator

    fe=-np.log(Z[np.nonzero(Z)])*kT
    pos= pos[np.nonzero(Z)]
    fe -=np.min(fe)
    
    force = np.gradient(fe,dx)
    return pos,fe,force


def bisection(array,value):
        '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
        and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
        to indicate that ``value`` is out of range below and above respectively.'''
        n = len(array)
        if (value < array[0]):
            return 0#-1
        elif (value > array[n-1]):
            return n-1
        jl = 0# Initialize lower
        ju = n-1# and upper limits.
        while (ju-jl > 1):# If we are not yet done,
            jm=(ju+jl) >> 1# compute a midpoint with a bitshift
            if (value >= array[jm]):
                jl=jm# and replace either the lower limit
            else:
                ju=jm# or the upper limit, as appropriate.
            # Repeat until the test condition is satisfied.
        if (value == array[0]):# edge cases at bottom
            return 0
        elif (value == array[n-1]):# and top
            return n-1
        else:
            return jl
        