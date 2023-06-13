import numpy as np
import matplotlib.pyplot as plt
from mempred import *
import mempred as mp
from scipy import optimize
from scipy import integrate
from scipy import stats
from scipy import interpolate
import pandas as pd
import tidynamics

def harm(x,k):
    return k/2*x**2

def create_xva(x,t):
    dt = t[1]-t[0]
    #v =np.gradient(x,dt,edge_order=2)
    v = np.zeros(len(x))
    v[0] = x[1]/dt
    v[1:] = (x[1:] -  x[:-1])/(dt)
    a = np.zeros(len(x))
    a[0] = x[1]/dt**2
    a[1:-1] = (x[2:] - 2*x[1:-1] + x[:-2])/(2*dt**2)
    xvaf = pd.DataFrame(np.array([t[1:-1],x[1:-1],v[1:-1], a[1:-1]]).T,
                   columns=['t','x', 'v', 'a'])
    return xvaf

def extract_kernel_tpf(xvaf,trunc,kT=2.494,bins='kde',physical=False,free_energy=True,mori=True,half_stepped=False,verbose=False,mode=0):
    
    dt = xvaf["t"].iloc[1] - xvaf["t"].iloc[0]
    if verbose:
        print('dt = ' + str(dt))
    
    corrv = mp.correlation(xvaf['v'])
    corra = mp.correlation(xvaf['a'])
    corrva = mp.correlation(xvaf['v'],xvaf['a'])
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
            
            
        else:
            
            pos,fe,force,force_array = extract_free_energy(xvaf['x'].values,bins=bins,kT=kT)
        
        
        if mori:
            #fit,pcov = optimize.curve_fit(harm,pos,fe,p0=(1))
            fit = np.mean(xvaf["v"].values**2)/(np.mean((xvaf['x'].values - np.mean(xvaf['x'].values))**2))
            if verbose:
                print(fit)
                plt.plot(pos,fe)
                plt.plot(pos,harm(pos,fit))
                plt.show()

            def dU(x,force=0,pos=0):
                return fit*(x-np.mean(xvaf['x'].values))

            force_array=(xvaf['x'].values-np.mean(xvaf['x'].values))*fit
        else:

            def dU(x,force=force,pos=pos):
                try:
                    force_trj = np.zeros(len(x))
                    for i in range(len(x)):
                        idx = bisection(pos,x[i])
                        force_trj[i] = force[idx]
                except:
                      idx = bisection(pos,x)
                      force_trj = force[idx]  
                      
                return force_trj

            if bins == 'kde': 
                force_array = dU(xvaf['x'].values)
            else:
                force_array=force_array
            
    else:
        
        dU=lambda x: 0
        force_array=np.zeros(len(xvaf))
    
    corrvU = mp.correlation(xvaf['v'],force_array)
    corraU = mp.correlation(xvaf['a'],force_array)
        

    tmax=int(trunc/dt)
    corrv = corrv[:tmax]
    corra = corra[:tmax]
    corrva = corrva[:tmax]
    corrvU = corrvU[:tmax]
    corraU = corraU[:tmax]
    kernel = np.zeros(len(corrv))

    if verbose:
        print("truncated after " + str(len(kernel)) + " steps")
    
   
    if verbose:
        print('calculated mass = ' + str(m))

    
    if half_stepped:
        kernel_half = np.zeros(len(corrv)) 
        prefac = 1/(corrv[0] + corrv[1])
    
        kernel_half[0] = 0
        for i in range(1,len(kernel_half)-1):
            kernel_half[i] = (-2*(m*corrv[i] + corrvU[i])/dt-np.sum(kernel_half[1:i+1][::-1]*(corrv[:i] + corrv[1:i+1])))*prefac

        kernel[0] = kernel_half[1]*2
        for i in range(1,len(kernel)-2):
            kernel[i] = kernel_half[1:][i]/2 + kernel_half[1:][i+1]/2

        
    else:
        prefac = 1./corrv[0]
        if mode ==0:
            kernel[0] = (m*corra[0] + corraU[0] )/corrv[0] #+corraU!!
        else:
            kernel[0] = (m*corra[0] - corraU[0] )/corrv[0]

        for i in range(1,len(kernel)-1):

            kernel[i] = 2*prefac*(- m*corrva[i]/dt - (kernel[0]/2)*corrv[i] - corrvU[i]/dt  -  np.sum(kernel[1:i+1]*corrv[:i][::-1]))

    ik = integrate.cumtrapz(kernel, dx = dt, initial = 0) #integrated kernel
    index_kernel = np.arange(0,len(kernel)*dt,dt)
    
    return corrv, corrva, corra, corrvU, corraU,index_kernel, kernel, ik,dU

def fit_vacf_discr(t,a, c, tau1,K,B):

    dt = t[1]-t[0]
    msd = msd_biexpo_delta_harm(a=a, b=0, c=c, tau1=tau1, tau2=tau1, K=K, B=B, t=t)
    cvv_exp = np.zeros(len(msd))
    cvv_exp[0] = msd[1]/dt**2
    cvv_exp[1:-1] = (msd[2:] - 2*msd[1:-1] + msd[:-2])/(2*dt**2)
    #cvv_exp[:-2] = (msd[2:] - 2*msd[1:-1] + msd[:-2])/(2*dt**2)

    return cvv_exp

def fit_msd_discr(t,a, c, tau1,K,B):

    dt = t[1]-t[0]
    msd = msd_biexpo_delta_harm(a=a, b=0, c=c, tau1=tau1, tau2=tau1, K=K, B=B, t=t)
    msd[0]=0
    
    return msd

def extract_kernel_estimator(xvaf,trunc,p0,bounds,end=100,no_fe = False,physical=False,verbose=False,fit_msd=False):
    dt = xvaf["t"].iloc[1] - xvaf["t"].iloc[0]
    if verbose:
        print('dt = ' + str(dt))
    
    corrv = mp.correlation(xvaf['v'])
    msd = tidynamics.msd(xvaf['x'].values)
    msd[0]=0
    t=np.arange(0,len(corrv)*dt,dt)

    popt1 = p0
    if fit_msd:
        popt1,pcov1 = optimize.curve_fit(fit_msd_discr,t[:end],msd[:end],p0=p0,maxfev=10000,bounds=bounds)
    else:
        popt1,pcov1 = optimize.curve_fit(fit_vacf_discr,t[:end],corrv[:end],p0=p0,maxfev=10000,bounds=bounds)
    a,c,tau1,K,B = popt1
    t=np.arange(0,trunc*dt,dt)
    kernel = a*np.exp(-t/tau1)
    kernel[0]+=c*2/dt
    kernel_i = a*tau1*(1-np.exp(-t/tau1))
    kernel_i+=c

    if verbose:
        if fit_msd:
            plt.scatter(t[:end],msd[:end],s=10,edgecolor='k',facecolor='')
            plt.plot(t[:end],fit_msd_discr(t[:end],*popt1),'r--')
            plt.axhline(y=0,color='k')
            plt.show()
        else:
            plt.scatter(t[:end],corrv[:end],s=10,edgecolor='k',facecolor='')
            plt.plot(t[:end],fit_vacf_discr(t[:end],*popt1),'r--')
            plt.axhline(y=0,color='k')
            plt.show()

    def dU(x,force=0,pos=0):
        if no_fe:
            return 0
        else:
            return K*(x-np.mean(xvaf['x'].values))
    kT = B
    return corrv, t, kernel, kernel_i,dU,kT,popt1


def extract_kernel_tpf_G(xvaf,trunc,kT=2.494,bins='kde',physical=False,free_energy=True,mori=True,verbose=False,half_stepped=False):
    
    dt = xvaf["t"].iloc[1] - xvaf["t"].iloc[0]
    if verbose:
        print('dt = ' + str(dt))
    
    corrv = mp.correlation(xvaf['v'])
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
            
            
        else:
            
            pos,fe,force,force_array = extract_free_energy(xvaf['x'].values,bins=bins,kT=kT)
            

        if mori:
            #fit,pcov = optimize.curve_fit(harm,pos,fe,p0=(1))
            fit = np.mean(xvaf["v"].values**2)/(np.mean((xvaf['x'].values - np.mean(xvaf['x'].values))**2))
            if verbose:
                print(fit)
                plt.plot(pos,fe)
                plt.plot(pos,harm(pos,fit))
                plt.show()

            def dU(x,force=0,pos=0):
                return fit*(x-np.mean(xvaf['x'].values))

            force_array=(xvaf['x'].values-np.mean(xvaf['x'].values))*fit
        else:

            def dU(x,force=force,pos=pos):
                try:
                    force_trj = np.zeros(len(x))
                    for i in range(len(x)):
                        idx = bisection(pos,x[i])
                        force_trj[i] = force[idx]
                except:
                      idx = bisection(pos,x)
                      force_trj = force[idx]  

                return force_trj

            if bins == 'kde': 
                force_array = dU(xvaf['x'].values)
            else:
                force_array=force_array
                
            
    else:
        
        dU=lambda x: 0
        force_array=np.zeros(len(xvaf))
        
    corrxU = mp.correlation(xvaf['x'],force_array)
        
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


def extract_free_energy(x,bins=100,kT=2.494): 
    #one-dimensional
    hist,edges=np.histogram(x, bins=bins, density=True)
    pos =(edges[1:]+edges[:-1])/2
    pos = pos[np.nonzero(hist)]
    hist = hist[np.nonzero(hist)]
    fe=-np.log(hist[np.nonzero(hist)])*kT

    fe_spline=interpolate.splrep(pos, fe, s=0, per=0)

    force=interpolate.splev(pos, fe_spline, der=1)
    force_array=interpolate.splev(x, fe_spline, der=1)

    return pos,fe,force,force_array


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
        