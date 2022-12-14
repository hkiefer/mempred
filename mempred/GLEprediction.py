import numpy as np
import pandas as pd

import math
import matplotlib.pyplot as plt
from mempred import *
import mempred as mp

from scipy.optimize import curve_fit
from scipy import stats

import warnings
warnings.filterwarnings('ignore')


class GLEPrediction:
    """
    Performing predictions of time-Series using the generalized Langevin equation
    This class includes
    - the extraction of correlation functions, free energy and memory kernel from a given trajectory (performed by memtools © Jan Daldrop, Florian Brüning) which is integrated in mempred
    - the reconstruction of random colored noise depending on the given memory kernel and history
    - prediction of the future trajectory depending on the past and memory
    - prediction using Langevin Equation with non-linear potential
    """
    def __init__(self, bins = "auto", cut = 1000, trunc = 80, dt = 1, last_value_correction = False, no_fe = False, plot_pred = False,physical=False,kde_mode = True,mori=False):
        self.trunc = trunc #depth of memory Kernel
        self.bins = bins #number of bins for the histogram (potential-extraction)
        self.dt = dt #time-step for Extraction and GLE Simulation (usually dt = 1 is used)
        self.no_fe = no_fe #exclude free energy term in GLE
        self.cut = cut #cut : length of historical trajectory for memory extraction
        self.last_value_correction = last_value_correction #Correction of Prediction-Trajectory
        self.plot_pred = plot_pred #Plots the Prediction automatically
        self.physical = physical #if True, mass m is calculated from equipartition theory, otherwise it's m =1
        self.kde_mode = kde_mode #use kernel density estimator for free energy calculation
        self.mori = mori #use a quadratic potential of mean force (fitted from data)
    def create_xvas(self, trj_array, time_arg=None): #Creating x-v-a-Dataframe for given x-trajectory (needed for memory extraction)
        xva_array=[]
        for trj in trj_array:
            if time_arg is None:
                time=np.arange(0,int(len(trj)*self.dt),self.dt)
            else:
                time=time_arg
            xf=mp.xframe(trj, time, fix_time=True)
            xvaf=mp.compute_va(xf)
            xva_array.append(xvaf)
        return xva_array
    
    def create_xvs(self, trj_array, time_arg=None): #Creating x-v-Dataframe for given x-trajectory (needed for prediction)
        xv_array=[]
        for trj in trj_array:
            if time_arg is None:
                time=np.arange(0,int(len(trj)*self.dt),self.dt)
            else:
                time=time_arg
            xf=mp.xframe(trj, time, fix_time=True)
            xvf=mp.compute_v(xf)
            xv_array.append(xvf)
        return xv_array
    
    def extractKernel(self, trj_array, time = None, G_method=True,fit_kernel = False, plot_kernel = False,fit_start = 0, fit_end = 3,kT = 2.494,p0 = [1,1,1,167,0,0]): 
        #Memory Kernel Extraction (with Volterra scheme)
        
        if not time is None:
            self.dt = time[1] - time[0]
        #print('found dt = ' + str(self.dt))
        trj_array[0] = trj_array[0][:self.cut]
 
        xva_array = self.create_xvas(trj_array, time)
        xva_array[0]=xva_array[0].reset_index()
        
        self.kernels = []
        self.corrv = mp.correlation(xva_array[:self.cut][0]['v'].values)
       
        
        if self.physical:
            self.m = kT/self.corrv[0]
        
        else:
            self.m = 1
            kT = self.corrv[0]
            
        if self.kde_mode:
            self.bins = "kde"
            
        if G_method:
            self.mem = mp.extract_kernel_G(xva_array[0][:self.cut],trunc=self.trunc,kT=kT,bins=self.bins,physical=self.physical,free_energy=not self.no_fe,mori=self.mori,verbose=False,half_stepped=False)
            self.kernel=self.mem[3]
            self.ikernel=self.mem[4]
            self.kernel_real = self.mem[3]
            self.kernel_index = self.mem[2]
        else:
            self.mem = mp.extract_kernel(xva_array[0][:self.cut],trunc=self.trunc,kT=kT,bins=self.bins,physical=self.physical,free_energy=not self.no_fe,mori=self.mori,verbose=False)
            self.kernel=self.mem[6]
            self.ikernel=self.mem[7]
            self.kernel_real = self.mem[6]
            self.kernel_index = self.mem[5]

        dU = self.mem[-1] #this we should not use for prediction
        
        if not self.no_fe:
            if self.kde_mode:
                
                pos,fe,force = mp.compute_pmf_kde(xva_array[0]['x'].values,dx=0.01,kT=kT)
                
                def dU(x,force=force,pos=pos):
                    idx = self.bisection(pos,x)
                    value = force[idx]
                    return value
                
            else:
                dU_memtools = dU
                #new potential interpolation
                self.x_min = np.min(trj_array[:self.cut]) #smaller value see large barrier
                self.x_max = np.max(trj_array[:self.cut]) #higher values are unconfined!

                def dU(x):
                    if x < self.x_min:
                        fc =dU_memtools(self.x_min)
                        mc = (((dU_memtools(self.x_min+0.0001) - fc) / 0.0001)**2)**0.5
                        #print(fc, mc)
                        value = mc*x + fc - mc*self.x_min #(harmonic barrier)
                    elif x > self.x_max:
                        fc =dU_memtools(self.x_max)
                        mc = (((dU_memtools(self.x_max+0.0001) - fc) / 0.0001)**2)**0.5
                        #print(fc, mc)
                        value = mc*x + fc - mc*self.x_max #(harmonic barrier)
                    else:
                        value = dU_memtools(x)
                    return value
            
        
        self.p0 = np.array(p0)
        popt = p0
        pcov = np.zeros(len(p0))
        self.kernel_data = self.kernel_real.copy()
        
        if fit_kernel: #Sometimes it is better to fit the kernel as an exponential decay and an oscillation
            self.p0[2]=self.kernel[1]/math.e
            start = fit_start
            steps = np.arange(fit_start+1, fit_end + 1)
            popt, pcov, fitted_kernel = self.fitted_kernel(self.kernel_index, self.kernel_real, start)
            
            RMSE = self.RMSE(fitted_kernel, self.kernel_real)
            for step in steps: #find optimal fit with length of delta function at start point
                
                opt, cov, fit = self.fitted_kernel(self.kernel_index, self.kernel_real, step)
                error = self.RMSE(fit,self.kernel_real)
                if error < RMSE:
                    RMSE = error
                    fitted_kernel = fit
                    popt = opt
                    pcov = cov
                else:
                    continue 
                self.kernel_real = fitted_kernel
            #print('fitted memory time: ' + str(np.absolute(np.round(1/popt[1],2))) + ' time units')
            #print('fitted osc. time: ' + str(np.absolute(np.round(popt[3],2))) + ' time units')
        
        
        if plot_kernel:
            print('plotting extracted memory kernel...')
            plt.scatter(self.kernel_index,self.kernel, s = 5)
            if fit_kernel:
                plt.plot(self.kernel_index, self.kernel_real, 'g--', lw = 2)
            plt.xlabel("t", fontsize = 'x-large')
            plt.ylabel("$\\Gamma(t)$", fontsize = 'x-large')
            plt.title('extracted memory kernel of trajectory')
            plt.tick_params(labelsize="x-large")
            plt.show()
            plt.close()
            
            print('plotting running integral of kernel...')
            plt.scatter(self.kernel_index,self.ikernel, s = 5)
            
            plt.xlabel("t", fontsize = 'x-large')
            plt.ylabel("G(t)", fontsize = 'x-large')
            plt.title('running integral of kernel')
            plt.tick_params(labelsize="x-large")
            plt.show()
            plt.close()
        
        self.popt = popt
        #Important for Prediction Class
        self.dU = dU
        self.integrate=IntegrateGLE_RK4(kernel = self.kernel_real,
                                    t = self.kernel_index, dt = self.dt, dU=self.dU, m = self.m)
        
        return self.mem,self.kernel_index, self.kernel_real, self.kernel_data,self.ikernel, self.dU,self.popt
    
    def set_kernel(self,p,noise=True,trunc=None):
        kernel_fit = self.func(self.kernel_index,*p)
        kernel_fit[0] = self.kernel_data[0]
        if noise:
            fit2 = self.func(self.kernel_index,*self.popt)
            noise = self.kernel_data  - fit2
            noise[0]*=0
            kernel_fit = kernel_fit + noise
            
        if trunc < self.trunc:
            self.trunc = trunc
            kernel_fit = np.append(kernel_fit[:self.trunc],np.zeros(len(kernel_fit)-self.trunc))
        self.kernel_real = kernel_fit
        self.integrate=IntegrateGLE_RK4(kernel = self.kernel_real,
                                    t = self.kernel_index, dt = self.dt, dU=self.dU, m = self.m)
        return self.kernel_real
    
    def predictGLE(self, trj_array, xvaf_seas = None,time = None, n_steps = 1, n_preds = 1, return_full_trjs = False, zero_noise = False, Langevin  = False, alpha = 1,cond_noise = None):
        actual = np.array(trj_array)
       
        self.alpha = alpha
        
        trj_array[0] = trj_array[0][:self.cut]
        xva_array = self.create_xvas(trj_array, time)
        zero_row = np.zeros((2,len(xva_array[0].columns)))
        top_row = pd.DataFrame(zero_row,columns=xva_array[0].columns)
        xva_array[0] = pd.concat([top_row, xva_array[0]]).reset_index(drop = True)
        xva_array[0]['x'] = xva_array[0]['x'].shift(-1) 
        xva_array[0]['x'].values[-1] = trj_array[0][-1]
        xva_array[0]=xva_array[0].reset_index()
        
        corr_fr_all = self.kernel_real

        if xvaf_seas is None:
            xvaf_seas = np.zeros(n_steps+self.cut)
            xvaf_seas=mp.xframe(xvaf_seas,np.arange(0,len(xvaf_seas)*self.dt,self.dt),fix_time=True)
            xvaf_seas['v'] = np.gradient(xvaf_seas['x'],self.dt)
            xvaf_seas['a'] = np.gradient(xvaf_seas['v'],self.dt)

        if not cond_noise is None:
            self.t_h = cond_noise #for cond noise generation, if cond_noise is None: we use a general generation technique (see RK4 class)
            #if self.trunc <= self.t_h:
                #self.t_h = self.trunc
            #print('use conditional random noise generator')
            #get last values of the historical random force
            
            t_fr, fr_all, corr_fr_all,fr_hist= self.compute_hist_fr(xva_array[0], xvaf_seas,self.kernel_index,self.kernel_real, self.dt,t_h = self.t_h)
        
        trj_pred_array = []           
        #xva_array[0]['v'] = np.append(0,(xva_array[0]['v'].values)[:-1]) 
        for i in range(0,n_preds):
            for cxva, trj in zip(xva_array, trj_array): 
                
                x0 = cxva.iloc[-1]["x"] #initial values are the last known values
                v0 = cxva.iloc[-1]["v"]
                cr = np.zeros(len(cxva)+n_steps)
                predef_x = np.zeros(len(cxva)+n_steps)
                predef_x[:len(cxva)] = cxva["x"]
           
                predef_v = np.zeros(len(cxva)+n_steps)
                predef_v[:len(cxva)] = cxva["v"]
                

                if not cond_noise is None:
                    cond_noise = np.zeros(len(predef_v))
                    #generate future steps conditional Gaussian process, starting at last known step of historical noise
                    noise,noise_g = self.gen_noise_cond(corr_fr_all,fr_hist,n_steps = n_steps)
                    cond_noise[len(cxva):] = noise[:n_steps]
                    
                
                trj_pred, _, _ = self.integrate.integrate(n_steps+len(cxva), x0=x0, v0=v0,
                                     predef_v=predef_v, predef_x=predef_x, xvaf_seas = xvaf_seas,
                                     zero_noise=zero_noise, n0=len(cxva), Langevin = Langevin, alpha = alpha,custom_noise_array = cond_noise)
                if self.last_value_correction:
                    try:
                        trj_pred[len(cxva):] = trj_pred[len(cxva):] + trj.values[-1] - trj_pred[len(cxva)]
                    except:
                        trj_pred[len(cxva):] = trj_pred[len(cxva):] + trj[-1] - trj_pred[len(cxva)]

                if not return_full_trjs:
                    trj_pred = trj_pred[len(cxva)+1:]
            
                trj_pred_array.append(trj_pred)
            if i == 0:
                trj_p = np.array([trj_pred_array[0]])
                continue
            elif i >= 1:
                pred = np.array([trj_pred_array])
                trj_p = np.concatenate((trj_p, pred[0]), axis = 0)
        trj_p_mean = np.mean(trj_p, axis = 0)
        error = np.std(trj_p, axis = 0)    

        index_pred = np.arange(0,len(trj_p_mean))     
        trj_p_pred = trj_p_mean[self.cut:]
        actual_plot = actual[0][self.cut:(self.cut + n_steps)]
        
        
        fr_trj = np.zeros(self.cut+n_steps)
        if not cond_noise is None:
            #fr_trj[self.cut-self.t_h:self.cut] = fr_hist
            fr_trj[self.cut-100-self.t_h:self.cut] = fr_all
            fr_trj[self.cut:self.cut+n_steps] = noise[:n_steps]
        
        if self.plot_pred:
     
            #plots prediction 
            print('plotting prediction...')
            plt.plot(actual[0], color="red", lw=2)
            plt.plot(actual[0][:self.cut], color="k", lw=2)
            plt.plot(index_pred[self.cut:],trj_p_mean[self.cut:], lw=2, label="Prediction")
            plt.fill_between(index_pred[self.cut:], (trj_p_mean-error)[self.cut:], (trj_p_mean+error)[self.cut:], color = "blue", alpha = 0.3)
            plt.axvline(x = len(xva_array[0]["x"].values), linestyle = "--")
            #already plotted above
            plt.legend(loc="best", fontsize="x-large", ncol=2)
            plt.title('predicted trajectory')
            plt.xlabel("t", fontsize="x-large")
            plt.ylabel("x", fontsize="x-large")
            plt.tick_params(labelsize="x-large")
            plt.show()
            plt.close()
            
            
            index = np.arange(0, n_steps, 1)
            plt.plot(index[:len(actual_plot)], actual_plot, color = "red", lw=2)
            plt.plot(index, trj_p_mean[self.cut:], lw=2, label = "Prediction")
            plt.fill_between(index, (trj_p_mean-error)[self.cut:], (trj_p_mean+error)[self.cut:], color = "blue", alpha = 0.3)
            plt.legend(loc="best", fontsize="x-large", ncol=2)
            plt.title('predicted trajectory')
            plt.xlabel("t", fontsize="x-large")
            plt.ylabel("x", fontsize="x-large")
            plt.tick_params(labelsize="x-large")
            plt.show()
            plt.close()
             
        return index_pred, trj_p_mean, trj_p_mean[self.cut:], error, actual[0], actual_plot, fr_trj, corr_fr_all
    
    def func(self,x, a, b, c,d ,e,f):
        #return a * np.exp(-b * x) + c
        return a*np.exp(-b*x) + c*np.cos((2*np.pi)*x/d  - e) + f
        
    
    def fitted_kernel(self, index, kernel, start):
        popt, pcov = curve_fit(self.func, index[start:], kernel[start:], bounds = (-np.inf, np.inf), maxfev=10000,p0 = self.p0) 

        fitted_kernel = np.append(kernel[:start], self.func(index[start:], *popt))

        return popt, pcov, fitted_kernel

    def RMSE(self, pred, real):
        return np.mean((pred - real)**2)**0.5   
 
    #uncoupled from RK4, because we need xvaf
    def compute_hist_fr(self,xvaf,xvaf_seas,t, kernel, dt,t_h = 100): 
    #Calculates random noise from given trajectory and extracted Kernel

        N = len(kernel)

        xvaf = xvaf[-(t_h+100):]
        xvaf_seas = xvaf_seas[-(t_h+100):]
        M = len(xvaf)
        if N < M:
            kernel = np.append(kernel,np.zeros(int(M - N)))
        x = np.array(xvaf["x"])
        v = np.array(xvaf["v"])
        a = np.array(xvaf["a"])

        x_seas = np.array(xvaf_seas["x"])
        v_seas = np.array(xvaf_seas["v"])
        a_seas= np.array(xvaf_seas["a"])
        
        m = self.m
        self.kT = m*self.corrv[0]*self.alpha
        
        fr = np.zeros(np.min((len(a), len(kernel))))
        fr[0] = m*a[0] + self.dU(x[0]) + m*a_seas[0] + self.dU(x_seas[0])
        for i in range(1,len(fr)):
            
                fr[i] =  m*a[i]  + 0.5*dt*kernel[0]*v[i] + 0.5*dt*kernel[i]*v[0] + dt*np.sum(kernel[1:i]*v[1:i][::-1])+ self.dU(x[i])
                fr[i] += m*a_seas[i] + 0.5*dt*kernel[0]*v_seas[i] + 0.5*dt*kernel[i]*v_seas[0] + dt*np.sum(kernel[1:i]*v_seas[1:i][::-1])+ self.dU(x_seas[i])

        fr_hist=fr[-t_h:]
        t_fr = np.arange(0,len(fr_hist)*dt,dt)
        corr_fr = mp.correlation(fr)[:self.trunc]
        return t_fr, fr, corr_fr,fr_hist #corr_fr will be used as covariance function for random force and fr_hist as last known values
    
    def gen_noise_cond(self,fr_corr,fr_hist,n_steps):
        
        fr_corr = np.append(fr_corr[:self.t_h],np.zeros(int(np.abs(len(fr_corr)-self.t_h))))           
        if n_steps > int(len(fr_corr)-self.t_h): #we use fr_hist with length trunc, which results in 0 n_steps for noise gen
            fr_corr = np.append(fr_corr, np.zeros(n_steps)) #after trunc, the kernel has to be decayed to zero!!
            
        if n_steps < int(len(fr_corr)-self.t_h):
            fr_corr = fr_corr[:int(n_steps+self.t_h)]
            
        N = len(fr_corr)

        C=self.alpha*(np.triu(np.array([np.roll(fr_corr, i) for i in range(0,len(fr_corr))]))+\
        np.triu(np.array([np.roll(fr_corr, i) for i in range(0,len(fr_corr))]),1).T)

        #t_h points of the process are given from previous section (execute previous section)
        past_force=fr_hist[-self.t_h:]
        #past_force=fr_hist
        
        # partition C
        a=len(past_force)
        b=N-a

        Caa=C[:a,:a]; Cab=C[:a,a:]; Cbb=C[a:,a:]
        Caa_inv=np.linalg.inv(Caa)
        Ctilde=Cbb-np.dot(Cab.T, np.dot(Caa_inv, Cab))
        mu=np.dot(Cab.T, np.dot(Caa_inv, past_force))
        
        # Compute Cholesky decomposition of conditional covariance matrix
        L=np.linalg.cholesky(Ctilde).real

        # Generate Gaussian process
        samp=np.random.normal(0,np.ones((b,)))
        GP=np.dot(L, samp)+mu #generated noise
        GP2=np.concatenate((past_force, GP)) #hist noise + gen noise

        return GP, GP2
    
    def bisection(self,array,value):
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


class IntegrateGLE_RK4: #Class for GLE Integration with Runge-Kutta 4
    def __init__(self, kernel, t, dt, m=1, dU = lambda x: 0.):
        self.kernel = kernel #extracted Kernel
        self.t = t #time array of Kernel
        self.m = m #mass
        #self.dt = self.t[1] - self.t[0]
        self.dt = dt #time-step for Prediction (usually same as memory kernel extraction)
        self.dU = dU #extracted free Energy for Prediction
        
    def integrate(self, n_steps, x0 = 0., v0 = 0., zero_noise = False, predef_x = None, predef_v = None, xvaf_seas=None,n0 = 0, custom_noise_array = None, Langevin = False, alpha = 1):
        
        x = x0
        v = v0
        
        self.alpha = alpha
        if predef_x is None:
            self.x_trj = np.zeros(n_steps)
        else: 
            assert (len(predef_x) == n_steps)
            assert (predef_x[n0 - 1] == x)
            self.x_trj = predef_x
           
        if predef_v is None:
            self.v_trj = np.zeros(n_steps)
        else: 
            assert (len(predef_v) == n_steps)
            assert (predef_v[n0 - 1] == v)
            self.v_trj = predef_v
        
        self.t_trj = np.arange(0., n_steps * self.dt, self.dt)

        #is usually set to zero!
        x_seas_trj = xvaf_seas['x'].values[:n_steps]
        v_seas_trj = xvaf_seas['v'].values[:n_steps]
        a_seas_trj = xvaf_seas['a'].values[:n_steps]
        
        if zero_noise:
            noise = np.zeros(n_steps)
            
        else:
            
            if custom_noise_array is None:
                if Langevin == False:
                    #Important because we append the known trajectory before the Prediction
                    noise_array = np.zeros(n0 + 2)
                
                    #Generating noise (which shouldn't be used)
                
                    noise = self.gen_noise(self.kernel, self.t, self.dt, n_steps = n_steps - n0)
                
                    noise_array = np.append(noise_array, noise)
                    #noise_array = np.append(noise_array, np.zeros(n_steps))
                    noise = noise_array
                
                else: #Optional: To Run a Langevin-Prediction with no memory and white noise
                    gamma = self.kernel[0]
                    self.kernel = np.zeros(n_steps-n0)
                    self.kernel[0] = gamma
            
                    sigma = self.alpha*math.sqrt(self.kernel[0]*2.494) #with kT
                    white_noise = np.zeros(n0 + 2)
                    for i in range(n0,n_steps):
                        white_noise = np.append(white_noise, math.sqrt(1/self.dt)*np.random.normal(0, sigma))
                    noise = white_noise
            
            else:
                
                assert (len(custom_noise_array) == n_steps)
                noise = custom_noise_array
         
                
        
        rmi = 0 #Starting RK4
        
        for i in range(n0, n_steps):
           
            rmi_old = rmi
            if i > 1:
                rmi = self.mem_red_integrand(self.v_trj[:i])
                rmi2 = self.mem_red_integrand(v_seas_trj[:i])
                v_old = self.v_trj[i-1]
     
            else:
                rmi = 0
                rmi_old = 0
                v_old = 0
                
            x, v = self.step_rk4(x,v,rmi,noise[i], v_old, rmi_old,x_seas_trj[i],v_seas_trj[i],a_seas_trj[i],rmi2)
            self.x_trj[i] = x
            self.v_trj[i] = v
            
            
        return self.x_trj, self.v_trj, self.t_trj

    #Function to run RK4-Simulation
    def mem_red_integrand(self, v):
        if len(v) < len(self.kernel):
            v = np.concatenate([np.zeros(len(self.kernel) - len(v) + 1) ,v ])
        integrand = v[:len(v) - len(self.kernel[1:]) - 1:-1] * self.kernel[1:]
        return (0.5 * integrand[-1] + np.sum(integrand[:-1])) * self.dt
    
    def f_rk4(self, x, v, rmi, noise, next_w, last_w, v_old, rmi_old,x_noise,v_noise,a_noise,rmi2):
        nv = v
        na = (-next_w * rmi - last_w * rmi_old - 0.5 * next_w * self.kernel[0]
              * v * self.dt - 0.5 * last_w * self.kernel[0] * v_old * self.dt
              - self.dU(x) + noise - a_noise*self.m - self.dU(x_noise) - 0.5 * next_w * self.kernel[0]
              * v_noise * self.dt -next_w * rmi2) / self.m
            
        return nv, na
        
    def step_rk4(self, x, v, rmi, noise, v_old, rmi_old,x_noise,v_noise,a_noise,rmi2):
        k1x, k1v = self.f_rk4(x, v, rmi, noise, 0.0, 1.0, v_old, rmi_old,x_noise,v_noise,a_noise,rmi2)
        k2x, k2v = self.f_rk4(x + k1x * self.dt / 2, v + k1v * self.dt / 2, rmi,
                             noise, 0.5, 0.5, v_old, rmi_old,x_noise,v_noise,a_noise,rmi2)
        k3x, k3v = self.f_rk4(x + k2x * self.dt / 2, v + k2v * self.dt / 2, rmi,
                             noise, 0.5, 0.5, v_old, rmi_old,x_noise,v_noise,a_noise,rmi2)
        k4x, k4v = self.f_rk4(x + k3x * self.dt, v + k3v * self.dt, rmi, noise,
                             1.0, 0.0, v_old, rmi_old,x_noise,v_noise,a_noise,rmi2)
        return x + self.dt * (
            k1x + 2. * k2x + 2. * k3x + k4x) / 6., v + self.dt * (
                k1v + 2. * k2v + 2. * k3v + k4v) / 6.  
    

    #-----##########-----




    #Function to construct random colored noise (uncondtional, old!!)
    def gen_noise(self,kernel, t, dt, n_steps):
    
        if n_steps > int(len(kernel)/2): #because we cut the kernel and after FT we divide the length of the Kernel by 2
            kernel = np.append(kernel, np.zeros(n_steps*2))
            t = np.arange(0,len(kernel))
            
        N = len(kernel)
        
        corrv = np.loadtxt('corrs.txt', skiprows = 1, usecols = 1)
        Gamma_arr = np.array(kernel)
        #Gamma_arr2=np.concatenate((np.flip(Gamma_arr[1:]),Gamma_arr[:-1]))

        m = self.m
        kT = m*corrv[0]*self.alpha
        #print("kT = ", str(kT))
        
        nhalf=int(np.floor(N/2)+1)
        T=N*dt # data length
        
        omega=np.array([(2*np.pi/T)*k for k in range(1,N+1)])
        #Gamma_arr[0] /=2
        G=kT*np.fft.rfft(Gamma_arr).real

        #G_ft=abs(2*kT*m*(2*T*G[:nhalf].real/N-T*Gamma_arr[0]/N))
        
        #sigma=np.sqrt(G_ft/T)      
        sigma=np.sqrt(2*G/N)     
        omega=omega[:nhalf]
        t=t[:nhalf]
        
        # coefficients of the fourier series
        a=np.random.normal(0,np.absolute(np.nan_to_num(sigma)))
        b=np.random.normal(0,np.absolute(np.nan_to_num(sigma)))
        
        def random_force(t):
            return(np.sum(a*np.cos(omega*t)+b*np.sin(omega*t)))

        random_f_vec=np.vectorize(random_force) # vectorize function

        noise = random_f_vec(t)
        #noise = np.append(noise, white_noise)
        return noise
