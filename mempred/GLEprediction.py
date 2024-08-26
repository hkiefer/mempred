import numpy as np
import pandas as pd

import math
import matplotlib.pyplot as plt
from mempred import *
import mempred as mp
import scipy

import scipy.integrate
from scipy.optimize import curve_fit

import warnings
warnings.filterwarnings('ignore')


class GLEPrediction:
    """
    Performing extraction and predictions of time-Series using the generalized Langevin equation
    This class includes
    - the extraction of correlation functions, free energy and memory kernel from a given trajectory (performed by memtools © Jan Daldrop, Florian Brüning) which is integrated in mempred
    - the reconstruction of random colored noise depending on the given memory kernel and history
    - prediction of the future trajectory depending on the past and memory
    - prediction using Langevin Equation with non-linear potential
    """
    def __init__(self, bins = "auto", cut = 1000, trunc = 80, dt = 1, last_value_correction = False, no_fe = False, plot_pred = False,physical=False,kde_mode = True,mori=False,disc = 1,hs_pred=False):
        
        self.trunc = trunc #int, depth of memory Kernel
        self.bins = bins #int, number of bins for the histogram (potential-extraction)
        self.dt = dt #int, time-step for Extraction and GLE Simulation (usually dt = 1 is used)
        self.no_fe = no_fe #boolean, exclude free energy term in GLE
        self.cut = cut #int, length of historical trajectory for memory extraction
        self.last_value_correction = last_value_correction #boolean, Correction of Prediction-Trajectory
        self.plot_pred = plot_pred #boolean, Plots the Prediction automatically
        self.physical = physical #boolean, if True, mass m is calculated from equipartition theory, otherwise it's m =1, and kT will be set to <v^2>
        self.kde_mode = kde_mode #boolean, use kernel density estimator for free energy calculation
        self.mori = mori #boolean, use a quadratic potential of mean force (fitted from data)
        self.disc = disc #int, discretization_scheme (if disc = 0, half-stepped velocities in extraction scheme, if disc = 1, full-stepped velocities in extraction scheme)
        self.hs_pred = hs_pred #boolean, use half-stepped or full-stepped velocities for prediction (half-stepped is not fixed until now)

    #creates dataframe for position and velocity from trajectory (half-stepped finite difference scheme)
    def compute_xv(self,xf):
        x = xf['x'].values
        t = xf['t'].values
        dt = t[1]-t[0]
        #v =np.gradient(x,dt,edge_order=2)
        v = np.zeros(len(x))
        v[0] = x[1]/dt
        v[1:] = (x[1:] -  x[:-1])/(dt)
        xvf = pd.DataFrame(np.array([t[1:-1] ,x[1:-1] ,v[1:-1] ]).T,
                    columns=['t','x', 'v'])
        return xvf

    #creates dataframe for position,velocity and acceleration from trajectory (half-stepped finite difference scheme)
    def compute_xva(self,xf):
        x = xf['x'].values
        t = xf['t'].values
        dt = t[1]-t[0]
        #v =np.gradient(x,dt,edge_order=2)
        v = np.zeros(len(x))
        v[0] = x[1]/dt
        v[1:] = (x[1:] -  x[:-1])/(dt)
        a = np.zeros(len(x))
        a[0] = x[1]/dt**2
        a[1:-1] = (x[2:] - 2*x[1:-1] + x[:-2])/(2*dt**2)
        #a[1:] = (v[1:] -  v[:-1])/(dt)
        xvaf = pd.DataFrame(np.array([t[1:-1],x[1:-1],v[1:-1], a[1:-1]]).T,
                    columns=['t','x', 'v', 'a'])
        return xvaf

    def create_xvas(self, trj_array, time_arg=None): #Creating x-v-a-Dataframe for given x-trajectory (needed for memory extraction)
        xva_array=[]
        for trj in trj_array:
            if time_arg is None:
                time=np.arange(0,int(len(trj)*self.dt),self.dt)
            else:
                time=time_arg
            xf=mp.xframe(trj, time, fix_time=True)
            
            if self.disc == 0: #half-stepped computation #for kernel extraction
                xvaf = self.compute_xva(xf)
            else:
                xvaf=mp.compute_va(xf) #full step computation #for predition
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
            
            if self.disc == 0:
                xvf=self.compute_xv(xf)
            else:
                xvf=mp.compute_v(xf)
            xv_array.append(xvf)
        return xv_array
    
    #Function to extract memory kernel from trajectory by Volterra Method (see extract_kernel.py), all extracted parameters will be saved in self.mem and used for the prediction integrator
    def extractKernel(self, trj_array, time = None, G_method=False,half_stepped=False,fit_kernel = False, plot_kernel = False,fit_start = 0, fit_end = 3,kT = 2.494,p0 = [1,1,1,167,0,0]): 
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
            self.mem = mp.extract_kernel_tpf_G(xva_array[0][:self.cut],trunc=self.trunc,kT=kT,bins=self.bins,physical=self.physical,free_energy=not self.no_fe,mori=self.mori,verbose=False,half_stepped=half_stepped)
            self.kernel=self.mem[3][:-1]
            self.ikernel=self.mem[4][:-1]
            self.kernel_real = self.mem[3][:-1]
            self.kernel_index = self.mem[2][:-1]
        else:
            self.mem = mp.extract_kernel_tpf(xva_array[0][:self.cut],trunc=self.trunc,kT=kT,bins=self.bins,physical=self.physical,free_energy=not self.no_fe,mori=self.mori,verbose=False,half_stepped=half_stepped)
            self.kernel=self.mem[6][:-1]
            self.ikernel=self.mem[7][:-1]
            self.kernel_real = self.mem[6][:-1]
            self.kernel_index = self.mem[5][:-1]

        #otherwise random force gen could be problematic later on...
        self.kernel -= self.kernel[-1]
        self.ikernel = scipy.integrate.cumtrapz(self.kernel, self.kernel_index, initial=0)

        self.kT = kT
        if not self.no_fe:
            if self.kde_mode:
                
                
                if self.mori:
                    def dU(x,force=0,pos=0):
                        return self.mem[-1](x) - np.mean(xva_array[0]['x'].values)

                else:
                    pos,fe,force = mp.compute_pmf_kde(xva_array[0]['x'].values,dx=0.01,kT=kT)
                    def dU(x,force=force,pos=pos):
                        idx = self.bisection(pos,x)
                        value = force[idx]
                        return value

            else:
                dU_memtools = self.mem[-1]
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
            
        else:
            def dU(x):
                return self.mem[-1](x)
        self.p0 = np.array(p0)
        popt_fit = p0
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
                    popt_fit = opt
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
                plt.plot(self.kernel_index, self.kernel_real, color='green',ls='--', lw = 2)
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
        
        self.popt_fit = popt_fit
        #a,c,tau,K,B
        if fit_kernel:
            self.popt = [self.popt_fit[0],self.popt_fit[2],1/self.popt_fit[1],self.kT/np.mean(xva_array[0]['x'].values**2),self.kT]#[ a, b, c,d ,e,f]
        else:
            self.popt = None #No fit of kernel (important for FDR correction)
       
        self.dU = dU

        #Important for Prediction Class
        if self.hs_pred:

            self.integrate=IntegrateGLE_RK4_half(kernel_half = (self.kernel_real[1:]+self.kernel_real[:-1])/2,
                                        t = self.kernel_index, dt = self.dt, dU=self.dU, m = self.m,kT=self.kT)
            
            #self.integrate=IntegrateGLE_RK4(kernel = self.kernel_real,
                                        #t = self.kernel_index, dt = self.dt, dU=self.dU, m = self.m,kT=self.kT)
        else:
            #initialization of the prediction simulation class (RK4)
            self.integrate=IntegrateGLE_RK4(kernel = self.kernel_real,
                                        t = self.kernel_index, dt = self.dt, dU=self.dU, m = self.m,kT=self.kT)
            
        return self.mem,self.kernel_index, self.kernel_real, self.kernel_data,self.ikernel, self.dU,self.popt
    
    
    #Function to extract memory kernel from trajectory by Discrete Estimation Method (see extract_kernel.py), all extracted parameters will be saved in self.mem and used for the prediction integrator
    def extractKernel_estimator(self, trj_array, time = None, plot_kernel = False,p0=0,bounds=0,end=100,verbose=False,fit_msd=False): 
        
        if not time is None:
            self.dt = time[1] - time[0]
        #print('found dt = ' + str(self.dt))
        trj_array[0] = trj_array[0][:self.cut]
 
        xva_array = self.create_xvas(trj_array, time)
        xva_array[0]=xva_array[0].reset_index()

        self.corrv = mp.correlation(xva_array[:self.cut][0]['v'].values)

        mem = mp.extract_kernel_estimator(xva_array[0][:self.cut],self.trunc,p0,bounds,end,no_fe=self.no_fe,physical=False,verbose=verbose,fit_msd=fit_msd)

        self.kernel=mem[2]
        self.ikernel=mem[3]
        self.kernel_real = mem[2]
        self.kernel_index = mem[1]
        self.popt = mem[-1]
        self.dU = mem[-3]
        self.kT = mem[-2]
        self.m = 1

        if plot_kernel:
            print('plotting extracted memory kernel...')
            plt.scatter(self.kernel_index,self.kernel, s = 5)
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
        

        #Important for Prediction Class
        if self.hs_pred:

            self.integrate=IntegrateGLE_RK4_half(kernel_half = (self.kernel_real[1:]+self.kernel_real[:-1])/2,
                                        t = self.kernel_index, dt = self.dt, dU=self.dU, m = self.m,kT=self.kT)
            
            #self.integrate=IntegrateGLE_RK4(kernel = self.kernel_real,
                                        #t = self.kernel_index, dt = self.dt, dU=self.dU, m = self.m,kT=self.kT)
        else:
            self.integrate=IntegrateGLE_RK4(kernel = self.kernel_real,
                                        t = self.kernel_index, dt = self.dt, dU=self.dU, m = self.m,kT=self.kT)
            
        return mem,self.kernel_index, self.kernel_real, self.kernel,self.ikernel, self.dU,self.popt

    #Helper function to set a kernel (e.g. smoothed from extracted data or a fit), which can be make the prediction more stable
    def set_kernel(self,t_data,kernel_data,m=None,kT=None,dU=None):
        
        self.kernel_data = kernel_data
        self.kernel = kernel_data
        self.kernel_real = kernel_data
        self.kernel_index = t_data

        if self.no_fe:
            self.dU=lambda x: 0
        else:
            try:
                self.dU(0)
            except:
                print('please provide a potential function dU(x), by using the functions extractKernel or extractKernel_estimator')
        if m is not None:
            self.m = m
        if kT is not None:

            self.kT = kT

        if self.hs_pred:

            self.integrate=IntegrateGLE_RK4_half(kernel_half = (self.kernel_real[1:]+self.kernel_real[:-1])/2,
                                        t = self.kernel_index, dt = self.dt, dU=self.dU, m = self.m,kT=self.kT)
            
            #self.integrate=IntegrateGLE_RK4(kernel = self.kernel_real,
                                        #t = self.kernel_index, dt = self.dt, dU=self.dU, m = self.m,kT=self.kT)
        else:
            self.integrate=IntegrateGLE_RK4(kernel = self.kernel_real,
                                        t = self.kernel_index, dt = self.dt, dU=self.dU, m = self.m,kT=self.kT)
            
        return self.kernel_real

    #Function to perform the GLE prediction
    def predictGLE(self, trj_array, time = None, n_steps = 1, n_preds = 1, return_full_trjs = False, zero_noise = False, Langevin  = False, alpha = 1,cond_noise = None,FDR=True,integrator='RK4',correct_fr_hist=False,params_correct=[0.1,10]):
        

        if self.hs_pred:
            self.disc=0
        else:
            self.disc = 1 #we used the forward discretization only to compute the right correlation functions (from half-stepped values), but we need the full step values of v and a for the prediction
        
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

        if not cond_noise is None:
            self.t_h = cond_noise #for cond noise generation, if cond_noise is None: we use a general generation technique (see RK4 class)
            #if self.trunc <= self.t_h:
                #self.t_h = self.trunc
            #print('use conditional random noise generator')
            #get last values of the historical random force

            if self.popt is None:
                correct_fr_hist = False
            
            if correct_fr_hist:
                #correct the velocities and acceleration to right ACF according to the fluctuation-dissipation relation with memory kernel
                #(discretization correction from Discrete Estimation method)
                #try:

                correct_dt, cut_acf = params_correct

                xvaf_corrected = xva_array[0].copy()
                corrv = mp.correlation(xvaf_corrected['v'])
                t = np.arange(0,len(corrv)*self.dt,self.dt)
                a,c,tau,K,B = self.popt

                corrv_real = mp.vacf_biexpo_delta_harm(a=a, b=0, c=c, tau1=tau, tau2=tau/10, K=K, B=B, t=t)

                try:
                    xvaf_corrected['v'][-(self.t_h+1000):] = self.reconstr_trj(xvaf_corrected['v'][-(self.t_h+1000):] ,corrv,corrv_real,cut_acf)
                except: #ensures positive definiteness of covariance matrix
                    corrv_real2 = corrv.copy()
                    corrv_real2[0] = corrv_real[0]
                    xvaf_corrected['v'][-(self.t_h+1000):] = self.reconstr_trj(xvaf_corrected['v'][-(self.t_h+1000):] ,corrv,corrv_real2,cut_acf)

                corra = mp.correlation(xvaf_corrected['a'])
                corra_real = mp.aacf_biexpo_delta_harm(a=a, b=0, c=c, tau1=tau, tau2=tau/10, K=K, B=B, t=t,DT=correct_dt)
                xvaf_corrected['a'][-(self.t_h+1000):]  = self.reconstr_trj(xvaf_corrected['a'][-(self.t_h+1000):] ,corra,corra_real,cut_acf)
                
                if self.hs_pred:
                    #t_fr, fr_all, corr_fr_all,fr_hist= self.compute_hist_fr(xvaf_corrected,self.kernel_index,self.kernel_real, self.dt,t_h = self.t_h)
                    t_fr, fr_all, corr_fr_all,fr_hist= self.compute_hist_fr_half(xvaf_corrected,self.kernel_index,(self.kernel_real[1:]+self.kernel_real[:-1])/2, self.dt,t_h = self.t_h)
                else:
                    t_fr, fr_all, corr_fr_all,fr_hist= self.compute_hist_fr(xvaf_corrected,self.kernel_index,self.kernel_real, self.dt,t_h = self.t_h)
                fr_all*=np.sqrt(correct_dt/self.dt)
                fr_hist*=np.sqrt(correct_dt/self.dt)
                corr_fr_all*=correct_dt/self.dt
                #corr_fr_all = self.kernel_real*self.kT
                #except:
                    #print('FDR correction only works with Discrete Estimation method!')
            else:
                if self.hs_pred:
                    #t_fr, fr_all, corr_fr_all,fr_hist= self.compute_hist_fr(xvaf_corrected,self.kernel_index,self.kernel_real, self.dt,t_h = self.t_h)
                    t_fr, fr_all, corr_fr_all,fr_hist= self.compute_hist_fr_half(xva_array[0],self.kernel_index,(self.kernel_real[1:]+self.kernel_real[:-1])/2, self.dt,t_h = self.t_h)
                else:
                    t_fr, fr_all, corr_fr_all,fr_hist= self.compute_hist_fr(xva_array[0],self.kernel_index,self.kernel_real, self.dt,t_h = self.t_h)
                


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
                    if FDR: #in GLE from the Mori-Formalism, the ACF of the random force should be proportional to the memory kernel
                        noise,noise_g = self.gen_noise_cond(self.kT*self.kernel_real,fr_hist,n_steps = n_steps)
                    else:
                        noise,noise_g = self.gen_noise_cond(corr_fr_all,fr_hist,n_steps = n_steps)
                    cond_noise[len(cxva):] = noise[:n_steps]
                    
                
                if self.hs_pred:

                    trj_pred, _, _ = self.integrate.integrate_half(n_steps+len(cxva), x0=x0, v0=v0,
                                        predef_v=predef_v, predef_x=predef_x,
                                        zero_noise=zero_noise, n0=len(cxva), Langevin = Langevin, alpha = alpha,custom_noise_array = cond_noise,integrator=integrator)
                        
                    #trj_pred, _, _ = self.integrate.integrate(n_steps+len(cxva), x0=x0, v0=v0,
                                        #predef_v=predef_v, predef_x=predef_x,
                                        #zero_noise=zero_noise, n0=len(cxva), Langevin = Langevin, alpha = alpha,custom_noise_array = cond_noise,integrator=integrator)
                else:
                    trj_pred, _, _ = self.integrate.integrate(n_steps+len(cxva), x0=x0, v0=v0,
                                        predef_v=predef_v, predef_x=predef_x,
                                        zero_noise=zero_noise, n0=len(cxva), Langevin = Langevin, alpha = alpha,custom_noise_array = cond_noise,integrator=integrator)
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
        fit = a * np.exp(-b * x) + d
        fit[0]+=2*c
        return fit
        #return a*np.exp(-b*x) + c*np.cos((2*np.pi)*x/d  - e) + f
        
    
    def fitted_kernel(self, index, kernel, start):
        popt, pcov = curve_fit(self.func, index[start:], kernel[start:], bounds = (-np.inf, np.inf), maxfev=10000,p0 = self.p0) 

        fitted_kernel = np.append(kernel[:start], self.func(index[start:], *popt))

        return popt, pcov, fitted_kernel

    def RMSE(self, pred, real):
        return np.mean((pred - real)**2)**0.5   
 
    #Function to compute random force from a given trajectory, uncoupled from RK4, because we need xvaf
    def compute_hist_fr(self,xvaf,t, kernel, dt,t_h = 100): 
    #Calculates random noise from given trajectory and extracted Kernel

        N = len(kernel)

        xvaf = xvaf[-(t_h+100):]
        M = len(xvaf)
        if N < M:
            kernel = np.append(kernel,np.zeros(int(M - N)))
        x = np.array(xvaf["x"])
        v = np.array(xvaf["v"])
        a = np.array(xvaf["a"])

        m = self.m
        #self.kT = m*self.corrv[0]*self.alpha
        
        fr = np.zeros(np.min((len(a), len(kernel))))
        fr[0] = m*a[0] + self.dU(x[0]) + 0.5*dt*kernel[0]*v[0]
        for i in range(1,len(fr)):
            
                fr[i] =  m*a[i]  + 0.5*dt*kernel[0]*v[i] + 0.5*dt*kernel[i]*v[0] + dt*np.sum(kernel[1:i]*v[1:i][::-1])+ self.dU(x[i])

        fr_hist=fr[-t_h:]
        t_fr = np.arange(0,len(fr_hist)*dt,dt)
        corr_fr = mp.correlation(fr)[:self.trunc]
        return t_fr, fr, corr_fr,fr_hist #corr_fr will be used as covariance function for random force and fr_hist as last known values
    
    #using half_stepped velocities
    def compute_hist_fr_half(self,xvaf,t, kernel_half, dt,t_h = 100): 
    #Calculates random noise from given trajectory and extracted Kernel

        N = len(kernel_half)

        xvaf = xvaf[-(t_h+100):]
        M = len(xvaf)
        if N < M:
            kernel_half = np.append(kernel_half,np.zeros(int(M - N)))
        x = np.array(xvaf["x"])
        v = np.array(xvaf["v"])
        a = np.array(xvaf["a"])

        m = self.m
        #self.kT = m*self.corrv[0]*self.alpha
        
        fr = np.zeros(np.min((len(a), len(kernel_half))))
        fr[0] = m*a[0] + self.dU(x[0]) + kernel_half[0]*v[0]*dt
        for i in range(1,len(fr)):
            
                fr[i] = m*a[i]  + dt*np.sum(kernel_half[:i+1][:N]*v[:i+1][::-1][:N])+ self.dU(x[i])
                #fr[i] =  m*a[i]  + 0.5*dt*kernel[0]*v[i] + 0.5*dt*kernel[i]*v[0] + dt*np.sum(kernel[1:i]*v[1:i][::-1])+ self.dU(x[i])

        fr_hist=fr[-t_h:]
        t_fr = np.arange(0,len(fr_hist)*dt,dt)
        corr_fr = mp.correlation(fr)[:self.trunc]
        return t_fr, fr, corr_fr,fr_hist #corr_fr will be used as covariance function for random force and fr_hist as last known values
    
    #Colored noise generator conditioned on historical noise
    def gen_noise_cond(self,fr_corr,fr_hist,n_steps):
        
        fr_corr = np.append(fr_corr[:self.t_h],np.zeros(int(np.abs(len(fr_corr)-self.t_h))))  #circumvent positive-semidefinite problem       
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
    
    #Helper function for FDR correction
    def gen_noise_correct(self,fr_corr,n_steps = 100):
    
        if n_steps > int(len(fr_corr)): #we use fr_hist with length trunc, which results in 0 n_steps for noise gen
                fr_corr= np.append(fr_corr, np.zeros(n_steps))
                
        if n_steps < int(len(fr_corr)):
                fr_corr = fr_corr[:int(n_steps)]
                
        N = len(fr_corr)  
                
        # Compute the covariance matrix from the Memory Kernel
        
        C=(np.triu(np.array([np.roll(fr_corr, i) for i in range(0,len(fr_corr))]))+\
        np.triu(np.array([np.roll(fr_corr, i) for i in range(0,len(fr_corr))]),1).T)

        # Compute Cholesky decomposition of conditional covariance matrix
        L=np.linalg.cholesky(C).real
        
        # Generate Gaussian process
        samp=np.random.normal(0,np.ones((N,)))
        GP=np.dot(L, samp)
        
        return GP

    #Helper function for FDR correction
    def reconstr_trj(self,x,acf,acf_new,cut_acf=10):

        n_steps = len(x)
        acf_res = acf_new-acf

        ns=int(1e4) #RAM limit!
        if ns < n_steps:
            
            M2 = int(n_steps/ns+1)
            GP = np.array([])
            for l in range(M2):

                GP2 = self.gen_noise_correct(acf_res[:cut_acf],n_steps=int(ns))
                GP = np.append(GP,GP2[-ns:])

            GP = GP.flatten()[:n_steps]
        
        else:
            
            GP= self.gen_noise_correct(acf_res[:cut_acf],n_steps=ns)
                
        x = x + GP[:n_steps] 
        
        return x
    
    #Interpolation helper function for mean force
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

#Class for GLE Integration with Runge-Kutta 4
class IntegrateGLE_RK4: 
    def __init__(self, kernel, t, dt, m=1, dU = lambda x: 0.,kT=2.494):
        self.kernel = kernel #array, extracted Kernel
        self.t = t #array, time array of Kernel
        self.m = m #float, mass
        #self.dt = self.t[1] - self.t[0]
        self.dt = dt #float, time-step for Prediction (usually same as memory kernel extraction)
        self.dU = dU #object, extracted free energy function for prediction
        self.kT=kT #float, temperature
        
    def integrate(self, n_steps, x0 = 0., v0 = 0., zero_noise = False, predef_x = None, predef_v = None,n0 = 0, custom_noise_array = None, Langevin = False, alpha = 1,integrator='RK4'):
        
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
            #self.kT=2.494
        else: 
            assert (len(predef_v) == n_steps)
            assert (predef_v[n0 - 1] == v)
            self.v_trj = predef_v
            #self.kT = np.mean(predef_v**2)/self.m
        
        self.t_trj = np.arange(0., n_steps * self.dt, self.dt)

        if zero_noise:
            noise = np.zeros(n_steps)
            
        else:
            
            if custom_noise_array is None:
                if Langevin == False:
                    #Important because we append the known trajectory before the Prediction
                    noise_array = np.zeros(n0 + 2)
                
                    #Generating noise (which shouldn't be used, as it does not know the last known random force)
                    noise = self.gen_noise(self.kernel, self.t, self.dt, n_steps = n_steps - n0)
                
                    noise_array = np.append(noise_array, noise)
                    #noise_array = np.append(noise_array, np.zeros(n_steps))
                    noise = noise_array
                
                else: #Optional: To run a Langevin-Prediction with no memory and white noise
                    gamma = self.kernel[0]
                    self.kernel = np.zeros(n_steps-n0)
                    self.kernel[0] = gamma
            
                    #sigma = self.alpha*math.sqrt(self.kernel[0]*2.494) #with kT
                    sigma = self.alpha*math.sqrt(self.kernel[0]*self.kT) #with kT
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
                #rmi = self.mem_red_integrand(self.v_trj[:i-1])
                v_old = self.v_trj[i-1]
     
            else:
                rmi = 0
                rmi_old = 0
                v_old = 0

            if integrator == 'RK4':

                x, v = self.step_rk4(x,v,rmi,noise[i], v_old, rmi_old)
                self.x_trj[i] = x
                self.v_trj[i] = v
            
            else: #use Euler integrator (better for discrete data with delta-kernel properties)
                aa =  (noise[i] - 0.5*self.dt*self.kernel[0]*self.v_trj[i-1] - rmi - self.dU(self.x_trj[i-1]))/self.m
                self.v_trj[i]=self.v_trj[i-1] +self.dt*aa
                self.x_trj[i]=self.x_trj[i-1] +self.dt*self.v_trj[i]
            
            
        return self.x_trj, self.v_trj, self.t_trj

    #Function to run RK4-Simulation
    def mem_red_integrand(self, v):
        if len(v) < len(self.kernel):
            v = np.concatenate([np.zeros(len(self.kernel) - len(v) + 1) ,v ])
        integrand = v[:len(v) - len(self.kernel[1:]) - 1:-1] * self.kernel[1:]
        return (0.5 * integrand[-1] + np.sum(integrand[:-1])) * self.dt
    
    def f_rk4(self, x, v, rmi, noise, next_w, last_w, v_old, rmi_old):
        nv = v
        na = (-next_w * rmi - last_w * rmi_old - 0.5 * next_w * self.kernel[0]
              * v * self.dt - 0.5 * last_w * self.kernel[0] * v_old * self.dt
              - self.dU(x) + noise) / self.m
            
        return nv, na
        
    def step_rk4(self, x, v, rmi, noise, v_old, rmi_old):
        k1x, k1v = self.f_rk4(x, v, rmi, noise, 0.0, 1.0, v_old, rmi_old)
        k2x, k2v = self.f_rk4(x + k1x * self.dt / 2, v + k1v * self.dt / 2, rmi,
                             noise, 0.5, 0.5, v_old, rmi_old)
        k3x, k3v = self.f_rk4(x + k2x * self.dt / 2, v + k2v * self.dt / 2, rmi,
                             noise, 0.5, 0.5, v_old, rmi_old)
        k4x, k4v = self.f_rk4(x + k3x * self.dt, v + k3v * self.dt, rmi, noise,
                             1.0, 0.0, v_old, rmi_old)
        return x + self.dt * (
            k1x + 2. * k2x + 2. * k3x + k4x) / 6., v + self.dt * (
                k1v + 2. * k2v + 2. * k3v + k4v) / 6.  
    
#----Half Stepped Integrator (very unstable for delta-kernels)-----
class IntegrateGLE_RK4_half: #Class for GLE Integration with Runge-Kutta 4 (half_stepped velocities)
    def __init__(self, kernel_half, t, dt, m=1, dU = lambda x: 0.,kT=2.494):
        self.kernel_half = kernel_half #extracted Kernel
        self.t = t #time array of Kernel
        self.m = m #mass
        #self.dt = self.t[1] - self.t[0]
        self.dt = dt #time-step for Prediction (usually same as memory kernel extraction)
        self.dU = dU #extracted free Energy for Prediction
        self.kT=kT
        
    def integrate_half(self, n_steps, x0 = 0., v0 = 0., zero_noise = False, predef_x = None, predef_v = None,n0 = 0, custom_noise_array = None, Langevin = False, alpha = 1,integrator='RK4'):
        
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

        if zero_noise:
            noise = np.zeros(n_steps)
            
        else:
            
            if custom_noise_array is None:
                if Langevin == False:
                    #Important because we append the known trajectory before the Prediction
                    noise_array = np.zeros(n0 + 2)
                
                    #Generating noise (which shouldn't be used)
                
                    #noise = self.gen_noise(self.kernel_half, self.t, self.dt, n_steps = n_steps - n0)
                    noise = np.zeros(n_steps-n0)

                    noise_array = np.append(noise_array, noise)
                    #noise_array = np.append(noise_array, np.zeros(n_steps))
                    noise = noise_array
                
                else: #Optional: To Run a Langevin-Prediction with no memory and white noise
                    gamma = self.kernel[0]
                    self.kernel = np.zeros(n_steps-n0)
                    self.kernel[0] = gamma
            
                    sigma = self.alpha*math.sqrt(2*self.kernel_half[0]*self.kT) #with kT
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
                rmi = self.mem_red_integrand_half(self.v_trj[:i])
                #rmi = self.mem_red_integrand(self.v_trj[:i-1])
                v_old = self.v_trj[i-1]
     
            else:
                rmi = 0
                rmi_old = 0
                v_old = 0

            if integrator == 'RK4':

                x, v = self.step_rk4_half(x,v,rmi,noise[i], rmi_old)
                self.x_trj[i] = x
                self.v_trj[i] = v
            
            else: #use Euler integrator (better for discrete data with delta-kernel properties)
                aa =  (noise[i] - rmi - self.dU(self.x_trj[i-1]))/self.m
                self.v_trj[i]=self.v_trj[i-1] +self.dt*aa
                self.x_trj[i]=self.x_trj[i-1] +self.dt*self.v_trj[i]
            
            
        return self.x_trj, self.v_trj, self.t_trj

    #Function to run RK4-Simulation
    def mem_red_integrand_half(self, v):
        if len(v) < len(self.kernel_half):
            #v = np.concatenate([np.zeros(len(self.kernel_half) - len(v) + 1) ,v ])
            v = np.append(np.zeros(len(self.kernel_half)-len(v)),v)

        N = len(self.kernel_half)
        return self.dt*np.sum(self.kernel_half[:N]*v[::-1][:N])
        
    
    def f_rk4_half(self, x, v, rmi, noise, next_w, last_w, rmi_old):
        nv = v
        na = (-next_w * rmi - last_w * rmi_old - self.dU(x) + noise) / self.m
            
        return nv, na
        
    def step_rk4_half(self, x, v, rmi, noise, rmi_old):
        k1x, k1v = self.f_rk4_half(x, v, rmi, noise, 0.0, 1.0, rmi_old)
        k2x, k2v = self.f_rk4_half(x + k1x * self.dt / 2, v + k1v * self.dt / 2, rmi,
                             noise, 0.5, 0.5, rmi_old)
        k3x, k3v = self.f_rk4_half(x + k2x * self.dt / 2, v + k2v * self.dt / 2, rmi,
                             noise, 0.5, 0.5, rmi_old)
        k4x, k4v = self.f_rk4_half(x + k3x * self.dt, v + k3v * self.dt, rmi, noise,
                             1.0, 0.0, rmi_old)
        return x + self.dt * (
            k1x + 2. * k2x + 2. * k3x + k4x) / 6., v + self.dt * (
                k1v + 2. * k2v + 2. * k3v + k4v) / 6.  





    #-----#####old#####-----

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
