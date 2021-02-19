import numpy as np
import pandas as pd

import math
import matplotlib.pyplot as plt
from mempred import *

import mempred as mp

from scipy.optimize import curve_fit

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
    def __init__(self, bins = "auto", cut = 1000, trunc = 80, dt = 1, last_value_correction = False, no_fe = False, plot_pred = False):
        self.trunc = trunc #depth of memory Kernel
        self.bins = bins #number of bins for the histogram (potential-extraction)
        self.dt = dt #time-step for Extraction and GLE Simulation (usually dt = 1 is used)
        self.no_fe = no_fe #exclude free energy term in GLE
        self.cut = cut #cut : length of historical trajectory for memory extraction
        self.last_value_correction = last_value_correction #Correction of Prediction-Trajectory (last known step)
        self.plot_pred = plot_pred #Plots the Prediction automatically
        
    def create_xvas(self, trj_array, time_arg=None): #Creating x-v-a-Dataframe for given x-trajectory (needed for memory extraction)
        xva_array=[]
        for trj in trj_array:
            if time_arg is None:
                time=np.arange(0,len(trj)*self.dt,self.dt)
            else:
                time=time_arg
            xf=mp.xframe(trj, time, fix_time=False)
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
    
    def extractKernel(self, trj_array, time = None, fit_kernel = False, plot_kernel = False,fit_start = 0, fit_end = 3,kT = 2.494): 
        #Memory Kernel Extraction (with memtools)
        
        if not time is None:
            self.dt = time[1] - time[0]
        #print('found dt = ' + str(self.dt))
        trj_array[0] = trj_array[0][:self.cut]
        xva_array = self.create_xvas(trj_array, time)
        self.kernels = []
        
        mem = mp.Igle(xva_array[:self.cut], kT = kT, trunc = self.trunc,verbose = False)
        
        mem.compute_corrs()
        corrv = np.loadtxt('corrs.txt', usecols =1)
        self.m = kT/corrv[0]
        
        if self.no_fe:
            mem.set_harmonic_u_corr(0.)
            dU=lambda x: 0
        else:
            mem.compute_fe(bins=self.bins)
            mem.compute_u_corr()
            dU_memtools=mem.dU
            
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
                    value = 0 #free moving particle
                else:
                    value = dU_memtools(x)
                return value
        
        self.kernel=mem.compute_kernel()
      
            
        self.kernel_real = self.kernel["k"].values
        self.kernel_index = np.arange(0,len(self.kernel_real)*self.dt,self.dt)
        popt = np.zeros(3)
        pcov = np.zeros(3)
        
        if fit_kernel: #Sometimes it is better to fit the kernel as an exponential decay
            
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
            
        if plot_kernel:
            print('plotting extracted memory kernel...')
            plt.scatter(self.kernel_index,self.kernel["k"], s = 5)
            if fit_kernel:
                plt.plot(self.kernel_index, self.kernel_real, 'g--', lw = 2)
            plt.xlabel("t", fontsize = 'x-large')
            plt.ylabel("$\\Gamma(t)$", fontsize = 'x-large')
            plt.title('extracted memory kernel of trajectory')
            plt.tick_params(labelsize="x-large")
            plt.show()
            plt.close()
            
            print('plotting running integral of kernel...')
            plt.scatter(self.kernel_index,self.kernel["ik"], s = 5)
            
            plt.xlabel("t", fontsize = 'x-large')
            plt.ylabel("G(t)", fontsize = 'x-large')
            plt.title('running integral of kernel')
            plt.tick_params(labelsize="x-large")
            plt.show()
            plt.close()
        #dt = self.kernel.index[1] - self.kernel.index[0]
        
        #Important for Prediction Class
        
        self.integrate=IntegrateGLE_RK4(kernel = self.kernel_real,
                          
          t = self.kernel_index, dt = self.dt, dU=dU, m = self.m)
        
        self.dU = dU
        return self.kernel_index, self.kernel_real, self.kernel["ik"].values, dU, popt
    
    def predictGLE(self, trj_array, time = None, n_steps = 1, n_preds = 1, return_full_trjs = False, zero_noise = False, Langevin  = False, alpha = 1,cond_noise = None):
        actual = np.array(trj_array)
       
        self.alpha = alpha
        
        trj_array[0] = trj_array[0][:self.cut]
        
        xva_array = self.create_xvas(trj_array, time)
        zero_row = np.zeros((2,len(xva_array[0].columns)))
        
        top_row = pd.DataFrame(zero_row,columns=xva_array[0].columns)
        xva_array[0] = pd.concat([top_row, xva_array[0]]).reset_index(drop = True)
        xva_array[0]['x'] = xva_array[0]['x'].shift(-1) 
        
        xva_array[0]['x'].values[-1] = trj_array[0][-1]
        trj_pred_array = []
        #trj_pred = np.zeros(len(xva_array)+n_steps)
        
        if not cond_noise is None:
            self.t_h = cond_noise #for cond noise generation, if cond_noise is None: we use a general generation technique (see RK4 class)
            #print('use conditional random noise generator')
            #get last values of the historical random force
            
            t_fr, fr_hist= self.compute_hist_fr(xva_array[0], self.kernel_index,self.kernel_real, self.dt,t_h = self.t_h)
           
                    
            
        for i in range(0,n_preds):
            for cxva, trj in zip(xva_array, trj_array): 
                
                #print(cxva)
                x0 = cxva.iloc[-1]["x"] #initial values are the last known values
                v0 = cxva.iloc[-1]["v"]
                #print(x0,v0)
                #print(len(cxva))
                cr = np.zeros(len(cxva)+n_steps)
                predef_x = np.zeros(len(cxva)+n_steps)
                predef_x[:len(cxva)] = cxva["x"]
           
                predef_v = np.zeros(len(cxva)+n_steps)
                predef_v[:len(cxva)] = cxva["v"]
                
                if not cond_noise is None:
                    cond_noise = np.zeros(len(predef_v))
                    #generate future steps conditional Gaussian process, starting at last known step of historical noise
                    noise,noise_g = self.gen_noise_cond(self.kernel_index,self.kernel_real,fr_hist,n_steps = n_steps)
                    cond_noise[len(cxva):] = noise[:n_steps]
                    
                
                trj_pred, _, _ = self.integrate.integrate(n_steps+len(cxva), x0=x0, v0=v0,
                                     predef_v=predef_v, predef_x=predef_x,
                                     zero_noise=zero_noise, n0=len(cxva), Langevin = Langevin, alpha = alpha,custom_noise_array = cond_noise)
                if self.last_value_correction:
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
            fr_trj[self.cut-self.t_h:] = noise_g
        
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
            #plt.savefig("run_figures/pred_all.png", bbox_inches='tight')
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
            #plt.savefig("run_figures/pred_future.png", bbox_inches='tight')
            plt.show()
            plt.close()
            #np.savetxt('run_figures/pred.txt', np.vstack((index_pred, trj_p_mean, error)).T, delimiter = ',')
             
        return index_pred, trj_p_mean, trj_p_mean[self.cut:], error, actual[0], actual_plot, fr_trj
    
    def func(self, x, a, b):
            #return a * np.exp(-b * x) + c
            return a * np.exp(-b * x)

    def fitted_kernel(self, index, kernel, start):
        popt, pcov = curve_fit(self.func, index[start:], kernel[start:], bounds = (-np.inf, np.inf), maxfev=1000000) 

        fitted_kernel = np.append(kernel[:start], self.func(index[start:], *popt))

        return popt, pcov, fitted_kernel

    def RMSE(self, pred, real):
        return np.mean((pred - real)**2)**0.5   
 
    #uncoupled from RK4, because we need xvaf
    def compute_hist_fr(self,xvaf,t, kernel, dt,t_h = 100): 
    #Calculates random noise from given trajectory and extracted Kernel
    #trunc is now flipped, so the last trunc-values of x-array

        N = len(kernel)
        x = np.array(xvaf["x"])
        v = np.array(xvaf["v"])
        a = np.array(xvaf["a"])
        
       
        corrv = np.loadtxt('corrs.txt', usecols =1)
       
        m = self.m
        self.kT = m*corrv[0]*self.alpha

        #print("Compute Random Force...")

        tmax=int(t_h/dt)
        
        x = np.flip(np.array(xvaf["x"]))
        v = np.flip(np.array(xvaf["v"]))
        a = np.flip(np.array(xvaf["a"]))
        
        x = x[:tmax]
        x = np.flip(x)
        v = v[:tmax]
        v = np.flip(v)
        a = a[:tmax]
        a = np.flip(a)

        prefac = 1./corrv[0]



        fr = np.zeros(np.min((len(a), len(kernel))))
        fr[0] = m*a[0] + self.dU(x[0])
        for i in range(1,len(fr)):
            
            fr[i] =  m*a[i] + 0.5*dt*kernel[0]*v[i] + 0.5*dt*kernel[i]*v[0] + dt*np.sum(kernel[1:i+1]*v[:i][::-1])+ self.dU(x[i])

        t_fr = np.arange(0,len(fr)*dt,dt)    

        return t_fr, fr
    
    def gen_noise_cond(self,t,kernel,fr_hist,n_steps):
        
        kernel = np.append(kernel[:self.t_h],np.zeros(int(np.abs(len(kernel)-self.t_h))))           
        if n_steps > int(len(kernel)-self.t_h): #we use fr_hist with length trunc, which results in 0 n_steps for noise gen
            #assert (kernel[-1] == 0), "The memory kernel has not decayed to zero after trunc! Choose a larger memory kernel depth!"
            kernel = np.append(kernel, np.zeros(n_steps)) #after trunc, the kernel has to be decayed to zero!!
            t = np.arange(0,len(kernel))
        if n_steps < int(len(kernel)-self.t_h):
            kernel = kernel[:int(n_steps+self.t_h)]
            t = np.arange(0,len(kernel))         
        N = len(kernel)

        C=self.kT*self.alpha*(np.triu(np.array([np.roll(kernel, i) for i in range(0,len(kernel))]))+\
        np.triu(np.array([np.roll(kernel, i) for i in range(0,len(kernel))]),1).T)

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


class IntegrateGLE_RK4: #Class for GLE Integration with Runge-Kutta 4
    def __init__(self, kernel, t, dt, m=1, dU = lambda x: 0., add_zeros = 0):
        self.kernel = kernel #extracted Kernel
        self.t = t #time array of Kernel
        self.m = m #mass
        #self.dt = self.t[1] - self.t[0]
        self.dt = dt #time-step for Prediction (usually same as memory kernel extraction)
        self.dU = dU #extracted free Energy for Prediction
           
    def integrate(self, n_steps, x0 = 0., v0 = 0., zero_noise = False, predef_x = None, predef_v = None, n0 = 0, custom_noise_array = None, Langevin = False, alpha = 1):
        
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
                
                    #Generating noise
                
                    noise = self.gen_noise(self.kernel, self.t, self.dt, n_steps = n_steps - n0)
                
                    noise_array = np.append(noise_array, noise)
                    #noise_array = np.append(noise_array, np.zeros(n_steps))
                    noise = noise_array
                
                else: #Optional: To Run a Langevin-Prediction with no memory and white noise
            
                    gamma = self.kernel[0]
                    self.kernel = np.zeros(n_steps-n0)
                    self.kernel[0] = gamma
            
                    sigma = math.sqrt(self.kernel[0]*2.494) #with kT
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
                v_old = self.v_trj[i-1]
     
            else:
                rmi = 0
                rmi_old = 0
                v_old = 0
                
            x, v = self.step_rk4(x,v,rmi,noise[i], v_old, rmi_old)
            self.x_trj[i] = x
            self.v_trj[i] = v
            
            
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
    
    #Function to construct random colored noise (see report for derivation)
    def gen_noise(self,kernel, t, dt, n_steps):
    
        if n_steps > int(len(kernel)/2): #because we cut the kernel and after FT we divide the length of the Kernel by 2
            kernel = np.append(kernel, np.zeros(n_steps*2))
            t = np.arange(0,len(kernel))
            
        N = len(kernel)
        
        corrv = np.loadtxt('corrs.txt', skiprows = 1, usecols = 1)
        Gamma_arr = np.array(kernel)
        #Gamma_arr2=np.concatenate((np.flip(Gamma_arr[1:]),Gamma_arr[:-1]))

        #dt = t[1] - t[0]
        #kT = 2.494
        m = 1 #please change if you want to predict physical trajectories!!
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

