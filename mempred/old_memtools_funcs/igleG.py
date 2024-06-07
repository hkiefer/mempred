from __future__ import print_function
from .correlation import *
from .igleplot import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

class IgleG(IglePlot):
    def __init__(self, *args, **kwargs):
        """
        Create an instance of the IgleG class, child of IglePlot class.
	This class implements the G method, which computes the integral G of
	the memory kernel directly from correlation functions. The kernel
	is then computed from the derivative of the function G.
        ----------
        """
        super(IgleG, self).__init__(*args, **kwargs)
        #self.plot=True
        self.plot = False
        self.k=None
        
    def compute_corrs(self, *args, **kwargs):
        if self.verbose:
            print("Calculate xx and vv correlation functions...")

        self.corrs=None
        for weight,xva in zip(self.weights,self.xva_list):
            if self.k is not None:
                xxcorrw=weight*pdcorr(xva,"x","x",self.trunc,"xx")
            vvcorrw=weight*pdcorr(xva,"v","v",self.trunc,"vv")

            if self.corrs is None:
                if self.k is not None:
                    self.corrs=pd.concat([xxcorrw,vvcorrw],axis=1)
                else:
                    self.corrs=vvcorrw
            else:
                if self.k is not None:
                    self.corrs["xx"]+=xxcorrw["xx"]
                self.corrs["vv"]+=vvcorrw["vv"]

        self.corrs/=self.weightsum
        #print(self.corrs)
        if self.saveall:
            self.corrs.to_csv(self.prefix+self.corrsfile,sep=" ")

    def compute_u_corr(self):
        if self.fe_spline is None:
            raise Exception("Free energy has not been computed.")
        if self.verbose:
            print("Calculate a/v grad(U(x)) correlation function...")

        # get target length from first element and trunc
        ncorr=self.xva_list[0][self.xva_list[0].index < self.trunc].shape[0]


        self.ucorr=pd.DataFrame({"xu":np.zeros(ncorr)}, \
        index=self.xva_list[0][self.xva_list[0].index < self.trunc].index\
              -self.xva_list[0].index[0])

        for weight,xva in zip(self.weights,self.xva_list):
            x=xva["x"].values
            corr=correlation(x,self.dU(x),subtract_mean=False)
            self.ucorr["xu"]+=weight*corr[:ncorr]

        self.ucorr/=self.weightsum

        if self.saveall:
            self.ucorr.to_csv(self.prefix+self.ucorrfile,sep=" ")

    def compute_mass(self):
        super(IgleG, self).compute_mass()

    def compute_kernel_harmonic(self, *args, **kwargs):
        print("Computing harmonic U")
        if self.mass is None:
            raise Exception("Mass has not been computed.")
        m=self.mass
        dt=self.dt
        k=self.k
        tmax=int(self.trunc/self.dt)

        v_acf = self.corrs['vv'].values[:tmax]
        x_acf = self.corrs['xx'].values[:tmax]
        kernel = np.zeros(min(len(v_acf),tmax))
        kernel_i = np.zeros(min(len(v_acf),tmax))

        prefac = 1/v_acf[0]

        for i in range(1,len(kernel)):
            kernel_i[i] = prefac*((x_acf[i]/x_acf[0]*v_acf[0] - v_acf[i])/dt*m - np.sum(kernel_i[:i]*v_acf[1:i+1][::-1]))*2

        self.kernel=pd.DataFrame({"k":np.gradient(kernel_i,dt),"ik":kernel_i},index=self.corrs.index.values[:tmax])
        self.kernel=self.kernel[["k","ik"]]
        if self.saveall:
            self.kernel.to_csv(self.prefix+self.kernelfile,sep=" ")

    def compute_kernel(self, second_kind=False, *args, **kwargs):
        if self.k is not None:

            self.compute_kernel_harmonic( *args, **kwargs)
            return
        
        dt=self.corrs.index[1]-self.corrs.index[0]
        #dt=self.dt
        tmax=int(self.trunc/dt)

        v_acf = self.corrs['vv'].values[:tmax]
        xu_cf = self.ucorr['xu'].values[:tmax]
        kernel = np.zeros(min(len(v_acf),tmax))
        kernel_i = np.zeros(min(len(v_acf),tmax))

        prefac = 1./v_acf[0]
        m=self.kT/v_acf[0]
        for i in range(1,len(kernel)):
            if second_kind==False:
                kernel_i[i] = prefac*((xu_cf[i]-xu_cf[0]+m*v_acf[0] - v_acf[i]*m )/dt - np.sum(kernel_i[:i]*v_acf[1:i+1][::-1]))*2
            else:
                kernel_i[i] = prefac*((xu_cf[i]- v_acf[i]*xu_cf[0]/v_acf[0])/dt - np.sum(kernel_i[:i]*v_acf[1:i+1][::-1]))*2
        
        for i in range(0,len(kernel)-1):
            kernel[i] = (kernel_i[i+1] - kernel_i[i])/dt
            
        #self.kernel=pd.DataFrame({"k":np.gradient(kernel_i,dt),"ik":kernel_i},index=self.corrs.index.values[:tmax])
        self.kernel=pd.DataFrame({"k":kernel,"ik":kernel_i},index=self.corrs.index.values[:tmax])
        self.kernel=self.kernel[["k","ik"]]
        if self.saveall:
            self.kernel.to_csv(self.prefix+self.kernelfile,sep=" ")

        return self.kernel

    def plot_corrs(self):
        plt.figure()
        plt.plot(self.corrs.index,self.corrs["xx"])
        plt.xscale("log")
        plt.xlabel("$t$")
        plt.ylabel("$\\langle xx\\rangle$")
        plt.title("Position autocorrelation function")
        plt.show(block=False)

        plt.figure()
        plt.plot(self.corrs.index,self.corrs["vv"])
        plt.xscale("log")
        plt.xlabel("$t$")
        plt.ylabel("$\\langle vv\\rangle$")
        plt.title("Velocity autocorrelation function")
        plt.show(block=False)



    def plot_kernel(self):
        super(IgleG, self).plot_kernel()
