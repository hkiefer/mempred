from .igleplot import *
from .conditionalCorrelation import *
import numpy as np


class IgleConditional(IglePlot):

    def __init__(self, xva_arg, conditionBinMidPoints, constMass = True, conditionVarName="x", **kwargs):
        """
        Create an instance of the IgleConditional class, child of IglePlot class. 
        This class allows computation of conditional correlation functions and memory kernels as
        an approximation for space dependent memory kernels.
        Additional parameters
        ----------
        conditionBinMidPoints : numpy array (required!)
		Numpy array of the midpoints for the binning along the conditional variable.
	conditionVarName : str, default="x"
		Name of the conditional variable. Needs to be one given in the input dataframe.
        constMass : bool, default=True
		Whether to assume a constant mass over the whole ensemble. 
		If false a conditional mass is assumed.
	"""
        super(IgleConditional, self).__init__(xva_arg, **kwargs)

        #self.igleList=[Igle(xva_arg,verbose=False,**kwargs)]
        #for i in range(conditionBinMidPoints.shape[0]):
        #    self.igleList.append(Igle(xva_arg,verbose=False,**kwargs))

        self.conditionVarName = conditionVarName
        self.conditionBinMidPoints = conditionBinMidPoints

        diffs = np.array(conditionBinMidPoints[1:]-conditionBinMidPoints[:-1])
        self.conditionBins = np.concatenate([conditionBinMidPoints[:-1]+diffs/2])

        self.constMass = constMass

    def compute_mass(self, mass = None):
        if self.verbose:
            print("Calculate masses...")
            print("Use kT:", self.kT)
            print("Assuming constant Mass: ",self.constMass)

        if self.corrs is None:
            raise Exception("Need correlation functions to compute masses.")
            #v2sum=0.
            #for i,xva in enumerate(self.xva_list):
            #    v2sum+=(xva["v"]**2).mean()*self.weights[i]
            #v2=v2sum/self.weightsum
            #self.mass=self.kT/v2
        else:
            self.mass=self.kT/self.corrs["vv"].iloc[0]


        if self.constMass:
            self.mass.iloc[:] = np.ones(self.mass.shape)* np.mean(self.mass.values)

        if mass is not None:
            self.constMass = True
            self.mass.iloc[:] = np.ones(self.mass.shape)* mass

        if self.verbose:
            print("Found masses:", self.mass)


    def compute_u_corr(self):
        if self.fe_spline is None:
            raise Exception("Free energy has not been computed.")
        if self.verbose:
            print("Calculate a/v grad(U(x)) correlation function...")

        # get target length from first element and trunc
        ncorr=self.xva_list[0][self.xva_list[0].index < self.trunc].shape[0]


        aucorr=pd.DataFrame(np.zeros((ncorr,self.conditionBinMidPoints.shape[0])), columns=self.conditionBinMidPoints, \
        index=self.xva_list[0][self.xva_list[0].index < self.trunc].index\
              -self.xva_list[0].index[0])
        self.ucorr=pd.concat([aucorr],keys= ['au'],axis=1)
        if self.first_order or self.hybrid:
            vucorr=pd.DataFrame(np.zeros((ncorr,self.conditionBinMidPoints.shape[0])), columns=self.conditionBinMidPoints, \
        index=self.xva_list[0][self.xva_list[0].index < self.trunc].index\
              -self.xva_list[0].index[0])
            self.ucorr=pd.concat([aucorr,vucorr],keys= ['au','vu'],axis=1)

        for weight,xva in zip(self.weights,self.xva_list):
            x=xva["x"].values
            a=xva["a"].values
            corr=condCorrelation(a,self.conditionBinMidPoints,b=self.dU(x),c=xva[self.conditionVarName].values,tLen=ncorr,subtract_mean=False)
            self.ucorr["au"]+=weight*corr.T

            if self.first_order or self.hybrid:
                v=xva["v"].values
                corr=condCorrelation(v,self.conditionBinMidPoints,b=self.dU(x),c=xva[self.conditionVarName].values,tLen=ncorr,subtract_mean=False)
                self.ucorr["vu"]+=weight*corr.T

        self.ucorr/=self.weightsum

        if self.saveall:
            self.ucorr.to_csv(self.prefix+self.ucorrfile,sep=" ")

    def compute_corrs(self):
        if self.verbose:
            print("Calculate vv, va and aa correlation functions...")

        self.corrs=None
        for weight,xva in zip(self.weights,self.xva_list):
            vvcorrw=weight*pdCondCorr(xva,"v","v",self.conditionBinMidPoints,self.conditionVarName,self.trunc)
            vacorrw=weight*pdCondCorr(xva,"v","a",self.conditionBinMidPoints,self.conditionVarName,self.trunc)
            aacorrw=weight*pdCondCorr(xva,"a","a",self.conditionBinMidPoints,self.conditionVarName,self.trunc)
            if self.corrs is None:
                self.corrs = pd.concat([vvcorrw,vacorrw,aacorrw],keys= ['vv', 'va', 'aa'],axis=1)
            else:
                self.corrs["vv"]+=vvcorrw
                self.corrs["va"]+=vacorrw
                self.corrs["aa"]+=aacorrw
        self.corrs/=self.weightsum
        if self.saveall:
            self.corrs.to_csv(self.prefix+self.corrsfile,sep=" ")

    def compute_kernel(self, first_order=None, hybrid=None, k0=0.):
        """
Computes the memory kernel. If you give a nonzero value for k0, this is used at time zero, if set to 0, the C-routine will calculate k0 from the second order memory equation.
        """
        if first_order is None:
            first_order=self.first_order
        if first_order and not self.first_order:
            raise Exception("Please initialize in first order mode, which allows both first and second order.")
        if hybrid is None:
            hybrid=self.hybrid
        if hybrid and not self.first_order and not self.hybrid:
            raise Excpetion("Please initialize in hybrid mode, which allows first, second order and hybrid.")
        if first_order and hybrid:
          raise Exception("First_order and hybrid computations are mutually exclusive! Please decide for one!")
        if self.corrs is None or self.ucorr is None:
            raise Exception("Need correlation functions to compute the kernel.")
        if self.mass is None:
            if self.verbose:
                print("Mass not calculated.")
            self.compute_mass()

        dt=self.corrs.index[1]-self.corrs.index[0]
        if self.verbose:
            print("Use dt:",dt)

        kernels=[]
        for cond in self.conditionBinMidPoints:

            v_acf=self.corrs["vv"][cond].values
            va_cf=self.corrs["va"][cond].values


            if first_order or hybrid:
                vu_cf=self.ucorr["vu"][cond].values
            #else: #at the moment
            a_acf=self.corrs["aa"][cond].values
            au_cf=self.ucorr["au"][cond].values

            kernel=np.zeros(len(v_acf))

            #print(v_acf,va_cf,a_acf*self.mass,au_cf,dt,k0,kernel)
            #print(self.mass[cond])
            if first_order:
                ckernel.ckernel_first_order_core(v_acf,va_cf*self.mass[cond],a_acf*self.mass[cond],vu_cf,au_cf,dt,k0,kernel)
            elif hybrid:
                ckernel.ckernel_hybrid_core(v_acf,va_cf,a_acf*self.mass[cond],au_cf,va_cf*self.mass[cond],vu_cf,dt,k0,kernel)
            else:
                ckernel.ckernel_core(v_acf,va_cf,a_acf*self.mass[cond],au_cf,dt,k0,kernel)


            ikernel=cumtrapz(kernel,dx=dt,initial=0.)
            kernel=pd.DataFrame({"k":kernel,"ik":ikernel},index=self.corrs.index)
            kernels.append(kernel)

        self.kernel=pd.concat(kernels,keys= self.conditionBinMidPoints,axis=1)
        if self.saveall:
            if first_order:
                self.kernel.to_csv(self.prefix+self.kernelfile_1st,sep=" ")
            else:
                self.kernel.to_csv(self.prefix+self.kernelfile,sep=" ")

        return self.kernel

