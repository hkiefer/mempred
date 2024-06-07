from .igleplot import *
from .igleu import Potentials
import numpy as np
import pandas as pd

class IgleC(IglePlot):
    def __init__(self, *args, **kwargs):
        """
        Create an instance of the IgleC class, child of IglePlot class. 
        This class allows continuously adding time-series data for analysis
        via the push_back_xva function.
        No additional parameters.
        ----------
        """
        super(IgleC, self).__init__(*args, **kwargs)
        self.corrs_index=0
        self.kwargs = kwargs
        self.plot=False
        open((self.prefix+"weightsums.txt"), 'w').close()

    def push_back_xva(self,xva_arg):
        if isinstance(xva_arg,pd.DataFrame):
            self.xva_list=[xva_arg]
        else:
            self.xva_list=xva_arg
        for xva in self.xva_list:
	        for col in ['t','x','v','a']:
	            if col not in xva.columns:
	                raise Exception("Please provide txva data frame, or an iterable collection (i.e. list) of txva data frames. And not some other shit.")
        self.weights=np.array([xva_arg.shape[0] for xva in self.xva_list],dtype=int)
        self.weightsum=np.sum(self.weights)
        self.corrsfile="corrs_"+str(self.corrs_index)+".txt"
        self.ucorrfile="ucorr_"+str(self.corrs_index)+".txt"
        self.compute_corrs()
        self.compute_u_corr()
        with open(self.prefix+"weightsums.txt", "a") as wfile:
            wfile.write("%i "%(self.weightsum))
        self.corrs_index=self.corrs_index+1

    def compute_kernel(self, **kwargs):
        weights=np.loadtxt(self.prefix+"weightsums.txt")
        corr,ucorr=None,None
        for i in range(weights.shape[-1]):
            if corr is None:
                corr=weights[i]*pd.read_csv(self.prefix+"corrs_"+str(i)+".txt", sep=" ", index_col=0)
            corr+=weights[i]*pd.read_csv(self.prefix+"corrs_"+str(i)+".txt", sep=" ", index_col=0)
            if ucorr is None:
                ucorr=weights[i]*pd.read_csv(self.prefix+"ucorr_"+str(i)+".txt", sep=" ", index_col=0)
            ucorr+=weights[i]*pd.read_csv(self.prefix+"ucorr_"+str(i)+".txt", sep=" ", index_col=0)
        totWeight=np.sum(weights)
        self.corrs=corr/totWeight
        self.ucorr=ucorr/totWeight
        super(IgleC,self).compute_kernel()


class IgleCU(IgleC):
	def __init__(self, u0=1.0, *args, **kwargs):
		"""
		Create an instance of the IgleCU class, child of IgleC class. 
		This class additionally implements analytical potentials as described 
		in the IgleU class.
		Available potentials are NONE,HARMONIC,DOUBLEWELL.
		Aditional parameters
		----------
		u0 : float, default=1.0
		    Potential strength
		"""
		super(IgleCU, self).__init__(*args, **kwargs)
		self.potential = Potentials.HARMONIC
		self.u0=u0

	def dU(self,x):
		if(self.potential==Potentials.HARMONIC):
		    return 2*x*self.u0*self.kT
		elif(self.potential==Potentials.DOUBLEWELL):
		    return 4*np.power(x,3)-4*x*self.u0*self.kT
		else:
		    print("WARNING: analytic potential type not implemented")
		return 0.0

	def compute_u_corr(self):
		self.fe_spline=True
		super(IgleCU,self).compute_u_corr()
