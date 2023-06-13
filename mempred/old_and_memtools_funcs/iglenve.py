from .igleplot import *
import numpy as np


k = 1.38064852e-23 # m2 kg s-2 K-1
amu = 1.66054e-27 # kg
#kT= k*temp*1e18/(amu*1e24) # amu nm^2/ps^2
#temp= kT * amu *1e24 /(1e18*k) # K

class IgleNVE(IglePlot):

    def __init__(self, *args, **kwargs):
        """
        Create an instance of the IgleNVE class, child of IglePlot class. 
	This class computes kT of the system, but needs a mass.
        No additional parameters.
        ----------
        """
        super(IgleNVE, self).__init__(*args, **kwargs)
        self.kTs=None

    def compute_mass(self):
        NotImplementedError("Cannot compute mass from NVE ensemble since temperature is ill-defined.")

    def compute_kT(self):
        assert self.mass is not None, "Need a given mass."
        assert self.corrs is not None, "Velocity autocorrelation needs to be computed first."

        if self.verbose:
            print("Calculating temperatures...")
            print("Use mass:", self.mass)

        v2=self.corrs["vv"].iloc[0]
        self.kT=v2*self.mass
        self.kT_list=[self.kT]

        if self.verbose:
            print("Found kT:", self.kT)
            print("T:", self.kT*amu*1e24/(k*1e18))

    def compute_kTs(self):
        assert self.xva_list is not None, "Need an xva_list to compute kT_list on."
        assert self.mass is not None, "Need a given mass."
        if self.verbose:
            print("Calculating temperatures...")
            print("Use mass:", self.mass)
        self.kT_list, self.T_list = [], []
        for i,xva in enumerate(self.xva_list):
            v2=(xva["v"]**2).mean()
            self.kT_list.append(v2*self.mass)
            self.T_list.append(v2*self.mass*amu*1e24/(k*1e18))

        if self.verbose:
            print("Found kT_list:", self.kT_list)
            print("Found T_list:", self.T_list)


    def compute_u_corr(self):
        assert len(self.kT_list) is len(self.xva_list), "IgleNVE needs member kT_list to be a list of the size of xva_list!"
        if self.fe_spline is None:
            raise Exception("Free energy has not been computed.")
        if self.verbose:
            print("Calculate a/v grad(U(x)) correlation function...")

        # get target length from first element and trunc
        ncorr=self.xva_list[0][self.xva_list[0].index < self.trunc].shape[0]


        self.ucorr=pd.DataFrame({"au":np.zeros(ncorr)}, \
        index=self.xva_list[0][self.xva_list[0].index < self.trunc].index\
              -self.xva_list[0].index[0])
        if self.first_order or self.hybrid:
            self.ucorr["vu"]=np.zeros(ncorr)

        for weight,xva,kT in zip(self.weights,self.xva_list,self.kT_list):
            x=xva["x"].values
            a=xva["a"].values
            corr=correlation(a,self.dU(x)/self.kT*kT,subtract_mean=False)
            self.ucorr["au"]+=weight*corr[:ncorr]

            if self.first_order or self.hybrid:
                v=xva["v"].values
                corr=correlation(v,self.dU(x)/self.kT*kT,subtract_mean=False)
                self.ucorr["vu"]+=weight*corr[:ncorr]

        self.ucorr/=self.weightsum

        if self.saveall:
            self.ucorr.to_csv(self.prefix+self.ucorrfile,sep=" ")

