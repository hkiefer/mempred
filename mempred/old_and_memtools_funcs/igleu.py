from .igleplot import *
from enum import Enum
import numpy as np

class Potentials(Enum):
    NONE = 0
    HARMONIC = 1
    DOUBLEWELL = 2


class IgleU(IglePlot):

  def __init__(self, *args, u0=1.0,  **kwargs):
        """
        Create an instance of the IgleU class, child of IglePlot class. 
        This class contains a dictionary of analytical potentials, 
        that are switched on as classInstance.potential = Potentials.HARMONIC.
        Available potentials are NONE,HARMONIC,DOUBLEWELL.
        Aditional parameters
        ----------
	    u0 : float, default=1.0
	        Potential strength
        """
        super(IgleU, self).__init__(*args, **kwargs)
        self.potential = Potentials.HARMONIC
        self.fe_spline = True
        self.u0 = u0

  def dU(self,x):
        if(self.potential==Potentials.NONE):
          return x*0.0
        elif(self.potential==Potentials.HARMONIC):
          return 2*x*self.u0*self.kT
        elif(self.potential==Potentials.DOUBLEWELL):
          return (4*np.power(x,3)-4*x)*self.u0*self.kT
        else:
          print("WARNING: analytic potential type not implemented")
          return 0.0


