from __future__ import print_function
import numpy as np
import pandas as pd

#from .mempred import *
from .GBMpred import *
from .GLEprediction import *
from .FFT_decomp_tools import *
from .GLE_filter_tools import *
from .correlation_functions_expos import *
from .extract_kernel import *
from .loaddata import *

from .Friction_Barometer import *
from .performance_GLE import *

from .mp_GridSearch import *
#from .old_unfixed.anomaly_detection import *

from .correlation import *
from .xvaframe import *


def ver():
    """
    Show the module version.
    """
    print("This is mempred version 3.0")
    print("Latest feature: integrate Mitterwallner method")
    
def xframe(x, time, fix_time=True, round_time=1.e-5, dt=-1):
    """
    Creates a pandas dataframe (['t', 'x']) from a trajectory. Currently the time
    is saved twice, as an index and as a separate field.

    Parameters
    ----------
    x : numpy array
        The time series.
    time : numpy array
        The respective time values.
    fix_time : bool, default=False
        Round first timestep to round_time precision and replace times.
    round_time : float, default=1.e-4
        When fix_time is set times are rounded to this value.
    dt : float, default=-1
        When positive, this value is used for fixing the time instead of
        the first timestep.
    """
    x=np.asarray(x)
    time=np.asarray(time)
    if fix_time:
        if dt<0:
            dt=np.round((time[1]-time[0])/round_time)*round_time
        time=np.linspace(0.,dt*(x.size-1),x.size)
        time[1]=dt
    df = pd.DataFrame({"t":time.ravel(),"x":x.ravel()}, index=time.ravel())
    df.index.name='#t'
    return df

def xvframe(x,v,time,round_time=1.e-5,fix_time=True,dt=-1):
    """
    Creates a pandas dataframe (['t', 'x', 'v']) from a trajectory. Currently the time
    is saved twice, as an index and as a separate field.

    Parameters
    ----------
    x : numpy array
        The time series' positions.
    v : numpy array
        The time series' velocities.
    time : numpy array
        The respective time values.
    fix_time : bool, default=False
        Round first timestep to round_time precision and replace times.
    round_time : float, default=1.e-4
        When fix_time is set times are rounded to this value.
    dt : float, default=-1
        When positive, this value is used for fixing the time instead of
        the first timestep.
    """
    x=np.asarray(x)
    v=np.asarray(v)
    time=np.asarray(time)
    if fix_time:
        if dt<0:
            dt=np.round((time[1]-time[0])/round_time)*round_time
        time=np.linspace(0.,dt*(x.size-1),x.size)
        time[1]=dt
    df = pd.DataFrame({"t":time.flatten(),"x":x.flatten(),"v":v.flatten()}, index=np.round(time/round_time)*round_time)
    df.index.name='#t'
    return df

def xvaframe(x,v,a,time,round_time=1.e-5,fix_time=True,dt=-1):
    """
    Creates a pandas dataframe (['t', 'x', 'v', 'a']) from a trajectory. Currently the time
    is saved twice, as an index and as a separate field.

    Parameters
    ----------
    x : numpy array
        The time series' positions.
    v : numpy array
        The time series' velocities.
    a : numpy array
        The time series' accelerations.
    time : numpy array
        The respective time values.
    fix_time : bool, default=False
        Round first timestep to round_time precision and replace times.
    round_time : float, default=1.e-4
        When fix_time is set times are rounded to this value.
    dt : float, default=-1
        When positive, this value is used for fixing the time instead of
        the first timestep.
    """
    x=np.asarray(x)
    v=np.asarray(v)
    a=np.asarray(a)
    time=np.asarray(time)
    if fix_time:
        if dt<0:
            dt=np.round((time[1]-time[0])/round_time)*round_time
        time=np.linspace(0.,dt*(x.size-1),x.size)
        time[1]=dt
    df = pd.DataFrame({"t":time.flatten(),"x":x.flatten(),"v":v.flatten(),"a":a.flatten()}, index=np.round(time/round_time)*round_time)
    df.index.name='#t'
    return df

def compute_a(xvf):
    """
    Computes the acceleration from a data frame with ['t', 'x', 'v'].

    Parameters
    ----------
    xvf : pandas dataframe (['t', 'x', 'v'])
    """
    #diffs=xvf.shift(-1)-xvf.shift(1)
    diffs=xvf-xvf.shift(1)
    dt=xvf.iloc[1]["t"]-xvf.iloc[0]["t"]
    xva=pd.DataFrame({"t":xvf["t"],"x":xvf["x"],"v":xvf["v"],"a":diffs["v"]/(2.*dt)},index=xvf.index)
    xva = xva[['t', 'x', 'v', 'a']]
    xva.index.name='#t'

    return xva.dropna()

def compute_v(xf):
    """
    Computes the velocity from a data frame with ['t', 'x'].

    Parameters
    ----------
    xf : pandas dataframe (['t', 'x'])
    """
    diffs=xf-xf.shift(-1)
    dt=xf.iloc[1]["t"]-xf.iloc[0]["t"]
    xv=pd.DataFrame({"t":xf["t"],"x":xf["x"],"v":diffs["x"]/(dt)},index=xf.index)
    xv = xv[['t', 'x', 'v']]
    xv.index.name='#t'

    return xv.dropna()


def compute_va(xf, correct_jumps=False, jump=360, jump_thr=270):
    """
    Computes velocity and acceleration from a data frame with ['t', 'x'] as
    returned by xframe.

    Parameters
    ----------
    xf : pandas dataframe (['t', 'x'])

    correct_jumps : bool, default=False
        Jumps in the trajectory are removed (relevant for periodic data).
    jump : float, default=360
        The size of a jump.
    jump_thr : float, default=270
        Threshold for jump detection.
    """
    diffs=xf-xf.shift(1)
    dt=xf.iloc[1]["t"]-xf.iloc[0]["t"]
    if correct_jumps:
        diffs.loc[diffs["x"] < jump_thr,"x"]+=jump
        diffs.loc[diffs["x"] > jump_thr,"x"]-=jump

    ddiffs=diffs.shift(-1)-diffs
    sdiffs=diffs.shift(-1)+diffs
    v_arr = np.gradient(xf["x"].values,dt)
    a_arr = np.gradient(v_arr,dt)
    xva=pd.DataFrame({"t":xf["t"],"x":xf["x"],"v":sdiffs["x"]/(2.*dt),"a":ddiffs["x"]/dt**2},index=xf.index)
    #xva = pd.DataFrame({"t":xf["t"],"x":xf["x"],"v":v_arr,"a":a_arr},index=xf.index)
    xva = xva[['t', 'x', 'v', 'a']]
    xva.index.name='#t'

    return xva.dropna()

def compute_a_delta(xvf):
    """
    Computes the acceleration from a data frame with ['t', 'x', 'v'].
    If velocities are known from this method computes acceleration only from the forward averages.
    The va correlation function is then decaying in one timestep not smear out the delta-function in the random force.

    Parameters
    ----------
    xvf : pandas dataframe (['t', 'x', 'v'])

    """
    diffs=xvf-xvf.shift(1)
    dt=xvf.iloc[1]["t"]-xvf.iloc[0]["t"]

    xva=pd.DataFrame({"t":xvf["t"],"x":xvf["x"],"v":xvf["v"],"a":diffs["v"]/(dt)},index=xvf.index)
    xva = xva[['t', 'x', 'v', 'a']]
    xva.index.name='#t'

    return xva.dropna()

def computeFrozenKernel(numpy_tf, trunc=None, file=None,kT=2.4943):
    """
    Computes the memory kernel from a 2D numpy array which contains times in the first and forces in the second column.

    Parameters
    ----------
    numpy_tf: 2D numpy array with times and forces

    """
    tf=numpy_tf
    meanF = np.mean(tf[:,1])
    xf=xframe(tf[:,1]-meanF,tf[:,0])
    kernel_frozen = pdcorr(xf,"x","x",trunc=trunc,oname="ff")
    ikernel=cumtrapz(kernel_frozen['ff'],x=kernel_frozen.index,initial=0.)
    kernel_ff=pd.DataFrame({"k":kernel_frozen['ff']/(kT),"ik":ikernel/(kT)},index=kernel_frozen.index.values)
    if file is not None:
        kernel_ff.to_csv(file,sep=" ")
    return kernel_ff
