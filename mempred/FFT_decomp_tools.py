import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from prophet import Prophet
from siml.detect_peaks import *
from scipy.optimize import curve_fit
from scipy.signal import butter,filtfilt



def decompose_Prophet(trj,value,cut,n_steps,yearly=True,weekly=True,daily=True,mode="additive",freq='D'):
    
    trj["date_time"] = pd.to_datetime(trj["date_time"]) #for prophet we need a date_time column
    data = trj.copy()
    data.index = data["date_time"]
    trj["decimal_date"] = data.index.year + (data.index.dayofyear -1)/365
    m = Prophet(yearly_seasonality=yearly, daily_seasonality = daily, weekly_seasonality = weekly,seasonality_mode=mode)
    if mode == "multiplicative":
        print("add regressor for yearly")
        m.add_seasonality('yearly', period=365, fourier_order=8, mode='additive')
        #m.add_regressor('regressor', mode='additive')
    df = pd.concat([trj['date_time'][:cut], trj[value][:cut]], axis=1, keys=['ds', 'y'])
    m.fit(df)


    future = m.make_future_dataframe(periods=n_steps,freq=freq)
    #future.tail()
    predProph = m.predict(future)
    #predProph.head()
    
    x_seas = predProph["yhat"].values
    x_noise = trj[value][:cut].values - x_seas[:cut]
    
    return predProph, x_seas, x_noise


def func_sin(t,*params):

    
    N = int(len(params)/3)
    func = np.zeros(len(t))
    fs = params[:N]
    As = params[N:2*N]
    phases = params[2*N:]
    for i in range(0,N):
        A,f,phi = np.abs(As[i]),np.abs(fs[i]),phases[i]      
        func += A * np.cos(2*np.pi * f * t + phi)
        
    return func 
def func_sin_fit(t,*params):

   
    N = int(len(params)/3)
    func = np.zeros(len(t))
    fs = params[:N]
    As = params[N:2*N]
    phases = params[2*N:]
    for i in range(0,N):
        A,f,phi = np.abs(As[i]),np.abs(fs[i]),phases[i]      
        func += A * np.cos(2*np.pi * f * t + phi)
        
    return np.real(func)


def extrapolate_fourier_analysis(t,x,cut,n_steps,deg_polyfit=1,mph=0.01,N=10,verbose=True,fit=True,lp_trend=False,lp_cut=1000):
    t = t[:cut]
    x = x[:cut]
    #yvalues_trend = scipy.signal.savgol_filter(yvalues,25,1)
    #yvalues_detrended = yvalues - yvalues_trend
    # we calculate the trendline and detrended signal with polyfit
    if deg_polyfit > 0:
        if lp_trend:
            fs = 1       # sample rate, Hz
            cutoff = 1/lp_cut     # desired cutoff frequency of the filter, Hz
            nyq = 0.5 * fs  # Nyquist Frequency
            order = 1      # sin wave can be approx represented as quadratic
            typ='low'
            cutoff2 = None
            xvalues_trend = butter_filter(x, cutoff, nyq, order,typ=typ,cutoff2=cutoff2)
        else:
            z2 = np.polyfit(t, x, deg_polyfit)
            p2 = np.poly1d(z2)
            xvalues_trend = p2(t)
        
        xvalues_detrended = x - xvalues_trend

        
    else:
        xvalues_trend = np.zeros(len(t)) + np.mean(x)
        xvalues_detrended = x - xvalues_trend
    
    fft_y_  = np.fft.fft(xvalues_detrended)
    fft_y = 2/len(t)*np.abs(fft_y_[:len(fft_y_)//2])
    mph = np.nanmax(fft_y)*mph
    indices_peaks = detect_peaks(fft_y, mph=mph)
    
    fft_x_ = np.fft.fftfreq(len(xvalues_detrended))
    fft_x = fft_x_[:len(fft_x_)//2]

    indices = np.arange(0,len(fft_y))
    # The number of harmonics we want to include in the reconstruction
    indices = indices[np.absolute(fft_y[indices_peaks]).argsort()][::-1][:N]
    indices_peaks = indices_peaks[indices]
    
    if verbose:
        print("found peak frequencies")
        #print(fft_x[indices_peaks], fft_y[indices_peaks], 1/fft_x[indices_peaks])
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(fft_x, fft_y)
        ax.scatter(fft_x[indices_peaks], fft_y[indices_peaks], color='red',marker='D')
        ax.set_title('frequency spectrum of Data', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        ax.set_xlabel(r'Frequency [1/$\Delta t$]', fontsize=14)
        ax.set_xscale('log')
        #for idx in indices_peaks:
            #x,y = fft_x[idx], fft_y[idx]
            #text = "  f = {:.2f}".format(x,y)
            #ax.annotate(text, (x,y))
        plt.show()

    fs = fft_x[indices_peaks]
    phases = np.angle(fft_y[indices_peaks])
    As = fft_y[indices_peaks]
    param = np.append(np.append(fs,As),phases)
    if fit:
        popt,pcov = curve_fit(func_sin_fit,t,xvalues_detrended,p0=param,maxfev=100000)
        param=popt
    t = np.arange(0,len(x)+n_steps)
    if deg_polyfit > 0:
        xvalues_trend = p2(t)
    else:
        xvalues_trend  = np.zeros(len(t)) + np.mean(x)
    x_seas = func_sin(t,*param) + xvalues_trend
    x_noise = x - x_seas[:cut]
    return param,x_seas, x_noise, xvalues_trend

#https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7
def butter_filter(data, cutoff, nyq, order,typ='low',cutoff2=None):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    if typ =="low" or typ == "high":
        b, a = butter(order, normal_cutoff, btype=typ, analog=False)
    else:
        b, a = butter(order, [normal_cutoff,cutoff2/nyq], btype=typ, analog=False)
    y = filtfilt(b, a, data)
    return y



def extrapolate_fourier_analysis_trend(t,x,cut,n_steps,find_peaks=False,mph=0.01,N=10,verbose=True,fit=True):
    t = t[:cut].copy()
    x = x[:cut].copy()

    x_mean = np.mean(x)
    x-=x_mean
    
    fft_y_  = np.fft.fft(x)
    fft_y = 2/len(t)*np.abs(fft_y_[:len(fft_y_)//2])
    
    fft_x_ = np.fft.fftfreq(len(x))
    fft_x = fft_x_[:len(fft_x_)//2]

    indices = np.arange(0,len(fft_y))

    if find_peaks:
        mph = np.nanmax(fft_y)*mph
        indices_peaks = detect_peaks(fft_y, mph=mph)
        # The number of harmonics we want to include in the reconstruction
        #indices = indices[np.absolute(fft_y[indices_peaks]).argsort()][::-1][:N]
        indices=indices[:np.min([N,len(indices_peaks)])]
        indices_peaks =indices_peaks[indices]

    else:
        indices_peaks = np.arange(0,len(fft_y))[:N]#[0,1,2,3,4,5]
    
    
    if verbose:
        print("found peak frequencies")
        #print(fft_x[indices_peaks], fft_y[indices_peaks], 1/fft_x[indices_peaks])
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(fft_x, fft_y)
        ax.scatter(fft_x[indices_peaks], fft_y[indices_peaks], color='red',marker='D')
        ax.set_title('frequency spectrum of Data', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        ax.set_xlabel(r'Frequency [1/$\Delta t$]', fontsize=14)
        ax.set_xscale('log')
        #for idx in indices_peaks:
            #x,y = fft_x[idx], fft_y[idx]
            #text = "  f = {:.2f}".format(x,y)
            #ax.annotate(text, (x,y))
        plt.show()

    fs = fft_x[indices_peaks]
    phases = np.angle(fft_y[indices_peaks])
    As = fft_y[indices_peaks]
    param = np.append(np.append(fs,As),phases)
    if fit:
        popt,pcov = curve_fit(func_sin_fit,t,x,p0=param,maxfev=100000)
        param=popt
    t = np.arange(0,len(x)+n_steps)
    
    x_trend = func_sin(t,*param)+x_mean
    x_res = x - x_trend[:cut]+x_mean
    return param,x_res, x_trend