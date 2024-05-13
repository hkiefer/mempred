import numpy as np
import matplotlib.pyplot as plt
#from scipy import interpolate
#import pandas as pd
#from prophet import Prophet #not needed
from siml.detect_peaks import *
from scipy.optimize import curve_fit
from scipy.signal import butter,filtfilt



from scipy.signal import butter,filtfilt

#https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7
def butter_filter(data, cutoff, nyq, order,typ='low',cutoff2=None,extrapolate=None):
    normal_cutoff = cutoff / nyq
    
    # Get the filter coefficients 
    if typ =="low" or typ == "high":
        b, a = butter(order, normal_cutoff, btype=typ, analog=False)
    else:
        cutoff2 = cutoff2 / nyq
        b, a = butter(order, [normal_cutoff,cutoff2], btype=typ, analog=False)
    
    if extrapolate == None:
        y = filtfilt(b, a, data)
    else:
        y = filtfilt(b, a, data,padlen=extrapolate)
    return y

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


def func_polyfit(t,a):
    return a*t# + b*t**2 + c*t**3

#Function to substract the trend by high pass filter
def get_trend(t_data,x_data,n_steps,find_peaks=False,mph=0.01,N=1,verbose=False,fit=True,polyfit=False):
    t = t_data.copy()
    x = x_data.copy()

    x_mean = np.mean(x)
    x-=x_mean #important for the fit
    
    fft_y_  = np.fft.fft(x)
    fft_y = 2/len(t)*np.abs(fft_y_[:len(fft_y_)//2])
    
    fft_x_ = np.fft.fftfreq(len(x))
    fft_x = fft_x_[:len(fft_x_)//2]

    indices = np.arange(0,len(fft_y))

    if find_peaks: #find the frequencies to filter out
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
        #ax.set_xscale('log')
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
        
        if polyfit:
            popt,pcov = curve_fit(func_polyfit,t,x,p0=[0],maxfev=100000)

        else:
            popt,pcov = curve_fit(func_sin_fit,t,x,p0=param,maxfev=100000)
            param=popt

    t = np.arange(0,len(x)+n_steps)
    
    if polyfit:
        x_trend = func_polyfit(t,*popt)+x_mean

    else:
        x_trend = func_sin(t,*param)+x_mean
    return param, x_trend
    

def get_seasonal_part(t_data,x_data,n_steps,mph=0.01,N=10,verbose=True,fit=True):
    t = t_data.copy()
    x = x_data.copy()
    
    fft_y_  = np.fft.fft(x)
    fft_y = 2/len(t)*np.abs(fft_y_[:len(fft_y_)//2])
    mph = np.nanmax(fft_y)*mph
    indices_peaks = detect_peaks(fft_y, mph=mph)
    
    fft_x_ = np.fft.fftfreq(len(x))
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
        popt,pcov = curve_fit(func_sin_fit,t,x,p0=param,maxfev=100000)
        param=popt
    t = np.arange(0,len(x)+n_steps)
   
    x_seas = func_sin(t,*param) 

    return param,x_seas

def filter_and_extrapolate_time_series(t_data,x_data,cut,n_steps,verbose=False,detrend=True,fit_trend_part=True,N_trend=1,fac_high=2*np.pi,deseasonalize=True,N_seas=5,fit_seas_part=True,polyfit=False):

    t = t_data[:cut].copy()
    x = x_data[:cut].copy()
    
    fs =1/(t[1]-t[0])

    if detrend:
        
        param_trend, trend = get_trend(t,x,n_steps,find_peaks=False,mph=0.01,N=N_trend,verbose=verbose,fit=True,polyfit=polyfit)

        if polyfit:
            x_detrended = x-trend[:cut]
            x_trend = trend

        elif np.max(param_trend[:int(len(param_trend)/3)])>0:
            x_filt = x.copy()
            for i in range(int(len(param_trend)/3)):
                try:
                    x_filt = butter_filter(x_filt, cutoff=np.abs(fac_high*param_trend[i]), nyq=1*0.5, order=1,typ='high',cutoff2=None)
                except:
                    continue
        
            x_shift = np.mean(x_filt)
            x_detrended = x_filt-x_shift

            t_pred = np.arange(0,len(x)+n_steps)
            
            if fit_trend_part:
                x_trend = x-x_filt-np.mean(x-x_filt)
                param_trend2,pcov = curve_fit(func_sin_fit,t,x_trend,p0=param_trend,maxfev=100000)

                if (np.mean((x_trend-func_sin(t,*param_trend2)[:cut])**2)/np.mean((x_trend)**2)) > 0.01:
                    
                    x_trend = func_sin(t_pred,*param_trend)+np.mean(x-x_filt)
                    x_detrended = x - x_trend[:cut]
                
                else:
                    x_trend = func_sin(t_pred,*param_trend2)+np.mean(x-x_filt)
                    x_trend = np.append(x_trend,func_sin(t_pred,*param_trend2)[cut:cut+n_steps])+np.mean(x-x_filt)
                    x_trend = x_trend+x_shift
            else:
                x_trend = func_sin(t_pred,*param_trend)+np.mean(x-x_filt)
                x_detrended = x - x_trend[:cut]

        else:
            x_detrended = x-trend[:cut]
            x_trend = trend
    else:
        x_detrended = x
        trend = np.zeros(len(x)+n_steps)
        x_trend  = np.zeros(len(x)+n_steps)


    if deseasonalize:
    
        param_seas, seas = get_seasonal_part(t,x_detrended,n_steps,mph=0.01,N=N_seas,verbose=verbose,fit=True)

    
        fw = fs/len(x_detrended)

        if np.max(param_seas[:int(len(param_seas)/3)])>0:
                
                x_filt = x_detrended.copy()
                for i in range(int(len(param_seas)/3)):
                    try:
                        x_filt -= butter_filter(x_filt, cutoff=np.abs(param_seas[i])-fw*10, cutoff2=np.abs(param_seas[i])+fw*10, nyq = fs*0.5,order=2,typ='band')

                        #plt.plot(x_detrended)
                        #plt.plot(x_filt)
                        #plt.show()
                    except:
                        continue
                
                x_filtered = x_filt
                x_seas = x_detrended-x_filt 

                t_pred = np.arange(0,len(x)+n_steps)

                if fit_seas_part:
                    param_seas2,pcov = curve_fit(func_sin_fit,t,x_seas,p0=param_seas,maxfev=100000)
                    
                    #x_seas = func_sin(t_pred,*param_seas2)+x_trend
                    x_seas= np.append(x_seas,func_sin(t_pred,*param_seas2)[cut:cut+n_steps]) +x_trend

                    
                else:
                    x_seas = seas + x_trend
                    x_filtered= x - x_seas[:cut]
        else:
            x_seas = seas + trend
            x_filtered = x - x_seas[:cut]
            

    else:
        x_filtered = x_detrended
        x_seas =  x_trend

    #plt.plot(x)
    #plt.plot(x_seas)
    #plt.show()
    #plt.plot(x_filtered)

    return x_filtered,x_seas,x_trend

