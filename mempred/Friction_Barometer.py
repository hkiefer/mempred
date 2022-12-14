from mempred import *
import mempred as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def friction_bar(trj, time, extr_len = 1000, trunc = 100, verbose_plot = True):
    gammas = np.array([])

    for i in range(extr_len,len(trj)):
        dat = trj[i-extr_len:i]
        t = time[i-extr_len:i]
        dt=time[1] - time[0]
        
        mem = GLEPrediction(cut = len(dat), dt = dt, trunc = trunc, plot_pred = False, no_fe = True)

        kernel = mem.extractKernel([dat])
        G = kernel[4]

        #t, G = np.genfromtxt('kernel_1st.txt', usecols = (0,2), skip_header = 1).T
    
        gammas = np.append(gammas, np.max(G))
        index = np.arange(extr_len,len(gammas)+extr_len)
        
    if verbose_plot:
            
        today = trj['Date'].dt.strftime('%Y-%m-%d')[len(trj)-1]
        first = trj['Date'].dt.strftime('%Y-%m-%d')[0]
        
        fig, ax1 = plt.subplots()

        ax1.plot(trj['Close'], label = 'Stock Price')
        ax1.set_xlabel('t [days]')
        ax1.set_ylabel('Close')
        
        ax1.axvline(x=0, linestyle = '--', color = 'k', label =first)
        ax1.axvline(x=len(trj), linestyle = '--', color = 'k', label =today)
        ax1.legend(loc = 'best')

        ax2 = ax1.twinx()
        
        ax2.plot(index, gammas, color = 'red')
        ax2.set_xlabel('t [days]')
        ax2.set_ylabel('Friction $\\gamma$')
        plt.show()

            
        
    return index, gammas
