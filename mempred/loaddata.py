import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

"""

Helper functions for loading finance data and weather data from different sources
Activation keys may be required for some sources

"""

import yfinance as yf #have to be installed first!
from wwo_hist import retrieve_hist_data #for weather data

#https://aroussi.com/post/python-yahoo-finance
    #https://github.com/ranaroussi/yfinance
def loaddata_yahoo(symbol, interval, start_date = '1985-01-01' , verbose_plot = False): 
    if interval == 'daily':
        xlabel = 'days'
        trj = yf.Ticker(symbol)
        trj = trj.history(period='max', start = start_date, interval = '1d')
        
    elif interval == 'weekly':
        xlabel = 'weeks'
        trj = yf.Ticker(symbol)
        trj = trj.history(period='max', start = start_date, interval = '1wk')
        
    elif interval == 'hourly':
        xlabel = 'hours'
        trj = yf.Ticker(symbol)
        trj = trj.history(interval = '1h')
        
        
    elif interval == 'minutely':
        xlabel = 'minutes'
        trj = yf.Ticker(symbol)
        trj = trj.history(period="7d", interval = '1m')
    trj = trj.reset_index()
    
    trj = trj.fillna(method='ffill')  
    if interval != 'minutely':
        
        trj = trj.replace(to_replace=0, method='ffill')
    
    if verbose_plot:
        plt.plot(trj['Close'].values, label = symbol)
        plt.xlabel(xlabel, fontsize = 'x-large')
        plt.ylabel("close", fontsize = 'x-large')
        plt.title('Loaded Trajectory')
        plt.tick_params(labelsize="x-large")
        plt.legend(loc = 'best')
        #plt.savefig("run_figures/trj.png", bbox_inches='tight')
        plt.show()
        plt.close()
        
    return trj

def load_csv(filename, start = '1980-01-01', value = 'Close', verbose_plot = False):
    
    trj = pd.read_csv(filename)
    
    start_date = start


    mask = (trj['Date'] > start_date)
    trj = trj.loc[mask]
    trj = trj.reset_index()

    if verbose_plot:
        plt.plot(trj['Close'].values)
        plt.xlabel('t', fontsize = 'x-large')
        plt.ylabel("close", fontsize = 'x-large')
        plt.title('Loaded Trajectory')
        plt.tick_params(labelsize="x-large")
        plt.legend(loc = 'best')
        #plt.savefig("run_figures/trj.png", bbox_inches='tight')
        plt.show()
        plt.close()
    return trj #returns pandas Dataframe

def load_temp(name, end_date, key, start_date = "6-DEC-2009", frequency = 24):
    api_key = key
    location_list = [name]
    hist_weather_data = retrieve_hist_data(api_key,
                                location_list,
                                start_date,
                                end_date,
                                frequency,
                                location_label = False,
                                export_csv = True,
                                store_df = True)
    
    trj = hist_weather_data[0].reset_index()
    trj = trj.drop(["index"], axis=1)
    return  trj #returns dataframe


def to_years(week, t0):
    return int(t0.year+(t0.isocalendar()[1]+week)/52.1429)

def week_range(n_weeks, t0):
    return int(to_years(np.arange(n_weeks), t0))

 
