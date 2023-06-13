import pandas as pd
#from pandas_datareader import data

import numpy as np
import matplotlib.pyplot as plt

#import quandl

import datetime as dt
from alpha_vantage.timeseries import TimeSeries #have to be installed first!
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

def loaddata(symbol, key = 'X13823W0M7RN4DRR', interval = 'daily', verbose_plot = False):
    #check latest version of alpha_vantage!
    
    if interval == 'daily':
        xlabel = 'days'
        trj = TimeSeries(key=key, output_format='pandas')
        trj, trj_data = trj.get_daily(symbol, outputsize = "full")
        #trj['date'] = trj['index']
        #trj = trj.drop(['index'], axis=1)
        trj = trj.sort_values(by='date')
        trj = trj.reset_index()
        #trj = pd.DataFrame(trj['close'], trj.index, columns = ['x'])
        
    elif interval == 'hourly':
        xlabel = 'hours'
        trj = TimeSeries(key=key, output_format='pandas')
        trj, trj_data = trj.get_intraday(symbol, interval = "60min", outputsize = "full")
        #trj['date'] = trj['index']
        #trj = trj.drop(['index'], axis=1)
        trj = trj.sort_values(by='date')
        trj = trj.reset_index()
        #trj = pd.DataFrame(trj['close'], trj.index, columns = ['x'])
        
    elif interval == 'minutely':
        xlabel = 'minutes'
        trj = TimeSeries(key=key, output_format='pandas')
        trj, trj_data = trj.get_intraday(symbol, interval = "1min", outputsize = "full")
       # trj['date'] = trj['index']
       # trj = trj.drop(['index'], axis=1)
        trj = trj.sort_values(by='date')
        trj = trj.reset_index()
        #trj = trj.reset_index()
        
    elif interval == 'weekly':
        xlabel = 'weeks'
        trj = TimeSeries(key=key, output_format='pandas')
        trj, trj_data = trj.get_daily(symbol, outputsize = "full")
        #trj['date'] = trj['index']
        #trj = trj.drop(['index'], axis=1)
        trj = trj.sort_values(by='date')
        trj = trj.reset_index()
        trj['days'] = trj.index
        trj=trj.assign(weeks=np.floor(trj["days"]/7).astype(int))
        ts_o=trj.copy()
        ts_o=ts_o.assign(date=ts_o.index)
        trj=trj.groupby("weeks").mean() #important to make sure consistent time step
        trj=trj.assign(date=ts_o.groupby("weeks")["date"].apply(lambda x: np.min(x)))
    
    trj = trj.fillna(method='ffill')  
    trj = trj.replace(to_replace=0, method='ffill')
    
    if verbose_plot:
        plt.plot(trj['4. close'], label = symbol)
        plt.xlabel(xlabel, fontsize = 'x-large')
        plt.ylabel("close", fontsize = 'x-large')
        plt.title('Loaded Trajectory')
        plt.tick_params(labelsize="x-large")
        plt.legend(loc = 'best')
        #plt.savefig("run_figures/trj.png", bbox_inches='tight')
        plt.show()
        plt.close()
       
    return trj #returns a pandas data frame

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

 
