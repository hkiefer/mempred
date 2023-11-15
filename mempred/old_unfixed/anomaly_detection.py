import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from mempred import *

import datetime as dt


class Anomaly_Detection:
    def __init__(self,step_size, trunc, conf_interval, date_col, value):
        self.step_size =  step_size
        self.trunc = trunc
        self.conf_interval = conf_interval
        self.date_col = date_col
        self.value = value

    def Error(self, y_pred, y):
        return np.absolute(y-y_pred)
    
    def SquaredError(self, y_pred,y):
        return (y_pred-y)**2

    def create_pred(self, data, cut):
    
        predict=GLEPrediction(cut = cut ,dt = 0.5, trunc=self.trunc, no_fe=True, plot_pred = False)
        kernel = predict.extractKernel([data[self.value].values])

        GLE=predict.predictGLE([data[self.value].values], n_steps=self.step_size + 1, n_preds = 10, return_full_trjs=True, zero_noise = False, alpha = 1)
        
        pred = GLE[2]
        real = data[self.value][cut:cut+self.step_size]
        new_dates = data[self.date_col][:cut] + dt.timedelta(days=self.step_size)

        new_dates = new_dates[-self.step_size:].values
        new_dates = pd.to_datetime(new_dates, format='%Y%m%d')
        #new_dates = new_dates.dt.round('D')
        
        pred_arr = np.array([new_dates,real,pred])
        
        predictions_df = pd.DataFrame(pred_arr.T, columns=['Date','Actual', 'Prediction'])
        predictions_df['Date'] = predictions_df['Date'].dt.round('D')
        #plt.plot(predictions_df['Actual'])
        #plt.plot(predictions_df['Prediction'])
        #plt.show()
        return predictions_df
    
    def search_anomaly(self, data, start, end):
        data[self.date_col] = pd.to_datetime(data[self.date_col])
        mask = (data[self.date_col] > start) & (data[self.date_col] <= end)
        data = data.loc[mask]
        data = data.reset_index()
        data = data.fillna(method = 'ffill')
        print('starting prediction at: ' + str(data[self.date_col][1000]))
        length = len(data)
        cuts = np.arange(1000,length,self.step_size)
        
        found_anomalies = []
        for cut in cuts:
            if (length - cut) < self.step_size:
                
                break
                
            predictions_df = self.create_pred(data, cut)
            #predictions_df['Error'] = self.SquaredError(predictions_df['Prediction'].values, predictions_df['Actual'].values)
            predictions_df['Error'] = self.Error(predictions_df['Prediction'].values, predictions_df['Actual'].values)
            #print(predictions_df['Error'])
            
            found = predictions_df['Date'].loc[predictions_df['Error'] > self.conf_interval]
            found = pd.to_datetime(found, format='%Y%m%d')
            found = found.dt.round('D')
            if len(found) > 0:
                print('found anomaly!')
                for j in range(len(found.values)):
                    print(found.values[j])
                
                    found_anomalies.append(found.values[j])
               # print(found_anomalies)
                
                
        if len(found_anomalies) == 0:
                    print('no anomalies found')
        found_anomalies = pd.to_datetime(found_anomalies)
        return found_anomalies
    
    def plot_anomalies(self,data, found_anomalies):
        marker = data.loc[data[self.date_col].isin(found_anomalies)]
        plt.plot(data.index, data[self.value])
        plt.scatter(marker.index, marker[self.value], color = 'red')
        plt.show()
       