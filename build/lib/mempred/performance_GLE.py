import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from mempred import *

import time

class performance_GLE:
    """
    different functions combined to GLE predictions
     - predict expected total return for n_steps into future
     - Calculates the Performance (RMSLE) for GLE predictions with respect to known data in the past
    """
    def __init__(self, n_starts, n_steps):
        self.n_starts = n_starts #number of different starting points of prediction (averaged in the end)
        self.n_steps = n_steps #number of GLE-prediction steps for every start points in n_start
    
    def squared_log_error(self,real, pred):
        return (np.log(pred + 1) - np.log(real + 1))**2

    def squared_error(self,real, pred):
        return (pred - real)**2
    
    def class_acc(self,actual, predicted,x0):
        return (np.sign(actual - x0) == np.sign(predicted - x0))


    
    def predict_change(self, symbols, interval, n_steps = None, verbose_plot = False):
        if n_steps == None:
            n_steps = self.n_steps
        change_preds = np.array([])
        for symbol in symbols: 

            trj = loaddata_yahoo(symbol, interval = interval, start_date = '2000-01-01', verbose_plot = verbose_plot)

            cut = len(trj)

            predict=GLEPrediction(bins=40,  cut = cut ,trunc=100, dt = 1, last_value_correction=True, no_fe=False, plot_pred = verbose_plot)
            kernel = predict.extractKernel([trj["Close"].values], fit_kernel = False)

            predGLE=predict.predictGLE([trj["Close"].values], n_steps=n_steps + 1, n_preds = 10, return_full_trjs=True, zero_noise = False, Langevin = False,cond_noise=1)


            change = (predGLE[2][-1] - predGLE[1][cut]) / predGLE[1][cut] * 100
            change_preds = np.append(change_preds, change)

            print('{} : {:.2f} %'.format(symbol, change))

        return change_preds

    def backtest(self, trj, value):


        start = time.time()
        
        acc_GLE_mean = np.zeros(self.n_steps)
        error_GLE_mean = np.zeros(self.n_steps)
        #index = np.array([])
        for j in range(0,self.n_starts):
            
            cut = len(trj)-j-self.n_steps

    
            predict=GLEPrediction(bins=10,  cut = cut ,trunc=100, dt = 1, last_value_correction=True, no_fe=False, plot_pred = False)
            kernel = predict.extractKernel([trj[value].values])
            predGLE=predict.predictGLE([trj[value].values], n_steps=self.n_steps+1, n_preds = 10, return_full_trjs=True, zero_noise = False,cond_noise=1)


            pred = predGLE[1][cut:cut+self.n_steps]
            real = trj[value][cut:cut+self.n_steps].values
            
            x0 = trj[value][cut-1]
            error =self.squared_log_error(real, pred)
            
            error_GLE_mean = error_GLE_mean + error
            
            
            acc = self.class_acc(real,pred,x0)
            acc_GLE_mean = acc_GLE_mean + acc
            
        
            
        error_GLE_mean = (error_GLE_mean/self.n_starts)**0.5
        
        acc_GLE_mean = (acc_GLE_mean/self.n_starts)

        print("> Compilation Time : ", time.time() - start)

        return error_GLE_mean, acc_GLE_mean

    def get_scores(self, symbols, start_date = "2001-01-01", interval = "daily", verbose_plot = False, metric ='RMSLE'):

        i = 0
        for symbol in symbols:
            
            print("predict " + str(symbol) + "...")
            trj = loaddata_yahoo(symbol = symbol, interval = interval, start_date = start_date, verbose_plot = False)
            scores = self.backtest(trj, value = 'Close')
            if metric == 'RMSLE':
                score = scores[0]
                
            elif metric == 'MDA':
                score = scores[1]
                          
            else:
                print('Please choose a metric! RMSLE or MDA')
                break
            
            if i == 0:
                scores_all = pd.DataFrame(data = score , columns = [str(symbol)])

            else:
                scores_all[str(symbol)] = score
            i += 1


        if verbose_plot:
            for symbol in symbols:
                index = np.arange(1, self.n_steps+1)
                plt.scatter(index, scores_all[symbol], label = symbol)
                plt.legend(loc="best", fancybox = False)
                plt.xlabel("steps")
                plt.ylabel(metric)
                plt.xlim(0, self.n_steps+(self.n_steps*0.4))
                #plt.xticks(np.arange(0,22,3))
                #plt.yticks(np.arange(0,1.1,0.2))
                #plt.savefig('RMSE_Time_Steps_Startpoints.pdf', bbox_inches='tight')
            plt.show()
        return scores_all
        
        def predict_quantile(trj, cut, n_pred, pred_type = 'GLE'):
            
            if pred_type == 'GLE': 
                
                predict=GLEPrediction(bins=10,  cut = cut ,trunc=100, dt = 1, last_value_correction=True, no_fe=False, plot_pred = False)
                kernel = predict.extractKernel([trj["Close"].values])

                simulations = np.zeros(n_pred)
                np.set_printoptions(threshold=5)
                for run in range(runs):    
    # Set the simulation data point as the last stock price for that run
                    simulations[run] = predict.predictGLE([trj["Close"].values], n_steps=self.n_steps+1, n_preds = 1,  return_full_trjs=True, zero_noise = False)[cut + n_steps  -1]

                

            elif pred_type == 'Langevin':
                predict=GLEPrediction(bins=10,  cut = cut ,trunc=100, dt = 1, last_value_correction=True, no_fe=False,
                                      plot_pred = False) 
                kernel = predict.extractKernel([trj["Close"].values])

                simulations = np.zeros(n_pred)
                np.set_printoptions(threshold=5)
                for run in range(runs):    
    # Set the simulation data point as the last stock price for that run
                    simulations[run] = predict.predictGLE([trj["Close"].values], n_steps=self.n_steps+1, n_preds = 1,  return_full_trjs=True, zero_noise = False, Langevin = True,cond_noise=1)[cut + n_steps  -1]
             
            elif pred_type == 'Langevin':
                simulations = np.zeros(n_pred)
                np.set_printoptions(threshold=5)
                for run in range(runs):    

                    simulations[run] = predictLangevin(trj, value = 'Close', cut = cut, dt = 1, n_steps = self.n_steps, scen_size = 1)[cut + self.n_steps  -1]
                
            
            q = np.percentile(simulations, 1) 
            plt.hist(simulations,bins=200)
            plt.show()
            
            return q, simulations.mean()
                







