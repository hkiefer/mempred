import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mempred import *

"""

Modules for grid search of the GLE prediction (find optimal extraction hyperparameters)

"""
def squared_log_error(real, pred):
    return (np.log(pred+1) - np.log(real+1))**2


def mp_grid_search(trj,value,cut, alphas, truncs,ts,fit_kernel = False,n_steps = 100):
    best_error_mean = 1
    best_error = np.zeros(n_steps)
    pred_GLE_best = np.zeros(n_steps)
    best_trunc = truncs[0]
    best_alpha = alphas[0]
    best_th = ts[0]
    bins = 100
    for trunc in truncs:
        for th in ts:
            for alpha in alphas:
                predict=GLEPrediction(bins=bins,  cut = cut ,trunc=trunc, dt = 1, no_fe=False, plot_pred = False,kde_mode=False)#,kde_mode=True)
                kkernel = predict.extractKernel([trj[value].values],fit_kernel = fit_kernel)
                predGLE=predict.predictGLE([trj[value].values], n_steps=n_steps, n_preds = 10, return_full_trjs=True, zero_noise = False,cond_noise = th,alpha = alpha)
                pred_GLE = predGLE[2]
                real = trj[value][cut:cut+n_steps].values
                error_GLE = squared_log_error(real, pred_GLE)
                if np.mean(error_GLE) < best_error_mean:
                    best_error_mean = np.mean(error_GLE)
                    pred_GLE_best = predGLE[2]
                    best_error = error_GLE
                    best_trunc = trunc
                    best_alpha = alpha
                    best_th = th
                #print(best_error_mean)    
    print('optimal alpha : ' + str(best_alpha))
    print('optimal trunc : ' + str(best_trunc))
    print('optimal conditional time : ' + str(best_th))
    return best_error, pred_GLE_best,best_trunc,best_alpha,best_th