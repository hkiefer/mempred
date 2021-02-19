import pandas as pd
import numpy as np
#import quandl
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()



#--------------------------------------------------- GEOMETRIC BROWNIAN MOTION ------------------------------------------------

# Parameter Definitions

# So    :   initial stock price
# scen_size : number of predictions
# dt    :   time increment -> a day in our case
# T     :   length of the prediction time horizon(how many time points to predict, same unit with dt(days))
# N     :   number of time points in prediction the time horizon -> T/dt
# t     :   array for time points in the prediction time horizon [1, 2, 3, .. , N]
# mu    :   mean of historical daily returns
# sigma :   standard deviation of historical daily returns
# b     :   array for brownian increments
# W     :   array for brownian path

def predictGBM(trj, value = 'Close', cut=1000, dt = 1, n_steps=1000, scen_size = 10, plot_pred = False):
    returns = (trj[value] - trj[value].shift(-1))/trj[value].shift(-1)               
    returns = returns[:cut]
    # Parameter Assignments
    So = trj[value][cut]

    N = n_steps / dt
    t = np.arange(1, int(N) + 1)
    mu = np.mean(returns)
    sigma = np.std(returns)
    
    b = {str(scen): np.random.normal(0, 1, int(N)) for scen in range(1, scen_size + 1)}
    W = {str(scen): b[str(scen)].cumsum() for scen in range(1, scen_size + 1)}

    # Calculating drift and diffusion components
    drift = (mu - 0.5 * sigma**2) * t
    #print(drift)
    diffusion = {str(scen): sigma * W[str(scen)] for scen in range(1, scen_size + 1)}
    #print(diffusion)
    # Making the predictions
    S = np.array([So * np.exp(drift + diffusion[str(scen)]) for scen in range(1, scen_size + 1)]) 
    S = np.hstack((np.array([[So] for scen in range(scen_size)]), S)) # add So to the beginning series
    # Plotting the simulations
    #print("plotting the simulations")
    if plot_pred :
        plt.figure()
        for i in range(scen_size):
            plt.title("Daily Volatility: " + str(sigma))
            plt.plot(S[i, :])
            plt.ylabel("predicted price", fontsize="x-large")
            plt.tick_params(labelsize="x-large")
            plt.xlabel('Prediction days', fontsize="x-large")
        plt.show()

    #calculaing Mean of simulations

    #S_mean = np.zeros(len(S[1,:]))
    #S_m = S[0,:]
    #for i in range(1,scen_size):
        #S_mean = S_mean + S[i, :]
        #S_m = np.concatenate((S_m, S[i,:]), axis = 0)
   
    S_mean = np.mean(S, axis = 0)
    
    S_error = np.std(S, axis = 0)
    #print(len(S_mean), len(S_error))
    S_index = np.arange(0, n_steps, 1)
    index_real=np.arange(0,len(trj[value][cut:]))
    
    if plot_pred:
        plt.plot(S_mean[:len(S_index)], label = 'Prediction')
        plt.fill_between(S_index, (S_mean-S_error)[:len(S_index)], (S_mean+S_error)[:len(S_index)], color = "blue", alpha = 0.3)
        plt.plot(index_real, trj[value][cut:],label = 'Real')
        plt.xlabel("t", fontsize="x-large")
        plt.ylabel("price", fontsize="x-large")
        plt.title('Mean Prediction by GBM')
        plt.tick_params(labelsize="x-large")
        plt.legend(loc="best", fontsize="x-large", ncol=2)
        plt.show()
    return S_index, S_mean, S_error, trj, index_real








