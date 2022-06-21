# -*- coding: utf-8 -*-
"""


Copula Example
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
# import random as rand
# from statistics import mean 
# from itertools import islice
import pandas as pd


#%% 
#################################################################
#################       Modules       ###########################
#################################################################




#%% 
#################################################################
###################       Data      #############################
#################################################################

#First will read in stock price data, calculate the daily returns, and std.
#dev. of the returns
#Set Directory  -If python file not in same location as data, otherwise can skip
import os
os.chdir("/Users/taihanrui/Downloads")

#Read in stock prices file
raw_data = {}
ndays = {}
tickers = ['XOM','SPY']
#Bring in Data via .csv
for i in tickers:
    file = i+'.csv'
    raw_data[i] = pd.read_csv(file, sep=';')
    raw_data[i]['Ticker'] = i
    raw_data[i]['Date'] = pd.to_datetime(raw_data['XOM']["Date"])
    raw_data[i]['Month'] = raw_data[i]["Date"].dt.month
    raw_data[i]['Year'] = raw_data[i]["Date"].dt.year
    raw_data[i]['EOM'] = raw_data[i]["Month"]!=raw_data[i]["Month"].shift(-1)
    raw_data[i]['ret'] = np.log(raw_data[i].loc[:,"Adj Close"]/
                                raw_data[i].loc[:,"Adj Close"].shift(1))
    #Drop Missing Returns
    raw_data[i].dropna(axis=0,inplace=True)
    ndays[i] = len(raw_data[i])

Monthly_Data = {}
nmonths = {}
for i in tickers:
    Monthly_Data[i] = raw_data[i].loc[raw_data[i]['EOM']==True].copy()
    #Generate Monthly returns
    Monthly_Data[i]['ret'] = np.log(Monthly_Data[i].loc[:,"Adj Close"]/
                                    Monthly_Data[i].loc[:,"Adj Close"].shift(1))
    #Drop Missing Returns
    Monthly_Data[i].dropna(axis=0,inplace=True)
    #Number of dates
    nmonths[i] = len(Monthly_Data[i])


#%% 
#################################################################
##############     Return Simulation    #########################
#################################################################

#Sort Rekt)
sorted_rets = {}
for i in tickers:
    sorted_rets[i] = raw_data[i].sort_values(by=['ret'],inplace=False).reset_index(drop=True)

#Pull returns from historical distribution
test_days = 30; #Sample length 
nTrials=10000; #Number of Monte Carlo Trials
Z = np.random.randn(test_days,nTrials) #Generate random variables from standard normal
U = norm.cdf(Z) #Convert z values to probabilities
idx = np.floor(ndays['XOM']*U)
#From probabilities U, grab the corresponding value from the sorted returns
XOM_Sim = sorted_rets['XOM']['ret'][idx.flatten()].values.reshape(idx.shape)

#From Returns generate prices
XOM_Sim_prices = np.ones((test_days+1,nTrials))*raw_data['XOM']['Adj Close'].tail(1).squeeze()
for i in range(1,test_days+1):
    XOM_Sim_prices[i,:] = XOM_Sim_prices[i-1,:]*np.exp(XOM_Sim[i-1,:])


#Plot
fig, ax = plt.subplots()
ax.hist(XOM_Sim_prices[-1], bins=20, linewidth=0.5, edgecolor="white")
ax.axvline(np.mean(XOM_Sim_prices[-1]), color='b', linestyle='dashed',
           label = "Mean", linewidth=1.5)
ax.axvline(np.percentile(XOM_Sim_prices[-1],5), color='g', linestyle='dashed',
           label = "5th Percentile",linewidth=1.5)
ax.axvline(np.percentile(XOM_Sim_prices[-1],95), color='r', linestyle='dashed',
           label = "95th Percentile", linewidth=1.5)
ax.set_xlabel('End Price', weight='bold')
ax.set_ylabel('Frequency', weight='bold')
ax.set_title('Simulated Ending Price', weight='bold', fontsize=12)
plt.show()

#Probability > 75?
print("Probability ending price is more than 75: "+str(np.sum(XOM_Sim_prices[-1]>75)/nTrials))

#%% 
#################################################################
#################     Vol Estimation    #########################
#################################################################
#Note on testing:
    #Want to generate a test set of data and training set of data
    #Will use last 90 days as test set and all previous data as our training set

#First plot Rolling 90-day vols - will use rolling 90-day vol as our
#baseline
vol_windows = 90
no_of_months = raw_data['XOM'].EOM.sum()
vol_XOM_90 = np.zeros((len(raw_data['XOM'])-vol_windows,2))
for i in range(len(raw_data['XOM'])-vol_windows):
    vol_XOM_90[i,0]=raw_data['XOM']['ret'].iloc[i:i+vol_windows].std();
    vol_XOM_90[i,1] = i

#Plot
fig, ax = plt.subplots()
ax.plot(vol_XOM_90[:,0], label='Rolling Vol')
ax.set_xlabel('Days Since 20', weight='bold')
ax.set_ylabel('Rolling Vol', weight='bold')
ax.set_title('Rolling 90-Day Volatility', weight='bold', fontsize=12)
plt.show()

#LR Equal Weighted Average (similar if assume 0 mean return)
vol_XOM = raw_data['XOM']['ret'].std()
vol_XOM_u = np.sqrt((1/len(raw_data['XOM']['ret']))*np.sum(raw_data['XOM']['ret']**2)) #assume 0 mean

#%% 
#################################################################
#################     Vol Prediction    #########################
#################################################################

#%%
##################
### LR Average ###
##################
#Now suppose I want to predict daily volatility over the next 90 days,
#starting from day 0. To do test this I will reserve the last 90 days of
#the sample as my testing sample.
test_days = 90;
#NOTE: end-test_days+1 is the end point of our training data, in general don't
#want to use all the data as information from long ago might not be
#relevant. Given that I am using daily data, can use 1-2 years with to get
#pretty good estimates.
#The training data is then:
training_vol = vol_XOM_90[0:-test_days,0]
training_ret = raw_data['XOM']['ret'][0:-test_days].values

#Example, use last two-year daily with equal weights, assuming constant vol
pred_vol_XOM_EW = np.zeros(test_days)
actual_vol_XOM = np.zeros(test_days)
for i in range(test_days):
    #Assume constant vol, using data up to prediction date
    pred_vol_XOM_EW[i]=np.mean(training_vol)

actual_vol_XOM = vol_XOM_90[-test_days:]

#Plot
fig, ax = plt.subplots()
ax.plot(actual_vol_XOM[:,0], label = 'Actual Vol', color='b')
ax.plot(pred_vol_XOM_EW, label = 'Predicted Vol', color='g')
ax.set_xlabel('Time', weight='bold')
ax.set_ylabel('Rolling Vol', weight='bold')
ax.set_title('Predicted vs. Actual 90-Day Volatility (EW)', weight='bold', fontsize=12)
plt.show()

#%%
####################################
###  90-day Avg and Weighted Avg ###
####################################
#Example, use last 90 days with EW or increasing weights (linear increase)
wts = np.zeros((vol_windows,1))
increment=2/(vol_windows*(1+vol_windows)) #Linear increment
wts = [increment*(i+1) for i in range(vol_windows)] #Wts for calculation
pred_vol_XOM_lin = np.zeros(test_days)
for i in range(test_days):
    pred_vol_XOM_lin[i] = np.sqrt(np.sum(wts*(training_ret[-vol_windows:]**2)))

#Plot Rolling 90-Day Vol with increasing weights
fig, ax = plt.subplots()
ax.plot(actual_vol_XOM[:,0], label = 'Actual Vol', color='b')
ax.plot(pred_vol_XOM_EW, label = 'Predicted EW Vol', color='g')
ax.plot(pred_vol_XOM_lin, label = 'Predicted Wtd Vol', color='r')
ax.set_xlabel('Time', weight='bold')
ax.set_ylabel('Rolling Vol', weight='bold')
ax.set_title('Predicted vs. Actual 90-Day Volatility (EW)', weight='bold', fontsize=12)
ax.legend()
plt.show()

#Compare sum of squared estimate of errors
diffs_EW = np.sum((actual_vol_XOM[:,0] - pred_vol_XOM_EW)**2)
diffs_lin = np.sum((actual_vol_XOM[:,0] - pred_vol_XOM_lin)**2)

#%%
#####################################
### MA Model (Combo of LR and SR) ###
#####################################
#NOTE: Our vol estimate so far is NOT Time Varying, to do this we need to
#introduce more complexity. Use combination of LR vol and SR vol (e.g., vol over
#last 90 days)
gamma = .5 #gamma is weight on LR vs. SR
increment=2*(1-gamma)/(vol_windows*(1+vol_windows)) #Linear increment
wts = [increment*(i+1) for i in range(vol_windows)] #Wts for calculation

#Example, use last 90 days with LR average and increasing weights (linear increase)
pred_vol_XOM_MA = np.zeros(test_days)
moving_vol = training_ret[-vol_windows:].copy()
for i in range(test_days):
    if i>0:
        moving_vol[0:-i] = training_ret[-vol_windows+i:].copy() #Update moving vol recursively
        moving_vol[-i:] = pred_vol_XOM_MA[0:i]
    pred_vol_XOM_MA[i] = np.sqrt(gamma*np.mean(training_vol)**2 + np.sum(wts*(moving_vol**2)))

#Plot Rolling 90-Day Vol with prior estimates and MA
fig, ax = plt.subplots()
ax.plot(actual_vol_XOM[:,0], label = 'Actual Vol', color='b')
ax.plot(pred_vol_XOM_EW, label = 'Predicted EW Vol', color='g')
ax.plot(pred_vol_XOM_lin, label = 'Predicted Wtd Vol', color='r')
ax.plot(pred_vol_XOM_MA, label = 'Predicted MA Vol', color='y')
ax.set_xlabel('Time', weight='bold')
ax.set_ylabel('Rolling Vol', weight='bold')
ax.set_title('Predicted vs. Actual 90-Day Volatility (EW)', weight='bold', fontsize=12)
ax.legend()
plt.show()

#Compare sum of squared estimate of errors
diffs_MA = np.sum((actual_vol_XOM[:,0] - pred_vol_XOM_MA)**2)
#Can also use an Exponential Weighted Moving Average (EWMA)
#where weights decrease exponentially, rather than linearly.
#See textbook chapter 10.6, but can show the current estimate.


#%%
#######################################################
### GARCH (Combo of LR and SR and daily innovation) ###
#######################################################
#The “(1,1)”in GARCH(1,1) indicates that sigma^2 is based on the most recent 
#observation of u^2 and the most recent estimate of the variance rate. 
#The more general GARCH(p,q) model calculates sigma^2 from the most recent p 
#observations on u^2 and the most recent q estimates of the variance rate.
#GARCH(1,1) is by far the most popular of the GARCH models.

#Note estimate directly via built in GARCH

#Will be using arch_model package, might need to install if not already done
#conda install -c conda-forge arch-py
from arch import arch_model
from arch.__future__ import reindexing
model = arch_model(training_ret*100, mean='Zero', vol='GARCH', p=1, q=1)
model_fit = model.fit()

alpha = model_fit.params['alpha[1]']
beta = model_fit.params['beta[1]']
omega = model_fit.params['omega']/10000
V_LR = omega/(1-alpha-beta)
estimated_annualized_LR_vol = np.sqrt(V_LR)*np.sqrt(252)

#Predicted Values Given Estimate (updating with new returns)
yhat = model_fit.forecast(horizon=test_days)
pred_vol_XOM_GARCH = np.sqrt(yhat.variance.values[-1])


pred_vol_XOM_GARCHEst = np.zeros(test_days)
#Expected Vol Path at time t+i (fixing starting vol to time 0)
for i in range(test_days):
    if i==0:
        pred_vol_XOM_GARCHEst[0] = np.sqrt(omega + beta*np.std(training_ret[-vol_windows:])**2 +
                                           alpha*raw_data['XOM']['ret'].iloc[-test_days].copy()**2)
    pred_vol_XOM_GARCHEst[i] = np.sqrt(V_LR + ((alpha+beta)**i)*(pred_vol_XOM_GARCHEst[0]**2-V_LR))

#Plot Rolling 90-Day Vol with prior estimates and Garch
fig, ax = plt.subplots()
ax.plot(actual_vol_XOM[:,0], label = 'Actual Vol', color='b')
ax.plot(pred_vol_XOM_EW, label = 'Predicted EW Vol', color='g')
ax.plot(pred_vol_XOM_lin, label = 'Predicted Wtd Vol', color='r')
ax.plot(pred_vol_XOM_MA, label = 'Predicted MA Vol', color='y')
ax.plot(pred_vol_XOM_GARCHEst, label = 'Predicted Garch Vol', color='black')
ax.set_xlabel('Time', weight='bold')
ax.set_ylabel('Rolling Vol', weight='bold')
ax.set_title('Predicted vs. Actual 90-Day Volatility (EW)', weight='bold', fontsize=12)
ax.legend()
plt.show()

diffs_GARCH = np.sum((actual_vol_XOM[:,0] - pred_vol_XOM_GARCHEst)**2)
#Plot RMSE (low is good)
ind=range(4)
fig, ax = plt.subplots()
ax.bar(ind[0], diffs_EW, label = 'Predicted EW Vol', color='g')
ax.bar(ind[1], diffs_lin, label = 'Predicted Wtd Vol', color='r')
ax.bar(ind[2], diffs_MA, label = 'Predicted MA Vol', color='y')
ax.bar(ind[3], diffs_GARCH, label = 'Predicted Garch Vol', color='black')
ax.set_ylabel('RMSE', weight='bold')
ax.set_title('Comparison of RMSEs', weight='bold', fontsize=12)
ax.set_xticks(ind)
ax.set_xticklabels(['EW', 'Wtd', 'MA', 'Garch'])
plt.show()

#%% Simulating Returns
########################################################################
### Predicted Returns using volatility estimates and expected return ###
########################################################################

#Note we could do this similar to before and just sample from past returns
#or apply estimates of expected return and constant volatility, with some 
#distribution of error terms (e.g., Normal or Student-t)
test_days = 30
#If doing with log returns can use 
XOM_expected_ret=np.mean(raw_data['XOM']['ret'])
NoiseTerms = np.random.randn(test_days,nTrials) #Using Normal Distribution
SimRets = ((XOM_expected_ret-0.5*np.std(raw_data['XOM']['ret'])**2)+
           np.std(raw_data['XOM']['ret'])*NoiseTerms)
#From Returns generate prices
XOM_Sim_prices = np.ones((test_days+1,nTrials))*raw_data['XOM']['Adj Close'].tail(1).squeeze()
for i in range(1,test_days+1):
    XOM_Sim_prices[i,:] = XOM_Sim_prices[i-1,:]*np.exp(SimRets[i-1,:])
EndPrices_ConstantVol = XOM_Sim_prices[-1,:]
ProbGreater75 = (EndPrices_ConstantVol>75)
ProbGreater75=np.mean(ProbGreater75)
print('Probability Greater than 75 in 30-days: '+str(ProbGreater75))

#Histogram of Ending Prices #Get couple of outliers, can exclude by setting chart max
fig, ax = plt.subplots()
ax.hist(EndPrices_ConstantVol, bins=30, linewidth=0.5, edgecolor="white")
ax.set_xlim(right=150)
plt.show()

#Plot Price paths:
fig, ax = plt.subplots(nrows=1)
for i in range(0,5):
    ax.plot(XOM_Sim_prices[:,i], label='Sim '+str(i))
ax.set_xlabel('Time Step', weight='bold')
ax.set_ylabel('Stock Price', weight='bold')
ax.set_title('Simulated Stock Prices for 5 Simulations', weight='bold', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.show()

###Can also call back to our estimates from the GARCH Model
model = arch_model(raw_data['XOM']['ret']*100, mean='ARX', lags=[1], vol='GARCH', p=1, q=1, dist="StudentsT")
model_fit = model.fit()
model_fit.summary()
yhat = model_fit.forecast(horizon=test_days,method='simulation', simulations=10000)
SimRets_GARCH = yhat.simulations.values[-1].T #Output is periods in rows, simulations in columns
SimRets_GARCH=SimRets_GARCH/100 #Scale back to returns

#From Returns generate prices
XOM_Sim_prices = np.ones((test_days+1,nTrials))*raw_data['XOM']['Adj Close'].tail(1).squeeze()
for i in range(1,test_days+1):
    XOM_Sim_prices[i,:] = XOM_Sim_prices[i-1,:]*np.exp(SimRets_GARCH[i-1,:])
EndPrices_Garch = XOM_Sim_prices[-1,:]
ProbGreater75 = (EndPrices_Garch>75)
ProbGreater75=np.mean(ProbGreater75)
print('Probability Greater than 75 in 30-days: '+str(ProbGreater75))

#Histogram of Ending Prices #Get couple of outliers, can winsorize a 99 percentile
pctile_99 = np.percentile(EndPrices_Garch,99)
#EndPrices_Garch = np.where(EndPrices_Garch > pctile_99, pctile_99, EndPrices_Garch)
fig, ax = plt.subplots()
ax.hist(EndPrices_Garch, bins=30, linewidth=0.5, edgecolor="white")
plt.show()

#Plot Price paths:
fig, ax = plt.subplots(nrows=1)
for i in range(0,5):
    ax.plot(XOM_Sim_prices[:,i], label='Sim '+str(i))
ax.set_xlabel('Time Step', weight='bold')
ax.set_ylabel('Stock Price', weight='bold')
ax.set_title('Simulated Stock Prices for 5 Simulations', weight='bold', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.show()

