# -*- coding: utf-8 -*-
"""
"""


import pandas as pd
import numpy as np
#from pandas.tseries.offsets import *
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from statsmodels.tsa.arima.model import ARIMA
#import statsmodels.formula.api as smf
#import statsmodels.tsa.api as smt
#import statsmodels.api as sm #Maybe drop
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import decomposition
#from scipy.stats import multivariate_t 
#import math

#%%
##############################################################################
########################   Parameters   ######################################
##############################################################################

DaysPerYear = 250
MonthsPerYear = 12 #Will use trading days
#reserved = DaysPerYear*2 #Use last two years of data
#Similar to prior examples from Lecture 6, lets preserve the most recent 90
#days as a test
forecast_months = 24
nTrials = 10000 #Number of simulations

#%%
##############################################################################
#######################   Bring in Data   ####################################
##############################################################################
#Set Directory  -If python file not in same location as data, otherwise can skip
import os
os.chdir()
df_Price = pd.read_excel("HPI_PO_metro.xls",
                         sheet_name = 'Sheet1',
                         header=0)
#Potentially deal with outliers for April 2020 with WTI
#df_Price['year']=df_Price['Month'].dt.year
#df_Price['month']=df_Price['Month'].dt.month
#df_Price.drop(columns=['Month'], inplace=True)
time = df_Price['Time'].copy()
df_Price.drop(columns=['Time'], inplace=True)
#df_Factors = pd.read_excel("Data_Post.xlsx", sheet_name='Factor_Wts',engine='openpyxl')
#df_Factors.rename(columns={'Unnamed: 0':'Commodity'}, inplace=True)
#df_Factors.set_index(['Commodity'], inplace=True)

#Get Returns
df_Ret= pd.DataFrame()

for x in df_Price.columns:
    df_Ret[x] = np.log(df_Price[x]/df_Price[x].shift(1))
df_Ret.dropna(inplace=True)
ndates = len(df_Ret)
Cities = df_Ret.columns.tolist()

fig, ax = plt.subplots()
ax.plot(df_Price)
ax.set_xlabel('Time Step', weight='bold')
ax.set_ylabel('Home Price Index', weight='bold')
ax.set_title('Historical Home Prices across Metro Areas', weight='bold', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.scatter(df_Ret[10420], df_Ret[10580], label='10420 & 10580')
ax.set_xlabel('10580', weight='bold')
ax.set_ylabel('10420', weight='bold')
ax.set_title('Historical Home Prices Returns across Metro Areas', weight='bold', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.show()



#%%
##############################################################################
######################   Forecast Prices   ###################################
##############################################################################
from arch import arch_model

alpha = {}
beta_GARCH= {}
omega = {}
V_LR = {}
AR_1 = {}
mu = {}
stdzd_resid = {}
sorted_resid = {}
sigma={}


for y in Cities:
    model = arch_model(df_Ret[y]*100, mean='ARX', lags=[1], vol='GARCH', p=1, q=1, dist="StudentsT")
    model_fit = model.fit()
    #print(model_fit.summary())
    
    #Return Parameters
    AR_1[y] = model_fit.params[str(y)+'[1]']
    mu[y] = model_fit.params['Const']
    
    #Volatility Parameters
    alpha[y] = model_fit.params['alpha[1]']
    beta_GARCH[y] = model_fit.params['beta[1]']
    omega[y] = model_fit.params['omega']
    
    #Volatility Parameters
    sigma[y] = np.std(model_fit.resid)

    #Residuals with unit variance
    stdzd_resid[y] = (model_fit.resid)/np.std(model_fit.resid)
    stdzd_resid[y].dropna(inplace=True)
    
    #Sorted Resid
    sorted_resid[y] = stdzd_resid[y].sort_values().reset_index(drop=True)

#Correlation of Residuals
stdzd_resid = pd.concat(stdzd_resid, names=['City'])
stdzd_resid=stdzd_resid.unstack(level=['City'])

rho_Ret = stdzd_resid[Cities].corr() #Correlation matrix
cov_Ret = stdzd_resid[Cities].cov()
#Convert to copula parameter (See Schimdt 4.2)
#cop_Ret = np.sin(rho_Ret*np.pi/2)


#Sample from Residuals (so using Bootstrap with T Distribution)
#For t-distribution could use combination of following:
df= 2
Z = np.random.default_rng().multivariate_normal(np.zeros(len(Cities)),
                                                rho_Ret,
                                                size=(forecast_months,nTrials))
Chi = np.random.chisquare(df=df, size=(forecast_months,nTrials))

Scale = np.sqrt(df/Chi)
Scale = Scale[:,:,np.newaxis]
T = Z * Scale
T_cdf = t.cdf(T, df=df)

pred_resid_AR = {}
x=0
for i in Cities:
    idx = np.floor(T_cdf[:,:,x].squeeze()*len(sorted_resid[i])).astype(int) 
    pred_resid_AR[i] = sigma[y]*sorted_resid[i][idx.flatten()].values.reshape(idx.shape)
    x+=1

#Also need to simulate our volatilities
pred_vol_GARCHEst = {}
#Expected Vol Path at time t+i (fixing starting vol to time 0)
for y in Cities:
    pred_vol_GARCHEst[y] = np.zeros((forecast_months,nTrials))
    for i in range(forecast_months):
        if i==0:
            pred_vol_GARCHEst[y][0] = np.sqrt(omega[y] + beta_GARCH[y]*np.std(df_Ret[y])**2 +
                                              alpha[y]*df_Ret[y].iloc[-1].copy()**2)
        else:   
            pred_vol_GARCHEst[y][i] = np.sqrt(omega[y] + alpha[y]*pred_resid_AR[y][i,:]**2+beta_GARCH[y]*pred_vol_GARCHEst[y][i-1]**2)


#Now can simulate returns
pred_ret_AR = {}
for y in Cities:
    pred_ret_AR[y] = np.zeros((forecast_months,nTrials))
    #For first period use last training period returns
    pred_ret_AR[y][0,:] = (mu[y]+ AR_1[y]*df_Ret[y].iloc[-1]+
                                 pred_vol_GARCHEst[y][0,:]*pred_resid_AR[y][0,:])
    for i in range(1,forecast_months):
        pred_ret_AR[y][i,:] = (mu[y]+ AR_1[y]*pred_ret_AR[y][i-1,:]+
                                     pred_vol_GARCHEst[y][i]*pred_resid_AR[y][i,:])
    #Scale back returns
    pred_ret_AR[y]=pred_ret_AR[y]/100
    pred_ret_AR[y] = winsorize(pred_ret_AR[y], limits=0.01)

#Now can convert to prices
pred_price_AR = {}
for i in Cities:
    pred_price_AR[i] = df_Price[i].iloc[-1]*np.exp(pred_ret_AR[i].cumsum(axis=0))


#Some extreme prices, so might need to deal with outliers
for x in Cities:
    pred_price_AR[x] = winsorize(pred_price_AR[x], limits=0.01)

#Plot distribution
for i in Cities[:20]:
    num_bins = 20
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(pred_price_AR[i][-1,:], num_bins, density=True)
    plt.axvline(x=df_Price[i].iloc[-1], color='red')
    ax.set_xlabel('Ending Price '+str(i))
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of Simulated '+str(i)+' Prices')
    plt.show()

#%%
##############################################################################
################   Check likelihood of Defaults   ############################
##############################################################################

pred_price_AR = pd.DataFrame({k:x.ravel() for k,x in pred_price_AR.items()},
                  index=pd.MultiIndex.from_product([np.arange(24), np.arange(10000)], names=['Month', 'Sim']))

pred_ret_AR = pd.DataFrame({k:x.ravel() for k,x in pred_ret_AR.items()},
                           index=pd.MultiIndex.from_product([np.arange(24), np.arange(10000)], names=['Month', 'Sim']))

portfolio_start = df_Price.iloc[-1].mean()
portfolio_ret = pred_ret_AR.mean(axis=1)
portfolio_price = pred_price_AR.mean(axis=1)
portfolio_ret=portfolio_ret.unstack(level=['Sim'])
portfolio_cum_ret = np.exp(portfolio_ret.cumsum(axis=0))
portfolio_price = portfolio_price.unstack(level=['Sim'])
#portfolio_change=portfolio_price/portfolio_start
num_defaults = np.sum(np.min(portfolio_cum_ret, axis=0)<0.88)

#For a AAA rated bond the average cumulative default rate over the next 
#5-years is 0.085.
print('Percentage of sims with default:'+str(num_defaults/nTrials))

#Histogram, actual 2011 Q4 was a 0.791 from 2006 Q4
fig, ax = plt.subplots()
n, bins, patches = ax.hist(portfolio_cum_ret.iloc[-1], bins=40, density=True)
plt.axvline(x=0.791, color='red')
ax.set_xlabel('Ending Cumulative Ret')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of Simulated  Changes')
plt.show()
    