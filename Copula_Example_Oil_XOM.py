# -*- coding: utf-8 -*-
"""

The following code works through several examples of forecasting an underlying
price series. For the example we use a back testing routine, were we use the
last 90 days to test our forecasts, and the previous 2-years of daily data to
form a training set.

The goal is to generate a set of forecasted returns and prices for XOM. We 
will assume that XOM is driven by an idiosynratic component as well as the 
movements in Crude prices as well as the S&P 500 (a stand in for the market).

To start we need a model of the return generating process. We can start by
assuming that XOM has a linear factor structure with the S&P 500 and Crude.
Therefore, we can estimate a risk exposure of XOM to the S&P 500 and Crude, 
generate a forecast of S&P 500 and Crude returns, and then use these to 
generate a forecast of XOM returns. 

There are several methods to forecast S&P 500 and Crude returns. We first
need a model of their expected returns. This could come from using past returns
or using market expectations (e.g., from futures). We then need an estimate of
volatility. This could come from past return volatility (e.g., constant, GARCH,
etc.), or again from market expectations (e.g., implied volatility). Once we
have these we can generate forecasted returns for S&P and Crude, and then use
these plus an idiosynratic term (i.e., using the volatility of XOM), to forecast
XOM.
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
#from scipy.stats import multivariate_t 
#import math

#%%
##############################################################################
########################   Parameters   ######################################
##############################################################################

DaysPerYear = 252 #Will use trading days
reserved = DaysPerYear*2 #Use last two years of data
#Similar to prior examples from Lecture 6, lets preserve the most recent 90
#days as a test
test_days = 90 #Number of days to test
nTrials = 10000 #Number of simulations

#%%
##############################################################################
#######################   Bring in Data   ####################################
##############################################################################
#Set Directory  -If python file not in same location as data, otherwise can skip
import os
os.chdir("/Users/taihanrui/Downloads")
df_Price = pd.read_excel("Prices.xlsx", sheet_name='Prices',engine='openpyxl')

#Get Returns
df_Ret= pd.DataFrame()

for x in ['XOM', 'SPX', 'Crude']:
    df_Ret[x] = np.log(df_Price[x]/df_Price[x].shift(1))

df_Ret['Tbills'] = df_Price['Tbills'] / 100
df_Ret['Tbills'] = (1+df_Ret['Tbills'])**(1/DaysPerYear)-1 #Convert to daily

df_Ret['Date'] = df_Price['Dates']

df_Ret.dropna(inplace=True)
ndates = len(df_Ret)



#Potentially deal with outliers for April 2020 with WTI
df_Ret['year']=df_Ret['Date'].dt.year
df_Ret['month']=df_Ret['Date'].dt.month

lb_Crude = df_Ret['Crude'].quantile(0.01)
ub_Crude = df_Ret['Crude'].quantile(0.99)

#Easiest way is to winsorize
df_Ret['Crude'] = winsorize(df_Ret['Crude'], limits=0.01)





#Lets first plot the returns to get an idea of what they look like, will
#just plot last 504 (2-years) days

fig, ax = plt.subplots()
fig.suptitle('Daily Returns of XOM, SPX, and WTI')

ax.plot(df_Ret['XOM'].iloc[-reserved:], label='XOM')
ax.plot(df_Ret['SPX'].iloc[-reserved:], label='SPX')
ax.plot(df_Ret['Crude'].iloc[-reserved:], label='WTI')
ax.set_ylabel('Return')
legend = ax.legend(loc='lower left', shadow=True, fontsize='x-large')
plt.show()

#Scatter of WTI and SPX
fig, ax = plt.subplots()
fig.suptitle('SPX vs WTI')
ax.scatter(df_Ret['SPX'].iloc[-reserved:],df_Ret['Crude'].iloc[-reserved:])
ax.set_xlabel('SPX Return')
ax.set_ylabel('WTI Return')
plt.show()

#Correlations over last 2-years
corr_ret = df_Ret[['XOM', 'SPX', 'Crude']].iloc[-reserved:].corr()

#%%
##############################################################################
######################   Risk Exposures   ####################################
##############################################################################
#First lets estimate a risk-exposure of XOM on Crude. Note we will
#be using excess returns so want the return less the rf.

df_ExRet = pd.DataFrame()
df_training = pd.DataFrame()
for x in ['XOM', 'SPX', 'Crude']:
    df_ExRet[x] = df_Ret[x] - df_Ret['Tbills']
    df_training[x] = df_ExRet[x].iloc[-(reserved+test_days):-test_days]

df_training['Tbills'] = df_Ret['Tbills'].iloc[-(reserved+test_days):-test_days]
#%First go around, lets just use return on oil as our risk factor
result = sm.ols(formula="XOM ~ Crude + SPX", data=df_training).fit()
beta = {}
beta['Crude'] = result.params['Crude']
beta['SPX'] = result.params['SPX']

#Now we can simulate movements in XOM based on movements in Oil, so if want
#to forecast price movements of XOM on price movements of Oil, we can
#simulate oil returns, simulate the noise in XOM after taking out oil
#returns, and then get an estimate of prices.

#First get residuals from regression of XOM on Oil and SPX:
XOM_resid = result.resid
fig, ax = plt.subplots()
ax.plot(XOM_resid)
ax.set_xlabel('Time');
ax.set_ylabel('Residuals');
fig.suptitle('Residuals of XOM Returns, after Oil and SPX');
plt.show()

#Risk Free Rate Estimate
rf_const = df_training['Tbills'].mean()

#%% 
##############################################################################
####################   Simulating Returns   ##################################
##############################################################################

#We now have several options on how to proceed. 

#1) Use a parametric model to forecast returns of S&P and Crude. This could 
#include doing the following:
    #a) Assume a GBM and independent returns
    #b) Assume a GBM and correlated returns
    #c) Estimate a AR/GARCH model of returns and/or residuals with or without
    #correlations.
#2) Non-parametrically sample (bootstrap) returns of S&P and Crude with or 
#without correlations.


#Simple: GBM with constant volatility and AR(1).

#For simplification, lets just assume that there is constant vol, and crude
#returns follow a GM Brownian motion (similar to Black-Scholes). 
#We could also bootstrap returns for Crude from the historical data by 
#randomly sampling (similar to Lecture 6). Or estimate an AR(1) model and 
#then generate noise by sampling from the normal distribution.

#Estimate AR(1,0,1) and Volatility for Crude:
#Vol estimate for crude and XOM
vol_est = {}
vol_est['Crude'] = df_training['Crude'].std() #Assuming constant vols
vol_est['SPX'] = df_training['SPX'].std() #Assuming constant vols
vol_est['XOM'] = XOM_resid.std() #Assuming constant vols

# fit model for Crude
model = ARIMA(df_training['Crude'], order=(1,0,0))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
rho = {}
mu = {}

rho['Crude'] = model_fit.params['ar.L1'] #rho for crude
mu['Crude'] = model_fit.params['const'] #mu for Crude

# fit model for SPX
model = ARIMA(df_training['SPX'], order=(1,0,0))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())

rho['SPX'] = model_fit.params['ar.L1'] #rho for SPX
mu['SPX'] =  model_fit.params['const'] #mu for SPX

#First predict for Crude
predict_excess_ret = {}
predict_excess_ret['Crude'] = np.zeros((test_days,nTrials))
predict_excess_ret['SPX'] = np.zeros((test_days,nTrials))

#For simplicity will run as loop, could also broadcast directly
for y in ['Crude', 'SPX']:
    #For first period use last training period returns
    predict_excess_ret[y][0,:] = (mu[y]+ 
                                  rho[y]*df_ExRet[y].iloc[-test_days]+
                                  vol_est[y]*np.random.randn(1,nTrials))
                                            #using standard normal
    #For remaining period use previous periods forecasted return
    for i in range(1,test_days):
        predict_excess_ret[y][i,:] = (mu[y]+ 
                                      rho[y]*predict_excess_ret[y][i-1,:] +
                                      vol_est[y]*np.random.randn(1,nTrials))

#predicted XOM returns are the rf rate + beta*(Crude_excess) + noise
predict_XOM_ret = (rf_const + beta['Crude']*predict_excess_ret['Crude'] + 
                   beta['SPX']*predict_excess_ret['SPX'] +
                   vol_est['XOM']*np.random.randn(test_days,nTrials))
predict_XOM_price = df_Price['XOM'].iloc[-test_days]*np.exp(predict_XOM_ret.cumsum(axis=0))

#Plot distribution
num_bins = 20
fig, ax = plt.subplots()
n, bins, patches = ax.hist(predict_XOM_price[-1,:], num_bins, density=True)
plt.axvline(x=df_Price['XOM'].iloc[-1], color='red')
ax.set_xlabel('Ending Price XOM')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of Simulated XOM Prices')
plt.show()


#Probability End price>75 -- Note different then in lecture 6, as prices are
#as of 90 days ago. Would instead want to rerun given starting price today. For 
#now will just assume same returns as above and just estimate off of a 
#different starting price.
predict_XOM_price = df_Price['XOM'].iloc[-1]*np.exp(predict_XOM_ret.cumsum(axis=0))


pred_XOM_end_price=predict_XOM_price[-1,:]
ProbGreater75 = (pred_XOM_end_price>75)
ProbGreater75=np.mean(ProbGreater75)
print ("Probability Greater than 75 in 90-days: {}".format(ProbGreater75))


#%%
##############################################################################
###############   Dealing with Correlated Series   ###########################
##############################################################################

#wE expect that Crude and SPX affect the returns of XOM and therefore we have
#modeled both. However, as we saw above these two returns are
#correlated. Let see what happens if because we didn't take this into account.

#We can start with our assumptions above:
#Plot results
num_bins = 20
fig, ax = plt.subplots()
ax.hist((predict_excess_ret['SPX'].flatten(), df_ExRet['SPX'].iloc[-test_days:].values),
        num_bins, density=True,
        label=['Predicted','Actual'])
ax.set_xlabel('Daily Returns')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of Actual vs Simulated SPX Returns')
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.show()

fig, ax = plt.subplots()
ax.hist((predict_excess_ret['Crude'].flatten(), df_ExRet['Crude'].iloc[-test_days:].values),
        num_bins, density=True,
        label=['Predicted','Actual'])
ax.set_xlabel('Daily Returns')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of Actual vs Simulated WTI Returns')
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.show()

#Scatter Plot
fig, (ax0, ax1) = plt.subplots(2,1)
fig.suptitle('SPX vs WTI')
ax0.scatter(df_ExRet['SPX'].iloc[-test_days:],df_ExRet['Crude'].iloc[-test_days:])
ax0.set_xlabel('SPX Actual Return')
ax0.set_ylabel('WTI Actual Return')
ax1.scatter(predict_excess_ret['SPX'].flatten('F'),predict_excess_ret['Crude'].flatten('F'))
ax1.set_xlabel('SPX Pred Return')
ax1.set_ylabel('WTI Pred Return')
plt.show()
#Note lack of correlation


#Can also show this if we had used a parametric distribution
#%Will examine it via backtesting over last 2-years (504) prices and returns.

#################################################################
##############   Bootsrapping Returns   #########################
#################################################################
#Now lets bootstrap just using random variables, rather than assuming a distribution 
#note for bootstrap will need sorted returns
sorted_rets = {}
sorted_rets['XOM'] = df_training['XOM'].sort_values().reset_index(drop=True)
sorted_rets['SPX'] = df_training['SPX'].sort_values().reset_index(drop=True)
sorted_rets['Crude'] = df_training['Crude'].sort_values().reset_index(drop=True)

#Next need two sets of random variables
pred_ret_boot = {}
#one for crude
Z1=np.random.normal(0,1,(test_days,nTrials)) #Standard normal val
U1 = norm.cdf(Z1) #Convert into prob
idx = np.floor(U1*len(sorted_rets['Crude'])).astype(int) 
#Grab return corresponding to that percentile
pred_ret_boot['Crude'] = sorted_rets['Crude'][idx.flatten()].values.reshape(idx.shape)

#one for SPX
Z2=np.random.normal(0,1,(test_days,nTrials)) #Standard normal val
U2 = norm.cdf(Z2) #Convert into prob
idx = np.floor(U2*len(sorted_rets['SPX'])).astype(int) 
#Grab return corresponding to that percentile
pred_ret_boot['SPX'] = sorted_rets['SPX'][idx.flatten()].values.reshape(idx.shape)

#Plot results
num_bins = 20
fig, ax = plt.subplots()
ax.hist((pred_ret_boot['SPX'].flatten('F'), df_ExRet['SPX'].iloc[-test_days:].values),
        num_bins, density=True,
        label=['Predicted','Actual'])
ax.set_xlabel('Daily Returns')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of Actual vs Simulated SPX Returns')
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.show()

fig, ax = plt.subplots()
ax.hist((pred_ret_boot['Crude'].flatten('F'), df_ExRet['Crude'].iloc[-test_days:].values),
        num_bins, density=True,
        label=['Predicted','Actual'])
ax.set_xlabel('Daily Returns')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of Actual vs Simulated WTI Returns')
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.show()

#Scatter Plot
fig, (ax0, ax1) = plt.subplots(2,1)
fig.suptitle('SPX vs WTI')
ax0.scatter(df_ExRet['SPX'].iloc[-test_days:],df_ExRet['Crude'].iloc[-test_days:])
ax0.set_xlabel('SPX Actual Return')
ax0.set_ylabel('WTI Actual Return')
ax1.scatter(pred_ret_boot['SPX'][:,0:5].flatten('F'),pred_ret_boot['Crude'][:,0:5].flatten('F'))
ax1.set_xlabel('SPX Pred Return')
ax1.set_ylabel('WTI Pred Return')
plt.show()

#Test Correlation
kendalltau(pred_ret_boot['SPX'].flatten(),pred_ret_boot['Crude'].flatten())[0]
#Still no correlation

#################################################################
################   Bootstrapping w/ Copula  #####################
#################################################################
#Will use same Z1, Z2, but generate a Bivariate Normal

rho_SPX_Crude = df_training[['SPX','Crude']].corr(method='kendall').iloc[1,0]
#Convert to copula parameter
cop_SPX_Crude = np.sin(rho_SPX_Crude*np.pi/2)

U1 = Z1
U2 = Z1*cop_SPX_Crude + Z2*np.sqrt(1-cop_SPX_Crude**2)

pred_ret_cop = {}
U1 = norm.cdf(U1) #Convert into prob
idx = np.floor(U1*len(sorted_rets['Crude'])).astype(int) 
pred_ret_cop['Crude'] = np.zeros((test_days,nTrials))
pred_ret_cop['Crude'] = sorted_rets['Crude'][idx.flatten()].values.reshape(idx.shape)

#one for SPX
U2 = norm.cdf(U2) #Convert into prob
idx = np.floor(U2*len(sorted_rets['Crude'])).astype(int) 
pred_ret_cop['SPX'] = np.zeros((test_days,nTrials))
pred_ret_cop['SPX'] = sorted_rets['SPX'][idx.flatten()].values.reshape(idx.shape)

#Plot results
num_bins = 20
fig, ax = plt.subplots()
ax.hist((pred_ret_cop['SPX'].flatten('F'), df_ExRet['SPX'].iloc[-test_days:].values),
        num_bins, density=True,
        label=['Predicted','Actual'])
ax.set_xlabel('Daily Returns')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of Actual vs Simulated SPX Returns')
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.show()

fig, ax = plt.subplots()
ax.hist((pred_ret_cop['Crude'].flatten('F'), df_ExRet['Crude'].iloc[-test_days:].values),
        num_bins, density=True,
        label=['Predicted','Actual'])
ax.set_xlabel('Daily Returns')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of Actual vs Simulated WTI Returns')
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.show()

#Scatter Plot
fig, (ax0, ax1) = plt.subplots(2,1)
fig.suptitle('SPX vs WTI')
ax0.scatter(df_ExRet['SPX'].iloc[-test_days:],df_ExRet['Crude'].iloc[-test_days:])
ax0.set_xlabel('SPX Actual Return')
ax0.set_ylabel('WTI Actual Return')
ax1.scatter(pred_ret_cop['SPX'][:,0:5].flatten('F'),pred_ret_cop['Crude'][:,0:5].flatten('F'))
ax1.set_xlabel('SPX Pred Return')
ax1.set_ylabel('WTI Pred Return')
plt.show()

#Test Correlation
kendalltau(pred_ret_cop['SPX'].flatten('F'),pred_ret_cop['Crude'].flatten('F'))[0]
kendalltau(df_ExRet['SPX'].iloc[-test_days:],df_ExRet['Crude'].iloc[-test_days:])[0]

#Some positive correlation, note does not exactly match but this is partially 
#due to underlying return series in testing vs. actual data

#More generally can use following for multivariate normal, or other
#distributions (e.g., t-distribution)
rho_SPX_Crude = df_training[['SPX','Crude']].corr(method='kendall') #Var-CoVar matrix
#Convert to copula parameter (See Schimdt 4.2)
cop_SPX_Crude = np.sin(rho_SPX_Crude*np.pi/2)
#Scale such that it has unit variance
Z = np.random.default_rng().multivariate_normal([0, 0],
                                                cop_SPX_Crude,
                                                size=(test_days,nTrials))
U = norm.cdf(Z) #Convert random variables to probabilities

#For t-distribution could use combination of following:
df= 10
Z = np.random.default_rng().multivariate_normal([0, 0],
                                                cop_SPX_Crude,
                                                size=(test_days,nTrials))
Chi = np.random.chisquare(df=df, size=(test_days,nTrials))

T1 = Z[:,:,0] * np.sqrt(df/Chi) #See Hull FRM 11.5
T2 = Z[:,:,1] * np.sqrt(df/Chi)

#Normal vs. T
fig, ax = plt.subplots()
ax.hist((T1.flatten('F'), Z[:,:,0].flatten('F')),
        num_bins, density=True,
        label=['T','Z'])
ax.set_xlabel('RV')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of T vs Z')
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.show()

T1_cdf = t.cdf(T1, df=df)
T2_cdf = t.cdf(T2, df=df)

pred_ret_copgen = {}
idx = np.floor(T1_cdf*len(sorted_rets['SPX'])).astype(int) 
pred_ret_copgen['SPX'] = sorted_rets['SPX'][idx.flatten()].values.reshape(idx.shape)

idx = np.floor(T2_cdf*len(sorted_rets['Crude'])).astype(int) 
pred_ret_copgen['Crude'] = sorted_rets['Crude'][idx.flatten()].values.reshape(idx.shape)
    
#Scatter Plot
fig, (ax0, ax1) = plt.subplots(2,1)
fig.suptitle('SPX vs WTI')
ax0.scatter(df_ExRet['SPX'].iloc[-test_days:],df_ExRet['Crude'].iloc[-test_days:])
ax0.set_xlabel('SPX Actual Return')
ax0.set_ylabel('WTI Actual Return')
ax1.scatter(pred_ret_copgen['SPX'][:,0:5].flatten('F'),pred_ret_copgen['Crude'][:,0:5].flatten('F'))
ax1.set_xlabel('SPX Pred Return')
ax1.set_ylabel('WTI Pred Return')
plt.show()

#Test Correlation
print(kendalltau(pred_ret_copgen['SPX'].flatten('F'),pred_ret_copgen['Crude'].flatten('F'))[0])
print(kendalltau(df_ExRet['SPX'].iloc[-test_days:],df_ExRet['Crude'].iloc[-test_days:])[0])



#%% 
##############################################################################
############   Applying Predicted Simulated Returns   ########################
##############################################################################
#We can now combine the copula method with bootstrapping for simulated 
#returns together with the factor model, and volatility estimates to get a
#simulated set of price paths for XOM. This then allows us to generate a 
#price distribution and make probabilistic statements.

#Can use the predicted returns from above for SPX and Crude

#Apply Factor model for XOM
#We want to estimate a 2-factor model of XOM returns on SPX and Crude

pred_ret_copgen['XOM'] = (rf_const + beta['Crude']*pred_ret_copgen['Crude'] + 
                          beta['SPX']*pred_ret_copgen['SPX'] + 
                          vol_est['XOM']*np.random.randn(test_days,nTrials))

#Plot results
fig, ax = plt.subplots()
ax.hist((pred_ret_copgen['XOM'].flatten('F'), df_Ret['XOM'].iloc[-test_days:].values),
        num_bins, density=True,
        label=['Predicted','Actual'])
ax.set_xlabel('Daily Returns')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of Actual vs Simulated XOM Returns')
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.show()

#Convert returns to end prices, will use last price of current series so 
#forecasting into futures. Note could update estimation to include last 90
#days.
predict_price = {}
#Starting at end
predict_price['XOM'] = df_Price['XOM'].iloc[-1]*np.exp(pred_ret_copgen['XOM'].cumsum(axis=0))

#Plot distribution
num_bins = 20
fig, ax = plt.subplots()
n, bins, patches = ax.hist(predict_price['XOM'][-1,:], num_bins, density=True)
plt.axvline(x=df_Price['XOM'].iloc[-1], color='red')
ax.set_xlabel('Ending Price XOM')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of Simulated XOM Prices')
plt.show()


#Probability End price>75 
pred_XOM_end_price=predict_price['XOM'][-1,:]
ProbGreater75 = (pred_XOM_end_price>75)
ProbGreater75=np.mean(ProbGreater75)
print ("Probability Greater than 75 in 90-days: {}".format(ProbGreater75))

#%%
##############################################################################
###################   Advanced Forecasting   #################################
##############################################################################

#NOTE: Rather than boostrapping could also make other assumptions and
#estimate returns with some kind of AR-model with or without GARCH
#adjustments and then draw the residuals via a copula. 

##### BELOW IS ADVANCED AND NOT COVERED COMPLETELY FOR THIS COURSE %%%%%%%
##### IT IS MEANT AS A REFERENCE FOR THOSE WHO WANT TO EXPLORE MORE 
##### IT  IS NOT NECCESSARILY BETTER %%%%%%%

#We could now calculate a GARCH model on these residuals, note that instead
#of using the raw returns we now use the residuals, after taking into
#account the movement in the return of XOM due to Oil. Note, this step may
#not be neccessary if it looks like the variance of the residuals is pretty
#constant, not clustered, not autocorrelated, etc. In this case could just 
#use the vol of the residuals. We could then estimate a GARCH AR(1,0,1)
#model for Crude, predicting returns and vols, and then use this to predict
#returns and Vols of XOM. 

# Rather than boostrapping could also make other assumptions and
# estimate returns with some kind of AR-model with or without GARCH
# adjustments and then draw the residuals via a copula.

### Applying copulas to residuals rather than returns (much more complicated)

###First we estimate AR(1) - GARCH models for the two series:
from arch import arch_model

alpha = {}
beta_GARCH= {}
omega = {}
V_LR = {}
AR_1 = {}
mu = {}
stdzd_resid = {}
sorted_resid = {}


for y in ['SPX','Crude']:
    model = arch_model(df_training[y]*100, mean='ARX', lags=[1], vol='GARCH', p=1, q=1, dist="StudentsT")
    model_fit = model.fit()
    print(model_fit.summary())
    
    #Return Parameters
    AR_1[y] = model_fit.params[str(y)+'[1]']
    mu[y] = model_fit.params['Const']/100
    
    #Volatility Parameters
    alpha[y] = model_fit.params['alpha[1]']
    beta_GARCH[y] = model_fit.params['beta[1]']
    omega[y] = model_fit.params['omega']/10000
    V_LR[y] = omega[y]/(1-alpha[y]-beta[y])

    #Residuals with unit variance
    stdzd_resid[y] = model_fit.resid/np.std(model_fit.resid)
    stdzd_resid[y].dropna(inplace=True)
    
    #Sorted Resid
    sorted_resid[y] = stdzd_resid[y].sort_values().reset_index(drop=True)

#Correlation of Residuals
rho_resid = np.cov((stdzd_resid['SPX'],stdzd_resid['Crude']))
#Convert to copula parameter
cop_resid = np.sin(rho_resid*np.pi/2)


#Sample from Residuals (so using Bootstrap with T Distribution)
df = 4
Z = np.random.default_rng().multivariate_normal([0, 0],
                                                cop_resid,
                                                size=(test_days,nTrials))
Chi = np.random.chisquare(df=df, size=(test_days,nTrials))

T1 = Z[:,:,0] * np.sqrt(df/Chi)
T2 = Z[:,:,1] * np.sqrt(df/Chi)

T1_cdf = t.cdf(T1, df=df)
T2_cdf = t.cdf(T2, df=df)

pred_resid = {}
idx = np.floor(T1_cdf*len(sorted_resid['SPX'])).astype(int) 
pred_resid['SPX'] = sorted_resid['SPX'][idx.flatten()].values.reshape(idx.shape)

idx = np.floor(T2_cdf*len(sorted_resid['Crude'])).astype(int) 
pred_resid['Crude'] = sorted_resid['Crude'][idx.flatten()].values.reshape(idx.shape)


#Also need to simulate our volatilities
pred_vol_GARCHEst = {}
#Expected Vol Path at time t+i (fixing starting vol to time 0)
for y in ['SPX','Crude']:
    pred_vol_GARCHEst[y] = np.zeros((test_days,nTrials))
    for i in range(test_days):
        if i==0:
            pred_vol_GARCHEst[y][0] = np.sqrt(omega[y] + beta_GARCH[y]*np.std(df_training[y])**2 +
                                              alpha[y]*df_training[y].iloc[-test_days].copy()**2)
        pred_vol_GARCHEst[y][i] = np.sqrt(V_LR[y] + ((alpha[y]+beta_GARCH[y])**i)*(pred_vol_GARCHEst[y][0]**2-V_LR[y]))


#Now can simulate returns
pred_ret_GARCHEst = {}
for y in ['SPX','Crude']:
    pred_ret_GARCHEst[y] = np.zeros((test_days,nTrials))
    #For first period use last training period returns
    pred_ret_GARCHEst[y][0,:] = (mu[y]+ AR_1[y]*df_ExRet[y].iloc[-test_days]+
                                 pred_vol_GARCHEst[y][0,:]*pred_resid[y][0,:])
    for i in range(1,test_days):
        pred_ret_GARCHEst[y][i,:] = (mu[y]+ AR_1[y]*pred_ret_GARCHEst[y][i-1,:]+
                                     pred_vol_GARCHEst[y][i,:]*pred_resid[y][i,:])

#Plot results
num_bins = 20
fig, ax = plt.subplots()
ax.hist((pred_ret_GARCHEst['SPX'].flatten('F'), df_ExRet['SPX'].iloc[-test_days:].values),
        num_bins, density=True,
        label=['Predicted','Actual'])
ax.set_xlabel('Daily Returns')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of Actual vs Simulated SPX Returns')
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.show()

fig, ax = plt.subplots()
ax.hist((pred_ret_GARCHEst['Crude'].flatten('F'), df_ExRet['Crude'].iloc[-test_days:].values),
        num_bins, density=True,
        label=['Predicted','Actual'])
ax.set_xlabel('Daily Returns')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of Actual vs Simulated WTI Returns')
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.show()

#Scatter Plot
fig, (ax0, ax1) = plt.subplots(2,1)
fig.suptitle('SPX vs WTI')
ax0.scatter(df_ExRet['SPX'].iloc[-test_days:],df_ExRet['Crude'].iloc[-test_days:])
ax0.set_xlabel('SPX Actual Return')
ax0.set_ylabel('WTI Actual Return')
ax1.scatter(pred_ret_GARCHEst['SPX'][:,0:5].flatten('F'),pred_ret_GARCHEst['Crude'][:,0:5].flatten('F'))
ax1.set_xlabel('SPX Pred Return')
ax1.set_ylabel('WTI Pred Return')
plt.show()

#Test Correlation
print(kendalltau(pred_ret_GARCHEst['SPX'].flatten('F'),pred_ret_GARCHEst['Crude'].flatten('F'))[0])
print(kendalltau(df_ExRet['SPX'].iloc[-test_days:],df_ExRet['Crude'].iloc[-test_days:])[0])


#Now Apply to XOM
pred_ret_GARCHEst['XOM'] = (rf_const + beta['Crude']*pred_ret_GARCHEst['Crude'] + 
                            beta['SPX']*pred_ret_GARCHEst['SPX'] + 
                            vol_est['XOM']*np.random.randn(test_days,nTrials))

#Plot results
fig, ax = plt.subplots()
ax.hist((pred_ret_GARCHEst['XOM'].flatten('F'), df_Ret['XOM'].iloc[-test_days:].values),
        num_bins, density=True,
        label=['Predicted','Actual'])
ax.set_xlabel('Daily Returns')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of Actual vs Simulated XOM Returns')
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.show()

#Convert returns to end prices, will use last price of current series so 
#forecasting into futures. Note could update estimation to include last 90
#days.
predict_price = {}
#Starting at end
predict_price['XOM'] = df_Price['XOM'].iloc[-1]*np.exp(pred_ret_GARCHEst['XOM'].cumsum(axis=0))

#Plot distribution
num_bins = 20
fig, ax = plt.subplots()
n, bins, patches = ax.hist(predict_price['XOM'][-1,:], num_bins, density=True)
plt.axvline(x=df_Price['XOM'].iloc[-1], color='red')
ax.set_xlabel('Ending Price XOM')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of Simulated XOM Prices')
plt.show()


#Probability End price>75 
pred_XOM_end_price=predict_price['XOM'][-1,:]
ProbGreater75 = (pred_XOM_end_price>75)
ProbGreater75=np.mean(ProbGreater75)
print ("Probability Greater than 75 in 90-days: {}".format(ProbGreater75))
