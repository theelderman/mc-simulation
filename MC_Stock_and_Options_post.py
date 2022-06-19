import numpy as np
import itertools
from matplotlib import pyplot as plt
from scipy.stats import norm
from statistics import mean 
from itertools import islice
import pandas as pd

#%% 
#################################################################
#################       Modules       ###########################
#################################################################

#Stock Movement simulations
# Want to model movement in stock prices. Need an underlying model of how
# stock prices move. For example, a binomial model or something akin to
# the Black-Scholes type assumptions.

# Binomial Movements:
# Lets start with assuming stock follows a binomial setup where it can
# either go up or down following each time step of length h. 
# Need time step, total period, up movements, down movements, and current
# price.

def stock_movementBinom(S,u,d,rf,T,n,Trials,p="RN"):
    """
     Simulation of stock movements using a Binomial (up/down) setup

    Parameters
    ----------
    S : Float
        Starting Price
    u : Float
        proportional up movement, assumed to be given in terms of h (T/n)
    d : Float
        proportional down movement, assumed to be given in terms of h (T/n)
    rf : Float
        risk free rate, assumes annualized
    T : Float
        length of time, assume in years.
    n : Int
        number of steps.
    Trials : Int
        number of simulations

    Returns
    -------
    stock : list
        List of stock prices for n period x Trials based on parameters.
    """
    h=T/n
    stock = np.ones((n,Trials))*S #Starting point is price at time 0.
    if p=="RN":
        p = (np.exp(rf*h)-d)/(u-d) #Use RN probablility
    else:
        p = 0.5 #Assume 50/50 chance of up movement, note this is generally wrong
    variation = norm.cdf(np.random.randn(n,Trials)) #random probabilities pulled from normal
    upmovements = variation<p #Decide whether step is up or down
    for i in range(1,n):
        stock[i,:] = stock[i-1,:]*(upmovements[i,:]*u+(1-upmovements[i,:])*d)
    return stock

def stock_movementBS(S,mu,sigma,T,n, Trials):
    """
     Note this is using simulation of log-returns - See OFOD,
     Chapter 20.6 of 8th Edition (Chapter 21 of 10th/11th edition) for
     more detailed discussion.

    Parameters
    ----------
    S : Float
        Starting Price
    mu : Float
        Expected of returns given in units of T.
    sigma : Float
        Volatility of returns given in units of T.
    T : Float
        length of time.
    n : Int
        number of periods.
    Trials : Int
        number of simulations

    Returns
    -------
    stock : list
        List of stock prices for n period x Trials based on parameters.
    """
    h=T/n
    stock = np.ones((n,Trials))*S #Starting point is price at time 0.
    variation = np.random.randn(n,Trials) #random variables pulled from standard normal
    for i in range(1,n):
        stock[i,:] = stock[i-1,:]*np.exp((mu-0.5*sigma**2)*h+
                                         sigma*np.sqrt(h)*variation[i,:]) #Applying black-scholes like setup, 
    return stock

def monte_vanilla_options(S,mu,sigma,rf,T,n,K,Trials):
    """
    Runs monte carlo valuation using BS stock movement assumptions for standard
    put and call options.

    Parameters
    S : Float
        Starting Price
    mu : Float
        Expected of returns given in units of T.
    sigma : Float
        Volatility of returns given in units of T.
    rf : float
        risk-fre rate
    T : Float
        length of time.
    n : Int
        number of periods.
    K : Float
        Strike Price.
    Trials : Int
        Number of trials.

    Returns
    -------
    List of Lists
        List of list of stock prices, list of call payoff, and list of put payoffs
        for European options
    """
    stocks = stock_movementBS(S,mu,sigma,T,n, Trials) #Get stock prices
    call = np.maximum(stocks[n-1,:]-K,0)*np.exp(-rf*T) #Val of call for each sim
        #This is the payoff at time T, then discounted back to today at risk-free
        #rate, so it is today's time value.
    put = np.maximum(K-stocks[n-1,:],0)*np.exp(-rf*T)#Val of put for each sim
    return [stocks,call,put]


def BSVal(S,sigma, rf, T, K, Call=True):
    """
    Calculates the Black-Scholes option value for a European option

    Parameters
    ----------
    S : Float
        starting price.
    sigma : float
        Volatility of returns given in units of T, annualized.
    rf : float
        risk-fre rate, annualized
    T : float
        Time to expiration, annualized.
    K : float
        Strike.
    Call : Boolean, optional
        Whether option is call (if Call, then True). The default is True.

    Returns
    -------
    val : Float
        Value of option.

    """
    d1 = (np.log(S/K)+(rf+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 -sigma*np.sqrt(T)
    if Call==True:
        val = S*norm.cdf(d1)-K*np.exp(-rf*T)*norm.cdf(d2)
    else:
        val = K*np.exp(-rf*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
    return val

#%%
#################################################################
##############       Example       ###########################
#################################################################
#Current stock price of 40, risk-free rate of 2%, standard deviation of 30%,
#3-months to maturity, strike price of 40
S0=170.330
rf=0.02 #This is the risk-free rate annualized.
mu=rf
sigma = 0.367214537 #This is the volatility, annualized
K = 165
nTrials = 10000
T = 1/12 #This is the time you want to run, expressed in years.
num_periods=30 #This is the number of periods you want to run.

#We can do this one of two ways: either using Black Scholes or Binomial to
#generate stock price movements. Lets compare:

#First for Binomial, lets generate an estimate for u and d given the above
#parameters on volatility, timing, and the risk-free rate
u = np.exp(sigma*np.sqrt(T/num_periods))
d = 1/u

prices_Binom = stock_movementBinom(S0,u,d,rf,T,num_periods,nTrials)
#Plot Price paths:
stock_prices = prices_Binom
fig, ax = plt.subplots(nrows=1)
for i in range(0,5):
    ax.plot(stock_prices[:,i], label='Sim '+str(i))
ax.set_xlabel('Time Step', weight='bold')
ax.set_ylabel('Stock Price', weight='bold')
ax.set_title('Simulated Stock Prices for 5 Simulations', weight='bold', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.show()

#We can also look at a histogram of ending prices
fig, ax = plt.subplots()
ax.hist(stock_prices[-1,:], bins=20, linewidth=0.5, edgecolor="white")
plt.show()

#Now we can do the same for using a Black-Scholes setup:
prices= monte_vanilla_options(S0,mu,sigma,rf,T,num_periods,K,nTrials)
#Plot Price paths:
stock_prices = prices[0]
fig, ax = plt.subplots(nrows=1)
for i in range(0,5):
    ax.plot(stock_prices[:,i], label='Sim '+str(i))
ax.set_xlabel('Time Step', weight='bold')
ax.set_ylabel('Stock Price', weight='bold')
ax.set_title('Simulated Stock Prices for 5 Simulations', weight='bold', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.show()


#We can also look at a histogram of ending prices 
# - Roughly similar to Binom, but smoother
fig, ax = plt.subplots()
ax.hist(stock_prices[-1,:], bins=20, linewidth=0.5, edgecolor="white")
plt.show()


#Calculate option value
call_value=mean(prices[1])
print("Call Option Value: ", call_value)

#Can compare to analytical:
BSVal(S0,sigma, rf, T, K, Call=True)

#Calculate option value along each point in path
#I convert to a dataframe to avoid running as a loop an alternative method would be to run:
    #option_vals = np.zeros((num_periods,nTrials))
    #for i in range(0,num_periods):
        #for j in range(0,nTrials):
            #option_vals[i,j] = BSVal(stock_prices[i,j],sigma,rf,T-i*T/num_periods,K)

#Movements and Option Payoffs
stock_prices = stock_movementBS(S0,mu, sigma,T,num_periods,nTrials)
option_vals = pd.DataFrame(np.zeros((num_periods,nTrials)))
prices = pd.DataFrame(stock_prices)
#The below is a little complicated, will break it down:
#    1) we take the prices data frame, which consists of the simulated prices
#    For each of these, we want to calculate the option value on that date.
#    2) To do so, we will apply the Black Scholes formula at each point in time,
#    note that the underlying price is changing, and the time to maturity as we
#    progress through time.
#    3) We will use the lambda function and apply to do this. df.apply() will
#    apply what ever function is in apply() to each row or column. 
#    The lambda function applies the function to each x, where x is the value 
#    in the given row/column.
#    This could als be done using the loop above, which is slower but more 
#    transparent.
option_vals =  prices.apply(lambda x: BSVal(x,sigma,rf,T-x.index*T/num_periods, K))

##look at 5th and 95th percentile
pctile95 = option_vals.quantile(0.95,1) #Take 95th pctile for each time step
pctile5 = option_vals.quantile(0.05,1) #Take 5th pctile for each time step
fig, ax = plt.subplots(nrows=1)
ax.plot(pctile95, label='95th Pctile')
ax.plot(pctile5, label='5th Pctile')
ax.set_xlabel('Time Step', weight='bold')
ax.set_ylabel('Option Value', weight='bold')
ax.set_title('Simulated Option Values Across Simulations', weight='bold', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.show()
'''
#%%
#################################################################
##############       Asian Options    ###########################
#################################################################

#Movements and Option Payoffs
stock_prices = stock_movementBS(S0,mu, sigma,T,num_periods,nTrials)
Asian_payoff = np.maximum(np.mean(stock_prices,0)-K,0)

#Asian Option Value
asian_call_value = mean(Asian_payoff)*np.exp(-rf*T)
print("Asian Option Value: ", asian_call_value)

plt.hist(Asian_payoff, bins=50, facecolor='b' )
plt.axvline(mean(Asian_payoff), color='k', linestyle='dashed', linewidth=2)
plt.xlabel('Option Payoff')
plt.ylabel('Number of Occurences')
plt.text(1, 700, r'Avg. Payoff')
plt.xlim(0, 14)
plt.show()
'''
