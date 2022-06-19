#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 21:41:41 2022

@author: mac
"""
import xlwings as xw
import pandas as pd
import random 
import numpy as np
from numpy.random import uniform
from matplotlib import pyplot as plt
import json

import os
os.chdir('/Users/taihanrui/PycharmProjects/pythonProject3')
book = xw.Book('dcf_model.xlsx')
book.sheets
dcf_sht = book.sheets('Sheet1') #rename the sheet with the dcf model
g_rate = dcf_sht.range ('E2').value #rename the cell with the growth rate
g_rate

price = dcf_sht.range('B45').value
price = float(price)
price


#Lets add a new sheet called Simulation
sim_sht = book.sheets.add('Simulation') #add a sheet called Simulation
g_mean = 0.10
g_std = 0.03
g_rate = random.normalvariate(g_mean,g_std)
g_rate

g_rates = [] #storage
dcf_values = [] #storage
book.sheets('Simulation').clear() #clear the sheet before simulating
num_sim = 10 #number of simulations

for i in range (num_sim):
    g_rate = random.normalvariate(g_mean, g_std)
    dcf_sht.range('E2').value = g_rate
    price =json.dumps(dcf_sht.range('B45').value)
    price = float(price)
    
    dcf_values.append(price)
    g_rates.append(g_rate)
    
    dcf_values_t= [[dcf_value]for dcf_value in dcf_values]
    g_rates_t= [[g_rate] for g_rate in dcf_values]
    
sim_sht.range('A1').value = 'Growth rate' #nameing the columns
sim_sht.range('C1').value = 'Value per share'
sim_sht.range('A2').value = g_rates_t #Starting from A2 put the simulated g rates
sim_sht.range('C2').value = dcf_values_t #Starting from C2 put the simulated dcf prices

#Create Data Frames
df = pd.DataFrame(dcf_values_t)
df.columns = ['Simulated Price']

#Plot distribution of estimated prices

sim_fig = plt.figure()
plt.hist(df, density = True, bins = 70)
plt.ylabel('Density')
plt.xlabel('$ Value per share')
plt.title('Distribution of value per share estimates')
plt.vlines(df.mean(),
           ymin = 0,
           ymax = 0.06 ,
           color = "red");

rng = book.sheets[0].range("H2") #save in excel - the location
sim_sht.pictures.add(sim_fig, name = 'Simulation', update = True, 
                    top = rng.top, left = rng.left)

#Add descriptive statistics
desc_stat = df.describe()
sim_sht.range('E2').value = desc_stat

#Lets use numpy to create a cumulative distribution function
fig_cdf = plt.figure()
x = np.sort(df['Simulated Price'])
y = np.arange(1,len(x)+1)/len(x)
plt.plot(x,y,marker = '.', linestyle = 'none')
plt.xlabel = ('$ Value per share')
plt.title('Cumulative Distribution Function')
plt.plot(x,y)
plt.show

rng = book.sheets[0].range("H21") #save in excel - the location
sim_sht.pictures.add(fig_cdf, name = 'CDF', update = True, 
                    top = rng.top, left = rng.left)

