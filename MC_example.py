import xlwings as xw
import pandas as pd
import random
import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt


book = xw.Book("MC_model_v1.xlsx")
book.sheets

dcf_sht = book.sheets('DCF')

g_rate = dcf_sht.range('E10').value
g_rate
price = dcf_sht.range('B37').value
price = float(price)
price

g_rate = dcf_sht.range('E10').value = 0.05
g_rate
price = dcf_sht.range('B37').value
price = float(price)
price

sim_sht = book.sheets.add("MC2")

g_mean = 0.02
g_std  = 0.003
g_rate = random.normalvariate(g_mean, g_std)
g_rate

g_rates    = []
dcf_values = []
book.sheets('MC2').clear
num_sim = 100

for i in range(num_sim):
    g_rate = random.normalvariate(g_mean, g_std)
    dcf_sht.range("E10").value = g_rate
    price = dcf_sht.range("B37").value
    price  = float(price)

    dcf_values.append(price)
    g_rates.append(g_rate)

    dcf_values_t = [[dcf_value] for dcf_value in dcf_values]
    g_rates_t = [[g_rate] for g_rate in g_rates]

sim_sht.range("a1").value = 'Terminal g'
sim_sht.range("c1").value = 'Share Price'
sim_sht.range("a2").value = g_rates_t
sim_sht.range("c2").value = dcf_values_t

df = pd.DataFrame(dcf_values_t)
df.columns = ['Sim Share Price']
df.head()

sim_fig = plt.figure()
plt.hist(df,density = True, bins = 10)
plt.ylabel('Density')
plt.xlabel('Share Price')
plt.title('MC Simulation')
plt.vlines(df.mean(),
    ymin = 0,
    ymax = 6,
    color = "red")
plt.vlines(38.7,
    ymin = 0,
    ymax = 6,
    color = "green")

rng = book.sheets[0].range("H2")
sim_sht.pictures.add(sim_fig, name = 'MC Plot', update = True, top=rng.top, left=rng.left)

desc_stat = df.describe()
sim_sht.range('E2').value = desc_stat

fig_cdf = plt.figure()
x = np.sort(df['Sim Share Price'])
y = np.arange(1,len(x)+1)/len(x)
plt.plot(x,y, marker='.', linestyle='none')
plt.show

rng = book.sheets[0].range("H25")
sim_sht.pictures.add(fig_cdf, name = 'MC CDF Plot', update = True, top=rng.top, left=rng.left)

