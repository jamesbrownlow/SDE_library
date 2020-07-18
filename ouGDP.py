# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 12:48:45 2020
Fit OU model to GDP

@author: DrJ
The solution to a SDE is a probability distribution
"""

from os import chdir
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sys import exit
#import MLEfitOU as OU
import sde_library as slick


chdir('E:\SDEgithub')


dataGDP = pd.read_csv('realGDP.csv', usecols=[1])


GDP = dataGDP['GDP']


plt.plot(GDP, color='black')
plt.title('GDP growth rate from Q2 1947 to present')
plt.xlabel('Quarter from 4/47')
plt.ylabel('GDP growth/decline %')
plt.grid()
plt.show()

startQtr = 276
startQtr= 200

quarterAxis = np.linspace(0,len(GDP[startQtr:]),(len(GDP[startQtr:])))
plt.plot(quarterAxis, GDP[startQtr:], color='black')
plt.title('GDP growth, last 4 years')
plt.title('GDP growth last 16 Q')
plt.xlabel('Quarter from 3/2016')
plt.ylabel('GDP growth/decline')
plt.grid()
plt.show()


# fit OU model

dt = 0.25
#dt=2
result = slick.ouMLEfit(np.array(GDP[startQtr:]),dt)
if (result != -1):
    mu, alpha, sigma =  slick.ouMLEfit(np.array(GDP[startQtr:]),dt)
else:
    print( 'ARMA had negative AR coefficient')
    exit()

print('reversion rate: {} per quarter'.format(round(1/alpha,2)))


#generate 8 quarters of sim runs
nRuns = 50
nQtr=8
tAxis = range(nQtr)
GDP0 = -5.0

# MLE

meanGDP = np.zeros(nQtr)
meanGDP[0] = GDP0

plt.figure(1)
for run in range(nRuns):
    p =[]
    p.append(meanGDP[0])
   
    for i in range(1,nQtr):
        p.append(p[i-1]*np.exp(-alpha*dt) + mu*(1-np.exp(-alpha*dt)) + \
                    sigma*np.sqrt((1-np.exp(-2*alpha*dt))/(2*alpha)) * \
                    norm.rvs(size=1))
        meanGDP[i] += (p[i]/nRuns)
    plt.plot(tAxis, p,'--')
    
plt.plot(tAxis,meanGDP,'k-')
plt.xlabel('Q, from 2020 Q1')
plt.ylabel('GNP growth %')
plt.title('OU- MLE model, GDP')
plt.grid()
plt.show()
    
    
print('growth rate back to nominal {} %, \n \
      half-life {} quarter'.format(round(mu,1), \
                round(np.log(2)/alpha,2)))
    
meanQ3, stdQ3 =  slick.ouDistn(GDP0, 2, mu, alpha, sigma)
GDPvalues = np.linspace(start = meanQ3-2*stdQ3, stop = meanQ3+2*stdQ3, num=50)
textstr = '$\mu=%.2f$\n$\sigma=%.2f$'%(mu, stdQ3)

plt.figure(2)
fig, ax = plt.subplots(1)
plt.plot(GDPvalues,norm.pdf(GDPvalues,loc=meanQ3, scale=stdQ3))
plt.xlabel('Expected GDP growth rate, Q3 2020')
plt.ylabel('probability density')
plt.title('OU predicted growth rate,GDP, Q3 2020')

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.grid()
plt.show()
      
# conditional likelihood fit

condLikeFit=slick.ou_fit(np.array(GDP[startQtr:]), dt )
#mleFIT = slick.ouMLEfit(GDP[startQtr:], dt)

mu,alpha,sigma = condLikeFit

if sigma > 0:
#    print('Error: sigma estimated to be <= 0')
#    exit()
    

    plt.figure(3)

    meanGDP = np.zeros(nQtr)
    GDP0 = -5.0

    for run in range(nRuns):
        plt.plot(tAxis,slick.ou_path(mu,alpha,sigma,dt, nQtr, GDP0),'--')
        meanGDP += slick.ou_path(mu,alpha,sigma,dt, nQtr, GDP0)/nRuns

    plt.plot(tAxis,meanGDP,'k-')    
    plt.xlabel('Q, from 2020 Q1')
    plt.ylabel('GNP growth %')
    plt.title('OU- Conditional Likelihood model, GDP')
    plt.grid()
    plt.show()


slick.ouProfileLike(np.array(GDP[startQtr:]), dt)

