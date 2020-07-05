from numpy import *
from numpy.random import standard_normal, seed
import warnings
import numpy.polynomial.laguerre as a
from matplotlib.pyplot import *
from time import time

import BinomialTree

t0=time()
warnings.simplefilter('ignore', np.RankWarning)
## Simulation  Parameters
seed(150000)      # seed  for  Python  RNG
M = 100        # time  steps
I = 10000       # paths  for  valuation
reg = 7      # no of  basis  functions
AP = True          # antithetic  paths
MM = True          # moment  matching  of RN
# ## Parameters  -- American  Put  Option
r = 0.06          # short  rate
vol = 0.2           # volatility
S0 = 40.           # initial  stock  level
T = 1.0           # time -to -maturity
V0_right = 4.478 # American  Put  Option (500  steps  bin. model)
dt = T/M           # length  of time  interval
df = exp(-r*dt)   # discount  factor  per  time  interval
## Function  Definitions
def  RNG(I):
    if AP == True:
        ran=standard_normal(I//2)
        ran=concatenate ((ran,-ran))
    else:
        ran=standard_normal(I)
    if MM == True:
        ran=ran - mean(ran)
        ran=ran/std(ran)
        return ran
def  GenS(I,M):
    S=zeros((M+1,I),'d')           # index  level  matrix
    S[0,:]=S0                      # initial  values
    for t in range(1,M+1,1):       # index  level  paths
        ran=RNG(I)
        S[t,:]=S[t-1,:]* exp((r-vol**2/2)*dt+vol*ran*sqrt(dt))
    return S
def IV(S):
    return  maximum(40.- S,0)
## Valuation  by LSM
def mainPricing(I,M,df):
    S=GenS(I,M)                     # generate  stock  price  paths
    h=IV(S)                       # inner  value  matrix
    V=IV(S)                       # value  matrix
    for t in range(M-1,-1,-1):
        #rg = polyfit(S[t,:], V[t+1,:]*df, reg)           # regression  at time t
        rg = a.lagfit(S[t, :], V[t + 1, :] * df, reg)
        C = a.lagval(S[t, :], rg, True)
        #C = polyval(rg, S[t, :])
        ##C = polyval(rg,S[t,:])                            # continuation  values
        V[t,:]= where(h[t,:]>C,h[t,:],V[t+1,:]*df)   # exercise  decision
    V0=sum(V[0,:])/I # LSM  estimator
    return V0

def mainPricingPol(I,M,df):
    S=GenS(I,M)                     # generate  stock  price  paths
    h=IV(S)                       # inner  value  matrix
    V=IV(S)                       # value  matrix
    for t in range(M-1,-1,-1):
        rg = polyfit(S[t,:], V[t+1,:]*df, reg)           # regression  at time t
        #rg = a.lagfit(S[t, :], V[t + 1, :] * df, reg)
        #C = a.lagval(S[t, :], rg, True)
        C = polyval(rg, S[t, :])
        ##C = polyval(rg,S[t,:])                            # continuation  values
        V[t,:]= where(h[t,:]>C,h[t,:],V[t+1,:]*df)   # exercise  decision
    V0=sum(V[0,:])/I # LSM  estimator
    return V0

def fixedTime(U,D,k):
    v = zeros((U-D)//k)
    for i in range(D,U, k):
        v[(i-D)//k] = mainPricing(i,M,df)
    return v

def fixedTime2(U,D,k):
    v = zeros((U-D)//k)
    for i in range(D,U, k):
        v[(i-D)//k] = mainPricingPol(i,M,df)
    return v
def fixedNumberOfPaths(U,D,k):
    v = zeros((U - D) // k)
    for i in range(D, U, k):
        v[(i - D) // k] = mainPricing(I, i, df)
    return v
r = 0.06          # short  rate
vol = 0.2           # volatility
S0 = 40.           # initial  stock  level
T = 1.0           # time -to -maturity
V0_right = 4.478 # American  Put  Option (500  steps  bin. model)
dt = T/M           # length  of time  interval
U = 20000
D = 100
k = 100
tv = zeros(len(range(D,U,k)))
tv[:]= BinomialTree.Binomial(5000,S0,40,r,vol,1,"P")
dif = fixedTime(U,D,k)-fixedTime2(U,D,k)
# Compares the value today of the European(blue) and American(red) Calls, V(S, t), as a function of S.
plot(range(D,U, k), fixedTime(U,D,k)+0.05, 'r',
     range(D,U, k),fixedTime2(U,D,k), 'k',
     range(D,U,k), tv, 'b')
xlabel('Number of paths');
ylabel('V(S,0)');
title('Price of American option depends on number of path');
savefig("second.png")

