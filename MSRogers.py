from numpy import *
from numpy.random import standard_normal, seed
import warnings
from matplotlib.pyplot import *
from time import time

from scipy.optimize import fsolve
from scipy.stats import norm


t0 = time()
warnings.simplefilter('ignore', np.RankWarning)
## Simulation  Parameters
#seed(150000)  # seed  for  Python  RNG
M = 50  # time  steps
I = 300  # paths  for  valuation
reg = 7  # no of  basis  functions
AP = True  # antithetic  paths
MM = True  # moment  matching  of RN
# ## Parameters  -- American  Put  Option
r = 0.06  # short  rate
v = 0.4  # volatility
S0 = 120.  # initial  stock  level
T = 0.5  # time -to -maturity
V0_right = 4.478  # American  Put  Option (500  steps  bin. model)
dt = T / M  # length  of time  interval
df = exp(-r * dt)  # discount  factor  per  time  interval
K = 100

def RNG(I):
    if AP == True:
        ran = standard_normal(I // 2)
        ran = concatenate((ran, -ran))
    else:
        ran = standard_normal(I)
    if MM == True:
        ran = ran - mean(ran)
        ran = ran / std(ran)
        return ran


def GenS(S0, I, M):
    S = zeros((M + 1, I), 'd')  # index  level  matrix
    S[0, :] = S0  # initial  values
    for t in range(1, M + 1, 1):  # index  level  paths
        ran = RNG(I)
        S[t, :] = S[t - 1, :] * exp((r - v ** 2 / 2) * dt + v * ran * sqrt(dt))
    return S


def IV_discounted(S):
    df = exp(-r*dt*arange(0,M+1, 1)).reshape(M+1,1)
    ex = maximum(100. - S, 0)
    res = multiply(df,ex)
    return multiply(df,ex)

def IV(S):
    df = exp(-r*dt*arange(0,M+1, 1)).reshape(M+1,1)
    ex = maximum(100. - S, 0)
    res = multiply(df,ex)
    return ex

def d_j(j, S, K, r, v, T,t):
    """
    d_j = \frac{log(\frac{S}{K})+(r+(-1)^{j-1} \frac{1}{2}v^2)T}{v sqrt(T)}
    """
    return (log(S/K) + (r + ((-1)**(j-1))*0.5*v*v)*(T-t))/(v*((T-t)**0.5))

def european_martingale(S,K,r,v,T,t):

    return -S * norm.cdf(-d_j(1, S, K, r, v, T, t)) + \
           K * exp(-r * (T-t)) * norm.cdf(-d_j(2, S, K, r, v, T, t))

def eur_martingale_table(S, K, r,v, T,I):
    V = zeros((M + 1, I), 'd')
    dt= T/M
    S0 = K
    #S = GenS(S0, I, M)
    RF = IV(S)
    RF_disc = IV_discounted(S)
    for t in range(0, M, 1):  # index  level  paths
        RF_test = RF_disc[t,:]
        #V[t, :] = np.where( RF_test > 0, european_martingale(S[t, :], K, r, v, T, t*dt) - european_martingale(K, K, r, v, T, t*dt),  RF_test)
        V[t, :] = european_martingale(S[t, :], K, r, v, T, t*dt)
    V[M,:] = maximum(K - S[M,:], 0)
    return V


def sup_fromPath(l, V, RF, I,M):
    res = np.amax( l*V -RF, axis=0)
    sum = 0
    for i in range(0, I):
        sum = sum + res[i]
    return sum/I


S = GenS(S0,I,M)
V = eur_martingale_table(S, K,r,v,T,I)
#S = GenS(S0,I,M)
RF = IV_discounted(S)
parameters = (V, RF, I, M)
l_star = fsolve(sup_fromPath, 1, args=parameters)
l_star = 1
I = 100000
S = GenS(S0, I,M)
V = eur_martingale_table(S, K,r,v,T,I)
S = GenS(S0,I,M)
RF = IV_discounted(S)
#S = GenS(S0, I,M)
V = eur_martingale_table(S, K,r,v,T,I)
print(sup_fromPath(l_star, V, RF, I,M))

#print(vanilla_put_price(100, 100, 0.06, 0.4, 0.5))