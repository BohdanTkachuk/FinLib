import numpy
import scipy.integrate as integrate
from numpy.ma import zeros
from scipy.special import comb
import quadpy

scheme = quadpy.c1.clenshaw_curtis(5)


def sumJi(d, i, n):
    sum = 0
    # need to add 1 to the upper bound
    for j in range(max(1, i - d), min(i, n - d - 1) + 1):
        sum = +comb(d, i - j, exact=False)
    return sum

'''def beta(d, n):
    b = zeros(n+1)
    for i in range(0, n + 1):
        b[i] = round((-1) ** (i - d) * sumJi(d, i, n), 4)
    return b
'''
def beta(n):
    b = zeros(n)
    for i in range(0, n):
        b[i] = (-1)**i
    return b

def beta_i(d, n, i):
    b_i =  round((-1) ** (i - d) * sumJi(d, i, n), 4)
    return b_i

def denomL(n, t, T, d):
    h = T / n
    sum = 0
    b = beta(n)
    for i in range(0, n):
        newSum = round(b[i], 2) / (t - (i+1) * h)
        sum = sum + newSum
    return sum

def L(t, d, n, T):
    h = T / n
    L = zeros(n)
    denom = denomL(n, t, T, d)
    beta_calculated = beta(d, n)
    for k in range(0, n ):
        L[k] = (round(beta_calculated[k]) / (t - (k+1) * h)) / denom
    return L

def Lk (t, d, n, T, k):
    h = T / n
    denom = denomL(n, t, T, d)
    L = (beta(n)[k]/(t - (k+1) * h)) / denom
    return L

def w(d, n, T):
    h = T / n
    w = zeros((n,n))
    scheme = quadpy.c1.gauss_legendre(5)
    for i in range(0, n ):
        for j in range(0, n):
            y = scheme.integrate(lambda x: Lk(x, d, n, T, j) / numpy.sqrt((i+1) * h - x), [0, (i+1) * h])
            w[i][j] = y
    return w

def w_quadr(n, T):
    w = zeros(n)
    d = 0
    scheme = quadpy.c1.clenshaw_curtis(5)
    for j in range(0, n):
        w[j] = scheme.integrate(lambda x: Lk(x, d, n, T, j), [0, T])