import numpy
import scipy.integrate as integrate
from numpy.ma import zeros, sqrt
from scipy.special import comb
import quadpy

scheme = quadpy.c1.clenshaw_curtis(5)


def sumJi(d, i, n):
    sum = 0
    # need to add 1 to the upper bound
    for j in range(max(1, i - d), min(i, n - d - 1) + 1):
        sum += comb(d, i - j, exact=False)
    return sum

def nu_k (n, k):
    d = 3
    b = round((-1) ** (k - d) * sumJi(d, k, n), 4)
    return b
'''
def nu_k (n, k):
    if k == 0 or k==n:
        return (-1)**k *0.5
    else:
        return (-1)**k
'''
def denomL(x, n, h):
    sum = 0
    for i in range(0,n+1):
        sum = sum + nu_k(n,i)/ (x - h*i)
    return sum


def L_k(x,k,n,h):
    nom_k = nu_k(n,k)/(x-h*k)

    return nom_k/denomL(x,n,h)

def w_quad(n, T):
    h = T/n
    w = zeros(n+1)
    scheme =quadpy.c1.gauss_legendre(40)
    for i in range(0, n+1):
        w[i] = scheme.integrate(lambda x: L_k(x, i, n, h), [0, T])
    return w

def w(n, T):
    h = T/n
    w = zeros((n+1,n+1))
    scheme = quadpy.c1.gauss_legendre(20)
    for i in range(1, n+1):
        for j in range(0, i+1):
            y = scheme.integrate(lambda x: L_k(x, j,n,h) / numpy.sqrt(h*i - x), [0, i * h])
            w[i][j] = y
    return w

'''def func_w (x,j,n,h,i):
    res = L_k(x,j,n,h) / numpy.sqrt(h*i - x)
    return res

def w(n, T, i, j):
    h = T/n
    w = 0
    scheme = quadpy.c1.gauss_legendre(10)
    w = scheme.integrate(lambda x: func_w(x,j,n,h,i), [0, i*h] )
    return w

'''
