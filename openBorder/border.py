from cmath import log, sqrt
from math import pi, exp, inf

from scipy.optimize import least_squares, fsolve
from scipy.special import comb
from numpy.ma import zeros
from scipy.stats import norm
import scipy.integrate as integrate
import quadpy

from openBorder.weights import w_quadr, w


def d_1(x, t, k, r, delta, sigma):
    return (log(x / k).real + (r - delta + sigma ** 2  * 0.5) * t) / (sigma * sqrt(t).real)

def d_2(x, t, k, r, delta, sigma):
    return d_1(x, t, k, r, delta, sigma) - sigma * sqrt(t)

def integral_sum(x, i, r, K, h, x_prev, delta):
    sum = 0
    w_i_j = w(d, n, T)
    for j in range(0, i + 1):

        test = w_i_j[i][j]
        sum = + w_i_j[i][j] * (r * K * exp(-r * ((i+1) * h - h * (j+1)) - 0.5 * d_2(x, (i+1) * h - h * (j+1), x_prev[j], r, delta, sigma) ** 2)
                           - delta * x * exp(-delta((i+1) * h - (j+1) * h) - 0.5 * d_1(x, (i+1) * h - h * (j+1), x_prev[j], r, delta, sigma) ** 2))
    return sum

def integral_sum_2(x, i, delta, h, x_prev):
    sum = 0
    w_q = w_quadr()
    for j in range(0, i + 1):
        sum = + w_q[j] * exp(-delta * ((i+1) * h - h * (j+1))) * norm.cdf(d_1(x, (i+1) * h - h * (j+1), x_prev[j], r, delta, sigma))
    return sum

def f(x, delta, i, K, sigma, r, T, n, x_prev):
    """
    delta = parameters[0]
    i = parameters[1]
    K = parameters[2]
    sigma = parameters[3]
    r  = parameters[4]
    T = parameters[5]
    n = parameters[6]
    x_prev = parameters[7]
    """
    h = T / n
    first_first_term = -x * exp(-delta * (i+1) * h).real * norm.cdf(d_1(x, (i +1) * h, K, r, delta, sigma), 0, 1).real
    first_second_term =  K / (sigma * sqrt(2 * pi * (i+1) * h).real) * exp(
        -(r * (i+1) * h + 1 / 2 * d_2(x, (i+1) * h, K, r, delta, sigma) ** 2).real)
    first_term = first_first_term + first_second_term
    second_term = -x / (sigma * sqrt(2 * pi * (i+1) * h)).real * exp(-(delta * (i + 1) * h + 1 / 2 * d_1(x, (i+1) * h, K, r, delta, sigma) ** 2)).real
    third_term =  + 1 / (sigma * sqrt(2 * pi)).real * integral_sum(x, i, r, K, h, x_prev, delta)
    - delta * x * integral_sum_2(x, i, delta, h, x_prev)
    return first_term + second_term + third_term
    '''return -x * exp(-delta * i * h).real * norm.cdf(d_1(x, i * h, K, r, delta, sigma), 0, 1) + K / (sigma * sqrt(2 * pi * i * h).real) * exp(
        -(r * i * h + 1 / 2 * d_2(x, i * h, K, r, delta, sigma) ** 2)).real
    -x / (sigma * sqrt(2 * pi * i * h)).real * exp(-(delta * i * h + 1 / 2 * d_1(x, i * h, K, r, delta, sigma) ** 2)).real
    + 1 / (sigma * sqrt(2 * pi)).real * integral_sum(x, i, r, K, h, x_prev, delta)
    - delta * x * integral_sum_2(x, i, delta, h, x_prev)'''

def initialB(K, r, delta):
    if (delta > r):
        return [(r / delta) * K]
    else:
        return [K]

K = 100
r = 0.08
sigma = 0.2
delta = 0
T = 3
n = 64-1
d = 3

x_prev = initialB(K, r, delta)

for i in range(0, n):
    parameters = (delta, i, K, sigma, r, T, n, x_prev)
    b_i = fsolve(f, 2, args=parameters)
    x_prev.append(b_i)

print(x_prev)
# solution_1 = fsolve(functions, 0, args=(delta, i, K, sigma, r,T,n,x_prev))
# solution_2 =  least_squares(f, 0, args=(delta, i, K, sigma, r,T,n,x_prev), bounds=(0, inf))
