from cmath import log, sqrt
from math import pi, exp, inf

from scipy.optimize import least_squares, fsolve
from scipy.optimize._tstutils import functions
from scipy.special import comb
from numpy.ma import zeros
from scipy.stats import norm
import scipy.integrate as integrate




def norm_cdf(x):
    """
    An approximation to the cumulative distribution
    function for the standard normal distribution:
    N(x) = \frac{1}{sqrt(2*\pi)} \int^x_{-\infty} e^{-\frac{1}{2}s^2} ds
    """
    k = 1.0 / (1.0 + 0.2316419 * x).real
    k_sum = k * (0.319381530 + k * (-0.356563782 +
                                    k * (1.781477937 + k * (-1.821255978 + 1.330274429 * k))))
    print(type(sqrt(k_sum)))
    if x.real >= 0.0:
        return (1.0 - (1.0 / ((2 * pi) ** 0.5)) * exp(-0.5 * x.real * x.real) * k_sum)
    else:
        return 1.0 - norm_cdf(-x)


def trapezoidal(f, a, b, n):
    h = float(b-a)/n
    result = 0.5*f(a) + 0.5*f(b)
    for i in range(1, n):
        result += f(a + i*h)
    result *= h
    return result


def sumJi(d, i, n):
    sum = 0
    # need to add 1 to the upper bound
    for j in range(max(1, i - d), min(i, n - d - 1) + 1):
        sum = +comb(d, i - j, exact=False)
    return sum


def beta(d, n):
    b = zeros(n+1)
    for i in range(0, n + 1):
        b[i] = round((-1) ** (i - d) * sumJi(d, i, n), 4)
    return b

def beta_i(d, n, i):
    b_i =  round((-1) ** (i - d) * sumJi(d, i, n), 4)
    return b_i

def denomL(n, t, T, d):
    h = T / n
    sum = 0
    b = beta(d, n)
    for i in range(1, n + 1):
        newSum = round(b[i], 2) / (t - i * h)
        sum = sum +newSum
    return sum


def L(t, d, n, T):
    h = T / n
    L = zeros(n+1)
    denom = denomL(n, t, T, d)
    beta_calculated = beta(d, n)
    for k in range(0, n + 1):

        L[k] = (round(beta_calculated[k]) / (t - (k+1) * h)) / denom
    return L

def Lk (t, d, n, T, k):
    h = T / n
    denom = denomL(n, t, T, d)
    L = (beta(d,n)[k]/(t - (k+1) * h)) / denom
    return L


def w(d, n, T):
    h = T / n
    w = zeros((n+1,n+1))
    for i in range(0, n + 1):
        for j in range(0, n+1):

            y, err = integrate.quad(lambda x: Lk(x, d, n, T, j) / sqrt((i+1) * h - x), 0, (i+1) * h)
            w[i][j] = y

    return w

def w_quadr(n, T):
    w = zeros(n + 1)
    for j in range(0, n + 1):
        w[j] = integrate.quad(lambda x: L(x, d, n, T)[j], 0, T)


def d_1(x, t, k, r, delta, sigma):
    return (log(x / k).real + (r - delta + sigma ** 2  * 0.5) * t) / (sigma * sqrt(t).real)


def d_2(x, t, k, r, delta, sigma):
    return d_1(x, t, k, r, delta, sigma) - sigma * sqrt(t)


def integral_sum(x, i, r, K, h, x_prev, delta):
    sum = 0
    w_i_j = w(d, n, T)
    for j in range(0, i + 1):
        sum = + w_i_j[i][j] * (r * K * exp(-r * (i * h - h * j) - 0.5 * d_2(x, i * h - h * j, x_prev[j]) ** 2)
                           - delta * x * exp(-delta(i * h - j * h) - 0.5 * d_1(x, i * h - h * j, x_prev[j]) ** 2))
    return sum



def integral_sum_2(x, i, delta, h, x_prev):
    sum = 0
    for j in range(0, i + 1):
        sum = + w_quadr[j] * exp(-delta * (i * h - h * j)) * norm.cdf(d_1(x, i * h - h * j, x_prev[j]))
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
    first_first_term = -x * exp(-delta * i * h).real * norm.cdf(d_1(x, i * h, K, r, delta, sigma), 0, 1).real
    first_second_term =  K / (sigma * sqrt(2 * pi * i * h).real) * exp(
        -(r * i * h + 1 / 2 * d_2(x, i * h, K, r, delta, sigma) ** 2).real)
    first_term = first_first_term + first_second_term
    second_term = -x / (sigma * sqrt(2 * pi * i * h)).real * exp(-(delta * i * h + 1 / 2 * d_1(x, i * h, K, r, delta, sigma) ** 2)).real
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
        return (r / delta) * K
    else:
        return K


K = 100
r = 0.08
sigma = 0.2
delta = 0
T = 3
n = 64-1
d = 3

x_prev = initialB(K, r, delta)

for i in range(1, n + 1):
    parameters = (delta, i, K, sigma, r, T, n, x_prev)
    b_i = fsolve(f, 2, args=parameters)
    x_prev.append(b_i)

print(x_prev)
# solution_1 = fsolve(functions, 0, args=(delta, i, K, sigma, r,T,n,x_prev))
# solution_2 =  least_squares(f, 0, args=(delta, i, K, sigma, r,T,n,x_prev), bounds=(0, inf))
