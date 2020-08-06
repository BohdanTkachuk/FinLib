from math import pi

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm

from VanillaEuropenPut import vanilla_put_price
from ValuingAmericanOprionBoundaryPremium.IntegralApproxWeights import w, L_k, w_quad


def d_1(x, t, k, r, delta, sigma):
    if t == 0 and x == k:
        return 0
    else:
        # we have division by zero during computation of price of option
        res = (np.log(x / k) + (r - delta + sigma ** 2 * 0.5) * t) / (sigma * np.sqrt(t))
        return res


def d_2(x, t, k, r, delta, sigma):
    res = d_1(x, t, k, r, delta, sigma) - sigma * np.sqrt(t)
    return res


def integral_sum_boundary(x, i, r, K, h, x_prev, delta, w_i_j):
    sum = 0
    for j in range(0, i):
        sum = sum + w_i_j[i][j] * (r * K * np.exp(
            r * (i * h - j * h) - 1 / 2 * d_2(x, i * h - j * h, x_prev[j], r, delta, sigma) ** 2) -
                                   delta * x * np.exp(
                    -delta * (i * h - j * h) - 1 / 2 * d_1(x, i * h - j * h, x_prev[j], r, delta, sigma) ** 2))
    sum = sum + w_i_j[i][i] * (r * K - delta * x)
    return sum


def integral_sum_boundary_2(x, i, delta, h, x_prev, w_q):
    sum = 0
    for j in range(0, i):
        sum = sum + w_q[j] * np.exp(-delta * (i * h - h * j)) * norm.cdf(
            d_1(x, h * i - j * h, x_prev[j], r, delta, sigma))
    sum = sum + w_q[i] * 1 / 2
    return sum


def integral_eq_func(x, delta, i, h, K, sigma, r, x_prev, w_q, w_i_j):
    # TODO this one was created for debbuging
    first_term = -x * np.exp(-delta * h * i) * norm.cdf(d_1(x, i * h, K, r, delta, sigma)) \
                 + K / (sigma * np.sqrt(2 * np.pi * i * h)) * np.exp(
        -(r * i * h + 1 / 2 * d_2(x, i * h, K, r, delta, sigma) ** 2))
    firs_first = - x / (sigma * np.sqrt(2 * np.pi * i * h)) * np.exp(
        -(delta * i * h + 1 / 2 * d_1(x, i * h, K, r, delta, sigma) ** 2))
    second_term = 1 / (sigma * np.sqrt(2 * pi)) * integral_sum_boundary(x, i, r, K, h, x_prev, delta, w_i_j)
    third_term = - delta * x * integral_sum_boundary_2(x, i, delta, h, x_prev, w_q)
    sum = first_term + firs_first + second_term + third_term
    return sum


def initialB(K, r, delta):
    if (delta > r):
        return [(r / delta) * K]
    else:
        return [K]


# TODO Interpolation doesn't work correctly, need to review
def B_t(t, x_prev, n, h):
    B_final = 0
    for j in range(0, n + 1):
        # Below for debugging
        print("Time ", j * h, " value ", x_prev[j])
        B_final = B_final + x_prev[j] * L_k(t, j, n, h)
    return B_final


def first_integral_option(r, K, t, h, S, B_n, sigma, delta, w):
    sum = 0
    for i in range(0, n + 1):
        sum = sum + w[i] * r * K * np.exp(-r * (t - i * h)) * norm.cdf(-d_2(S, t - i * h, B_n[i], r, delta, sigma))
    return sum


def second_integral_option(r, K, t, h, S, B_n, sigma, delta, w):
    sum = 0
    for i in range(0, n + 1):
        sum = sum + w[i] * delta * S * np.exp(-delta * (t - i * h)) * norm.cdf(
            -d_1(S, t - i * h, B_n[i], r, delta, sigma))
    return sum


def AmericanPut(r, K, t, h, S, B_n, sigma, delta, w):
    res = vanilla_put_price(S, K, r, delta, sigma, t) + first_integral_option(r, K, t, h, S, B_n, sigma, delta, w) \
          - second_integral_option(r, K, t, h, S, B_n, sigma, delta, w)
    return res


def solutionBoundary(delta, h, K, sigma, r, w_q, w_i_j):
    x_prev = initialB(K, r, delta)
    for i in range(1, n + 1):
        parameters = (delta, i, h, K, sigma, r, x_prev, w_q, w_i_j)
        # for one case, we have problem in optimization and warning
        b_i = fsolve(integral_eq_func, 90, args=parameters)
        x_prev.append(b_i)
    return x_prev


K = 100
r = 0.06
sigma = 0.4
delta = 0
T = 0.5
n = 100
h = T / n

# weights for integral approximation
w_q = w_quad(n, T)
w_i_j = w(n, T)

# initial price
S = 100

# t is remaining time to maturity
t = T
B_n = solutionBoundary(delta, h, K, sigma, r, w_q, w_i_j)
print(AmericanPut(r, K, T, h, S, B_n, sigma, delta, w_q))

# Plot graph for boundary using interpolation
'''
xvals = np.arange(0.009, 3, 0.01)  # Grid of 0.01 spacing from -2 to 10
yvals = B_t(xvals, x_prev, n, h)  # Evaluate function on x
plt.plot(xvals, yvals)  # Create line plot with yvals against xvals
plt.show()  # Show the figure
'''
