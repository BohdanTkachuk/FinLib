from scipy.optimize import fsolve

from ValuingAmericanOprionBoundaryPremium.IntegralApproxWeights import w_quad, L_k


def integral_sum(x, x_prev, i, h, n, w):
    sum = 0
    w_test = w
    for j in range(0, i):
        sum = sum + w_test[j] * j * h * i * h * x_prev[j]
    sum = sum + w_test[i] * i * h * i * h * x
    return sum


def func(x, x_prev, i, h, n, w):
    return (i * h) ** 5 - (i * h) ** 8 / 7 + integral_sum(x, x_prev, i, h, n, w) - x


def initialB():
    return [0]


x_prev = initialB()
T = 2
n = 400
h = T / n
w = w_quad(n, T)
for i in range(1, n + 1):
    parameters = (x_prev, i, h, n, w)
    b_i = fsolve(func, 0, args=parameters)
    x_prev.append(b_i)


def B_t(t, x_prev, n, T):
    h = T / n
    B_final = 0
    for j in range(0, n + 1):
        print("Time ", j * h, " value ", x_prev[j])
        B_final = B_final + x_prev[j] * L_k(t, j, n, h)
    return B_final
