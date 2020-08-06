from scipy.optimize import fsolve

from ValuingAmericanOprionBoundaryPremium.IntegralApproxWeights import L_k, w


def integral_sum(x, i, x_prev, n, T, w):
    sum = 0
    h = T / n
    for j in range(0, i):
        sum = sum + w[i][j] * x_prev[j] * (i * h - j * h) ** (3 / 2)
    sum = sum + w[i][i] * x * (i * h - i * h)
    return sum


# solving t - t^3 / 6 + \int_{0}^{1}{f(s)*(t - s)^{3/2}/((t - s)^{1/2})} - f(t)
def func(x, i, x_prev, h, n, T, w):
    res = i * h - (i * h) ** 3 / 6 + integral_sum(x, i, x_prev, n, T, w) - x
    return res


def initialB():
    return [0]


def B_t(t, x_prev, n, h):
    B_final = 0
    # need to change n+1
    for j in range(0, n + 1):
        print("Time ", j * h, " value ", x_prev[j])
        B_final = B_final + x_prev[j] * L_k(t, j, n, h)
    return B_final


T = 1
n = 100
w_i_j = w(n, T)
x_prev = initialB()
h = T / n
for i in range(1, n + 1):
    parameters = (i, x_prev, h, n, T, w_i_j)
    b_i = fsolve(func, 1, args=parameters)
    x_prev.append(b_i)

print(x_prev)
