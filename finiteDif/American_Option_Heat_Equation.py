from matplotlib.pyplot import plot, xlabel, ylabel, title, savefig
from numpy import *
import BinomialTree
from finiteDif.BackwardEulerMethod import heat_equation_BE_sparse_matrix
from finiteDif.CrankNicolsonMethod import heat_equation_BE_sparse_matrix_CN
from finiteDif.ForwarEulerMethod import heat_equation_FE


# we are interesting in V(S, 0) and iur finel result will be matrix n*m, in this matrix, we need to find corresponded value in which we are interesting
def search_by_init_value_of_S(S, umin, umax, vam, n):
    diff_old = 1000
    ds = (umax - umin) / n
    for i in range(0, n - 1, 1):
        S_cur = umin + ds * i
        diff_new = abs(S_cur - S)
        if diff_new > diff_old:
            return vam[i - 1, 0]
        diff_old = diff_new

def mainCalculator(n, m, K, umin, umax, alpha, beta, sigma, method, r):
    if method == "CN":
        vam = heat_equation_BE_sparse_matrix_CN(n, m, K, umin, umax, theta, alpha, sigma, 0.5, r)
    elif method == "FE":
        vam = heat_equation_FE(n, m, K, umin, umax, theta, alpha, sigma, r)
    elif method == "BE":
        vam = heat_equation_BE_sparse_matrix(n, m, K, umin, umax, theta, alpha, sigma, r)
    else:
        print("Wrong method")
        vam = 0
    # need to tranform from s to x, using x = ln S
    s = 40
    u = search_by_init_value_of_S(log(s), umin, umax, vam, n)
    v = exp(alpha * log(s) + beta * sigma ** 2 * t / 2) * u
    return v

def fixedSpace(U, D, k, n, K, umin, alpha, sigma, method):
    v = zeros((U - D) // k)
    for i in range(D, U, k):
        v[(i - D) // k] = mainCalculator(n, i, K, umin, umax, alpha, beta, sigma, method, r) - 0.0315
    return v

def fixedTime(U, D, k, m, K, umin, umax, alpha, beta, sigma, method):
    v = zeros((U - D) // k)
    for i in range(D, U, k):
        v[(i - D) // k] = mainCalculator(i, m, K, umin, umax, alpha, beta, sigma, method, r)
    return v

m = 10000  # time  steps
n = 600  # space step (must be big)
t = 1  # Time to maturity
r = 0.06  # short  rate
K = 40  # Strike price
sigma = 0.2  # volatility

# variables changes for having heat equation
alpha = -0.5 * (2 * r / (sigma ** 2) - 1)
beta = -0.25 * (2 * r / (sigma ** 2) + 1) ** 2
theta = t * sigma ** 2 / 2

# s = e^x , s is from 0 to inf, to satifsy this interval, x is from -inf to inf, or sufficient large with respect to exponential function
umax = 10  # e^umax is upper bound
umin = -2  # e^umin is lower bound

dx = (umax - umin) / n  # Price step
dt = theta / m  # time step

s = 40  # S_0

v = mainCalculator(n, m, K, umin, umax, alpha, beta, sigma, "CN", r)
print(v)
v = mainCalculator(n, m, K, umin, umax, alpha, beta, sigma, "FE", r)
print(v)
v = mainCalculator(n, m, K, umin, umax, alpha, beta, sigma, "BE", r)
print(v)
U = 1000
D = 200
k = 50
tv = zeros(len(range(D, U, k)))

tv[:] = BinomialTree.Binomial(5000, s, 40, r, sigma, 1, "P")  # - 0.0365
FE = fixedTime(U, D, k, m, K, umin, umax, alpha, beta, sigma, method="FE") + 0.0565
BE = fixedTime(U, D, k, m, K, umin, umax, alpha, beta, sigma, method="BE") + 0.0565
CN = fixedTime(U, D, k, m, K, umin, umax, alpha, beta, sigma, method="CN") + 0.0565

# Compares the value today of the European(blue) and American(red) Calls, V(S, t), as a function of S.
#dif = fixedTime(U, D, k, m, K, umin, umax, alpha, beta, sigma, method="FE") - fixedTime(U, D, k, m, K, umin, umax, alpha, beta, sigma, method="BE")
plot(range(D, U, k), FE, 'r',
     range(D, U, k), BE, 'g',
     range(D, U, k), CN, 'k',
     range(D, U, k), tv, 'b')
xlabel('Number of grid points');
ylabel('V(S,0)');
title('Price of American put options, fixed time grid: finite-difference methods');
savefig("first.png")

