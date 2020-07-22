from os import truncate

from numpy import *
from matplotlib.pyplot import *

# Simulation  Parameters
m = 1060  # time  steps
n = 300  # Number of share price points

# Parameters  -- American  Put  Option
r = 0.06  # short  rate
sigma = 0.2  # volatility
t = 1.0  # time -to -maturity
smax = 80  # Maximum share price considered
smin = 0  # Minimum share price considered
e = 40  # exercise price
ds = (smax - smin) / n  # Price step
dt = t / m  # length  of time  interval

# Initializing the matrix of the option values: v is the European and vam is the American option
def initial_option(n, m, e):
    v = zeros((n, m), 'd')
    # expiry: V(S, T) = max(e - S, 0)
    v[:, 0] = fmax(e - (smin + arange(n) * ds ), zeros(n))
    # V(S, t) =  Ee ^ (-r(T - t)) as S -> infinity.
    #v[0, 1: m] =  e * exp(-r * arange(1, m) * dt)
    return v

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def pricing_options(n, m, smin, ds, e, d):
    v = initial_option(n, m, e)
    vam = initial_option(n, m, e)
    # Determining the matrix coefficients of the explicit algorithm, according to BS model
    aa = 0.5 * dt * (sigma * sigma * multiply(arange(0, n - 2), arange(0, n - 2)) - (r - d) * arange(0, n - 2)).T
    bb = 1 - dt * (sigma * sigma * multiply(arange(0, n - 2), arange(0, n - 2)) + r).T
    cc = 0.5 * dt * (sigma * sigma * multiply(arange(0, n - 2), arange(0, n - 2)) + (r - d) * arange(0, n - 2)).T
    # Implementing the explicit algorithm
    for i in range(1, m, 1):
        p =multiply(bb, v[1: n - 1, i - 1]) + multiply(cc, v[2: n, i - 1]) + multiply(aa, v[0: n - 2, i - 1])
        #v[1: n - 1, i] = truncate(multiply(bb, v[1: n - 1, i - 1]) + multiply(cc, v[2: n, i - 1]) + multiply(aa, v[0: n - 2, i - 1]), 4)
        v[1: n - 1, i] = multiply(bb, v[1: n - 1, i - 1]) + multiply(cc, v[2: n, i - 1]) + multiply(aa, v[0: n - 2, i - 1])
        # Checks if early exercise is better for the American Option
        vam[1: n - 1, i] = fmax(
            multiply(bb, vam[1: n - 1, i - 1]) + multiply(cc, vam[2: n, i - 1]) + multiply(aa, vam[0: n - 2, i - 1]),
            vam[1: n - 1, 0])
    # Reversal of the time components in the matrix as the solution of the Black Scholes equation was performed
    # backwards
    v = fliplr(v)
    vam = fliplr(vam)
    # Compares the value today of the European(blue) and American(red) Calls, V(S, t), as a function of S.
    plot(smin + multiply(ds, arange(0, n - 1)), vam[0:n - 1, 0], 'r', smin + multiply(ds, arange(0, n - 1)), v[0:n - 1, 0], 'b')
    xlabel('S');
    ylabel('V(S,t)');
    title('European (blue) and American (red) Call Options');
    savefig("first.png")
    return vam

def search_by_init_value_of_S (S, umin, ds, vam, n):
    diff_old = 1000
    for i in range (0, n-1, 1):
        S_cur = umin+ds*i
        diff_new = abs(S_cur - S)
        if diff_new > diff_old:
            return vam[i-1,0]
        diff_old = diff_new

d= 0
vam = pricing_options(n, m, smin, ds, e, d)
S_0 = 40
print(search_by_init_value_of_S(S_0, smin, ds, vam, n))
