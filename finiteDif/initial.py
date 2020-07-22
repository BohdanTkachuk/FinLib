from numpy.core._multiarray_umath import fmax
from numpy.ma import *

def initial_option(n, r, m, e, umin, dt, ds, alpha, sigma):
    v = zeros((n, m), 'd')
    # expiry: V(S, T) = exp(-Alpha * x) max(K - exp(x), 0)
    v[:, 0] = multiply(exp(-alpha * (umin + arange(n) * ds)), fmax(e - exp(umin + arange(n) * ds), zeros(n)))
    # boundary conditions: for x=umin V(umin, tau) = K exp(-2r tau / sigma^2) for x = umax V(umax, tau) = 0, that was defined before
    v[0, 1:m] = e * exp(-r * 2 * arange(0, m-1) * dt / sigma ** 2)
    return v
