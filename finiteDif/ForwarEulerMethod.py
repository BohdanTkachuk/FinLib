from numpy import fliplr
from numpy.core._multiarray_umath import fmax
from numpy.ma import multiply
from finiteDif.initial import initial_option

def heat_equation_FE(n, m, e, umin, umax, theta, alpha, sigma,r):
    dx = (umax - umin) / n  # Price step
    dt = theta / m  # time step
    vam = initial_option(n, r, m, e, umin, dt, dx, alpha, sigma)
    F = dt / dx ** 2
    if F>= 0.5:
        print("unstable!!!")
    # Implementing the explicit algorithm
    for i in range(1, m, 1):
        # Checks if early exercise is better for the American Option
        vam[1: n - 1, i] = fmax(
            vam[1: n - 1, i - 1] + multiply(F, vam[2: n, i - 1] + vam[0: n - 2, i - 1] - 2 * vam[1: n - 1, i - 1]),
            vam[1: n - 1, 0])
    # Reversal of the time components in the matrix as the solution of the Black Scholes equation was performed
    # backwards
    vam = fliplr(vam)
    return vam