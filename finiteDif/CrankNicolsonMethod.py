# Thete = 1/2 CN method
from numpy import fliplr
from numpy.core._multiarray_umath import fmax
from numpy.ma import *
from scipy import sparse
from scipy.sparse.linalg import spsolve

from finiteDif.initial import initial_option

def heat_equation_BE_sparse_matrix_CN(n, m, e, umin, umax, theta, alpha, sigma, Theta, r):

    dx = (umax - umin) / n  # Price step
    dt = theta / m  # time step

    vam = initial_option(n, r, m, e, umin, dt, dx, alpha, sigma)
    F = dt / dx ** 2
    # Representation of sparse matrix and right-hand side
    main = zeros(n)
    lower = zeros(n - 1)
    upper = zeros(n - 1)
    b = zeros(n)
    # Precompute sparse matrix
    main[:] = 1 + 2 * F * Theta
    lower[:] = -F * Theta  # 1
    upper[:] = -F * Theta  # 1
    # Insert boundary conditions
    main[0] = 1
    main[n - 1] = 1
    A = sparse.diags(
        diagonals=[main, lower, upper],
        offsets=[0, -1, 1], shape=(n, n),
        format='csr')
    # Implementing the explicit algorithm
    for i in range(1, m, 1):
        # compute b
        b[1:n - 2] = vam[1:n - 2, i - 1] + F * (1 - Theta) * (
                vam[2:n - 1, i - 1] - 2 * vam[1:n - 2, i - 1] + vam[0:n - 3, i - 1])
        # same here, need to review
        b[0] = vam[0, i]
        b[n - 1] = 0

        # u = linalg.inv(A).dot(b)
        u = spsolve(A, b)
        vam[1: n - 1, i] = fmax(u[1:n - 1], vam[1: n - 1, 0])
    # Reversal of the time components in the matrix as the solution of the Black Scholes equation was performed
    # backwards
    vam = fliplr(vam)
    return vam