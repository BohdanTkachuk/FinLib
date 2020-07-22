

import numpy
import quadpy

#x = lambda a : a ** 2

scheme = quadpy.c1.clenshaw_curtis(5)
#scheme.show()



scheme = quadpy.c1.gauss_legendre(5)
val = scheme.integrate(lambda x: x**2, [0.0, 3.0])
print("Gauss-Legendre ", val)
def funcTest():
    w = numpy.zeros((10,10))
    for i in range(0,10):
        for j in range(0,10):
            w[i][j] = i-j
    return w

test_list = [10]
test_list.append(15)
print(test_list[1])