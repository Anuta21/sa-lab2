from numpy.polynomial import Polynomial 

def chebyshev(degree):
    result = [Polynomial([-1, 2]), Polynomial([1])]
    for i in range(degree):
        result.append(Polynomial([-2, 4])*result[-1] - result[-2])
    del result[0]
    return result

def legendre(degree):
    result = [Polynomial([1])]
    for i in range(degree):
        if i == 0:
            result.append(Polynomial([-1, 2]))
            continue
        result.append((Polynomial([-2*i - 1, 4*i + 2])*result[-1] - i * result[-2]) / (i + 1))
    return result

def laguerre(degree):
    result = [Polynomial([1])]
    for i in range(degree):
        if i == 0:
            result.append(Polynomial([1, -1]))
            continue
        result.append(Polynomial([2*i + 1, -1])*result[-1] - i * i * result[-2])
    return result

def hermite(degree):
    result = [Polynomial([0]), Polynomial([1])]
    for i in range(degree):
        result.append(Polynomial([0,2])*result[-1] - 2 * i * result[-2])
    del result[0]
    return result