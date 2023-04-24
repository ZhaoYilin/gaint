import scipy.special as sc
from math import exp, gamma, sqrt

def boys(N,x):
    """Boys function for the calculation of coulombic integrals.

    Parameters
    ----------
    N : int
        Order of boys function

    x : float
        Varible for boys function.

    Returns
    -------
    result : float
        The boys function f_{N}(x).
    """
    result =  sc.hyp1f1(N+0.5,N+1.5,-x)/(2.0*N+1.0)
    return result

def boys_recursion(N, x, f_N):
    """Returns the answer to the boys function f_{N - 1}(x) using the
    answer for the boys function f_{N}(x).

    Parameters
    ----------
    N : int
        Order of boys function

    x : float
        Varible for boys function.

    f_N : float 
        Answer for the boys function f_{N}(x).

    Returns
    -------
    result : float
        The boys function f_{N - 1}(x).
    """
    result =  (exp(-x) + 2 * x * f_N) / (2 * N - 1)
    return result
