import numpy as np
import scipy.special as sc
from math import exp, gamma, sqrt

def boys(n,t):
    """Boys function for the calculation of coulombic integrals.

    Parameters
    ----------
    n : int
        Order of boys function

    t : float
        Varible for boys function.

    Raises
    ------
    TypeError
        If boys function order is not an integer.

    ValueError
        If boys function order n is not a none negative number.
    """
    if not isinstance(n, int):
        raise TypeError("Boys function order n must be an integer")
    if n < 0:
        raise ValueError("Boys function order n must be a none negative number")    
    if not isinstance(t, float):
        raise TypeError("Boys function varible t must be integer or float")
    return sc.hyp1f1(n+0.5,n+1.5,-t)/(2.0*n+1.0)

def boys_recursion(N, x, f_N):
    """Returns the answer to the boys function f_{v - 1}(x) using the
    answer for the boys function f_{v}(x).

    Parameters
    ----------
    v : {int, float}
    x : {int, float}
    f_v : float

    Returns
    -------
    result : float
        The boys function f_{N - 1}(x).
    """
    result =  (exp(-x) + 2 * x * f_N) / (2 * N - 1)
    return result
