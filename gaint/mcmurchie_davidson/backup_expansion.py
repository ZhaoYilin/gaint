import numpy as np

def E1d(i,j,t,A,B,a,b):
    """Recursive definition of Hermite Gaussian coefficients.

    Parameters
    ----------
    a : float 
        Primitive Cartesian Gaussian exponent.

    b : float 
        Primitive Cartesian Gaussian exponent.

    i : int
        Angular momentum quantum number.

    j : int
        Angular momentum quantum number.

    t : int
        Number nodes in Hermite Gaussian.

    A : float
        1-dimensional coordinate.

    B : float
        1-dimensional coordinate.

    Returns
    -------
    result : float
        The non-normalizecd overlap interals in one dimension.
    """


    ''' Recursive definition of Hermite Gaussian coefficients.
        Returns a float.
        a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
        i,j: orbital angular momentum number on Gaussian 'a' and 'b'
        Qx: distance between origins of Gaussian 'a' and 'b'
    '''
    p = a + b
    q = a*b/p
    AB = A-B
    if (t < 0) or (t > (i + j)):
        # out of bounds for t
        return 0.0
    elif i == j == t == 0:
        # base case
        return np.exp(-q*AB*AB) # K_AB
    elif j == 0:
        # decrement index i
        return (1/(2*p))*E1d(i-1,j,t-1,A,B,a,b) - \
               (q*AB/a)*E1d(i-1,j,t,A,B,a,b)    + \
               (t+1)*E1d(i-1,j,t+1,A,B,a,b)
    else:
        # decrement index j
        return (1/(2*p))*E1d(i,j-1,t-1,A,B,a,b) + \
               (q*AB/b)*E1d(i,j-1,t,A,B,a,b)    + \
               (t+1)*E1d(i,j-1,t+1,A,B,a,b)
