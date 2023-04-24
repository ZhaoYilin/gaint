import numpy as np
from gaint.boys import boys

def R3d(t,u,v,n,p,PCx,PCy,PCz,RPC):
    """ Returns the Coulomb auxiliary Hermite integrals

    Parameters
    ----------
    t,u,v: int   
        order of Coulomb Hermite derivative in x,y,z

    n : int
        order of Boys function

    PCx,y,z: 
        Cartesian vector distance between Gaussian
        composite center P and nuclear center C

    RPC: float     
        Distance between P and C
    Returns
    -------
    result : float
        Hermite integral.
    """
    T = p*RPC*RPC
    result = 0.0
    if t == u == v == 0:
        result += np.power(-2*p,n)*boys(n,T)
    elif t == u == 0:
        if v > 1:
            result += (v-1)*R3d(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)
        result += PCz*R3d(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
    elif t == 0:
        if u > 1:
            result += (u-1)*R3d(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)
        result += PCy*R3d(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
    else:
        if t > 1:
            result += (t-1)*R3d(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)
        result += PCx*R3d(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    return result
