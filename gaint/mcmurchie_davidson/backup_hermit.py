import numpy as np
from gaint.boys import boys

def R3d(t,u,v,n,p,PCx,PCy,PCz,RPC):
    ''' Returns the Coulomb auxiliary Hermite integrals
        Returns a float.
        Arguments:
        t,u,v:   order of Coulomb Hermite derivative in x,y,z
                 (see defs in Helgaker and Taylor)
        n:       order of Boys function
        PCx,y,z: Cartesian vector distance between Gaussian
                 composite center P and nuclear center C
        RPC:     Distance between P and C
    '''
    T = p*RPC*RPC
    val = 0.0
    if t == u == v == 0:
        val += np.power(-2*p,n)*boys(n,T)
    elif t == u == 0:
        if v > 1:
            val += (v-1)*R3d(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)
        val += PCz*R3d(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
    elif t == 0:
        if u > 1:
            val += (u-1)*R3d(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCy*R3d(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
    else:
        if t > 1:
            val += (t-1)*R3d(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCx*R3d(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    return val
