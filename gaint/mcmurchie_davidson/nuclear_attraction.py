import numpy as np

class NuclearAttraction(object):
    """The McMurchie Davidson scheme for overlap integral over primitive Gaussian orbitals.

    Attributes
    ----------
    E1d : function
        One dimensional overlap function.

    R3d : function
        One dimensional overlap function.

    Methods
    -------
    __init__(self)
        Initialize the instance.
    """
    def __init__(self):
        """Initialize the instance.
        """
        from gaint.mcmurchie_davidson.expansion_coefficient import E1d
        from gaint.mcmurchie_davidson.hermite_coulomb import R3d
        self.E1d = E1d
        self.R3d = R3d

    def __call__(self, pga, pgb, C):
        """Evaluates nuclear attraction integral over two primitive gaussian orbitals.

        Parameters
        ----------
        pga: PrimitiveGaussian
            The first primitive gaussian orbital.

        pgb: PrimitiveGaussian
            The second primitive gaussian orbital.
    
        C: List[float,float,float]
            Coordinate of nuclei.

        Return
        ------
        result : float
            Integral value.
        """
        a = pga.exponent
        b = pgb.exponent
        p = a + b

        mu = (a*b)/(a+b)
        i,k,m = pga.shell
        j,l,n = pgb.shell

        # P the centre-of-charge coordinate
        A = np.array(pga.origin)
        B = np.array(pgb.origin)
        P = (a*A + b*B)/p
        XPC = P-C
        RPC = np.linalg.norm(P-C)

        result = 0.0
        for t in range(i+j+1):
            Eijt = self.E1d(i,j,t,A[0],B[0],a,b)
            for u in range(k+l+1):
                Eklu = self.E1d(k,l,u,A[1],B[1],a,b)
                for v in range(m+n+1):
                    Emnv = self.E1d(m,n,v,A[2],B[2],a,b)
                    Rtuv = self.R3d(t,u,v,0,p,XPC[0],XPC[1],XPC[2],RPC)
                    result += Eijt*Eklu*Emnv*Rtuv
        result *= 2*np.pi/p
        return result

if __name__ == '__main__':
    # Coordinate of H2O molecule
    H2O = [[0., 1.43233673, -0.96104039],
    [0., -1.43233673, -0.96104039],
    [0., 0., 0.24026010]]

    # Orbital exponents
    OrbCoeff = np.array([[3.425250914, 0.6239137298, 0.168855404],
    [3.425250914, 0.6239137298, 0.168855404],
    [130.7093214, 23.80886605, 6.443608313],
    [5.033151319, 1.169596125, 0.38038896],
    [5.033151319, 1.169596125, 0.38038896],
    [5.033151319, 1.169596125, 0.38038896],
    [5.033151319, 1.169596125, 0.38038896]])

    # H1s, H2s, O1s, O2s, O2px , O2py, O2p
    FCenter = [H2O[0], H2O[1], H2O[2], H2O[2], H2O[2], H2O[2], H2O[2]]
    CartAng = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [1, 0, 0], [0, 1, 0], [0, 0, 1]]

    from gaint.gauss import PrimitiveGaussian
    pg1 = PrimitiveGaussian(1.0,FCenter[0],CartAng[0],OrbCoeff[0,0])
    pg2 = PrimitiveGaussian(1.0,FCenter[6],CartAng[6],OrbCoeff[6,0])
    V = NuclearAttraction()
    v17 = V(pg1,pg2,FCenter[0])
    print(np.isclose(v17,-0.0000854386))
