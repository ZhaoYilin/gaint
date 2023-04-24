import numpy as np

class ElectronRepulsion:
    """The Obara-Saika scheme for three-dimensional nuclear attraction integral over
    primitive Gaussian orbitals.
    Attributes
    ----------

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

    def __call__(self, pga, pgb, pgc, pgd):
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
        c = pgc.exponent
        d = pgd.exponent
        p = a + b
        q = c + d
        mu = (a*b)/(a+b)
        nu = (c*d)/(c+d)
        alpha = (p * q) / (p + q)

        l1,m1,n1 = pga.shell
        l2,m2,n2 = pgb.shell
        l3,m3,n3 = pgc.shell
        l4,m4,n4 = pgd.shell

        A = np.array(pga.origin)
        B = np.array(pgb.origin)
        C = np.array(pgc.origin)
        D = np.array(pgd.origin)
        P = (a*A+b*B)/(a+b)
        Q = (c*C+d*D)/(c+d)
        R = (p*P+q*Q)/(p+q)
        XPQ = P-Q
        RPQ = np.linalg.norm(P-Q)

        result = 0.0
        for t in range(l1+l2+1):
            Et = self.E1d(l1,l2,t,A[0],B[0],a,b)
            for u in range(m1+m2+1):
                Eu = self.E1d(m1,m2,u,A[1],B[1],a,b)
                for v in range(n1+n2+1):
                    Ev = self.E1d(n1,n2,v,A[2],B[2],a,b)
                    Eab = Et*Eu*Ev
                    for tau in range(l3+l4+1):
                        Etau = self.E1d(l3,l4,tau,C[0],D[0],c,d)
                        for nu in range(m3+m4+1):
                            Enu = self.E1d(m3,m4,nu ,C[1],D[1],c,d)
                            for phi in range(n3+n4+1):
                                Ephi = self.E1d(n3,n4,phi,C[2],D[2],c,d)
                                Ecd = Etau*Enu*Ephi

                                R = self.R3d(t+tau,u+nu,v+phi,0,\
                                    alpha,XPQ[0],XPQ[1],XPQ[2],RPQ)

                                result += np.power(-1,tau+nu+phi)*Eab*Ecd*R

        result *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q))
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
    pga = PrimitiveGaussian(1.0,FCenter[0],CartAng[0],OrbCoeff[0,0])
    pgb = PrimitiveGaussian(1.0,FCenter[6],CartAng[6],OrbCoeff[6,0])
    Eri = ElectronRepulsion()
    eri1717 = Eri(pga,pgb,pga,pgb)
    print(np.allclose(eri1717,1.9060888184873294e-08))
