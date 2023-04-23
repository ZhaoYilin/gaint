import numpy as np

from gaint.gauss import PrimitiveGaussian

class Kinetic(object):
    """The Obara-Saika scheme kinetic energy integral over primitive Gaussian orbitals.

    Attributes
    ----------
    p : float 
        The total exponent.

    mu : float
        The reduced exponent.

    P : List[float,float,float] 
        The centre of charge coordinate.
    """
    def __init__(self):
        """Initialize the instance.
        """
        from gaint.obara_saika.overlap import Overlap
        self.p = 0
        self.mu = 0
        self.P = ()
        self.overlap = Overlap()

    def __call__(self, pga, pgb):
        """Evaluates kinetic energy integral over two primitive gaussian orbitals.

        Parameters
        ----------
        pga: PrimitiveGaussian
            The first primitive gaussian orbital.

        pgb: PrimitiveGaussian
            The second primitive gaussian orbital.

        Return
        ------
        result : float
            Integral value.
        """
        Sij = self.overlap.S1d(0,pga,pgb)
        Skl = self.overlap.S1d(1,pga,pgb)
        Smn = self.overlap.S1d(2,pga,pgb)

        Tij = self.T1d(0,pga,pgb)
        Tkl = self.T1d(1,pga,pgb)
        Tmn = self.T1d(2,pga,pgb)

        Tab = Tij*Skl*Smn+Sij*Tkl*Smn+Sij*Skl*Tmn
        return Tab

    def T1d(self, r, pga, pgb):
        """Evaluates one dimensional kinetic energy integral over two primitive gaussian orbitals.

        Parameters
        ----------
        r: int
            The spatial index.

        pga: PrimitiveGaussian
            The first primitive gaussian orbital.

        pgb: PrimitiveGaussian
            The second primitive gaussian orbital.
        """
        a = pga.exponent
        b = pgb.exponent
        p = a + b
        mu = (a*b)/(a+b)

        A = np.array(pga.origin)
        B = np.array(pgb.origin)
        P = (a*A+b*B)/p
        XAB = A-B
        XPA = P-A

        if pga.shell[r] > 0:
            return self.recursive(r, *self.gaussian_factory(r, pga, pgb))
        elif pgb.shell[r] > 0:
            return self.recursive(r, *self.gaussian_factory(r, pgb, pga))
        else:
            # Starting from the spherical Gaussians.
            S00 = np.power(np.pi/p,0.5)*np.exp(-mu*XAB[r]**2)
            T00 = (a-2*a**2*(XPA[r]**2+1./(2*p)))*S00
            return T00


    def recursive(self, r, pga, pgb, pga_1, pga_2, pgb_1):
        """Run the recurrence.

        Parameters
        ----------
        r : int
            Cartesian index 0, 1, 2. 

        pga : PrimitiveGaussian
            The primitive gaussian orbital.

        pgb : PrimitiveGaussian
            The primitive gaussian orbital.

        pga_1 : PrimitiveGaussian
            The primitive gaussian orbital.

        pga_2 : PrimitiveGaussian
            The primitive gaussian orbital.

        pgb_1 : PrimitiveGaussian
            The primitive gaussian orbital.

        Return
        ------
        result : float
            Integral value.
        """
        term1 = term2 = term3 = term4 = term5 = 0

        a = pga.exponent
        b = pgb.exponent
        p = a + b
        mu = (a*b)/(a+b)

        A = np.array(pga.origin)
        B = np.array(pgb.origin)
        P = (a*A+b*B)/p
        XPA = P-A

        if XPA[r] != 0:
            term1 = XPA[r] * self.T1d(r, pga_1, pgb)
        if pga_1.shell[r] >= 0:
            term2 = pga_1.shell[r] * (1 / (2 * p)) * self.T1d(r, pga_2, pgb)
        if pgb.shell[r] >= 0:
            term3 = pgb.shell[r] * (1 / (2 * p)) * self.T1d(r, pga_1, pgb_1)
        term4 =  (2*a*b) / p * self.overlap.S1d(r, pga, pgb)
        if pga_1.shell[r] >= 0:
            term5 = pgb.shell[r] * (b / p) * self.overlap.S1d(r, pga_2, pgb)
        return term1 + term2 + term3 + term4 - term5

    def gaussian_factory(self, r, pga, pgb):
        """Generate all gaussian orbitals in the Obara-Saikai recurrence equation.

        Parameters
        ----------
        r : int
            Cartesian index 0, 1, 2. 

        pga : PrimitiveGaussian
            The primitive gaussian orbital.

        pgb : PrimitiveGaussian
            The primitive gaussian orbital.

        Return
        ------
        result : Tuple(pg, pg, pg, pg, pg)
            Tuple of 5 PrimitiveGaussian orbital instance. 
        """
        ca = pga.coefficient
        cb = pgb.coefficient

        a = pga.exponent
        b = pgb.exponent

        A = pga.origin
        B = pgb.origin

        i,k,m = pga.shell
        j,l,n = pgb.shell

        if r == 0:
            pga_i_1 = PrimitiveGaussian(ca, A, (i - 1, k, m), a)
            pga_i_2 = PrimitiveGaussian(ca, A, (i - 2, k, m), a)
            pgb_j_1 = PrimitiveGaussian(cb, B, (j - 1, l, n), b)
            return pga, pgb, pga_i_1, pga_i_2, pgb_j_1
        elif r == 1:
            pga_k_1 = PrimitiveGaussian(ca, A, (i, k - 1, m), a)
            pga_k_2 = PrimitiveGaussian(ca, A, (i, k - 2, m), a)
            pgb_l_1 = PrimitiveGaussian(cb, B, (j, l - 1, n), b)
            return pga, pgb, pga_k_1, pga_k_2, pgb_l_1
        elif r == 2:
            pga_m_1 = PrimitiveGaussian(ca, A, (i, k, m - 1), a)
            pga_m_2 = PrimitiveGaussian(ca, A, (i, k, m - 2), a)
            pgb_n_1 = PrimitiveGaussian(cb, B, (j, l, n - 1), b)
            return pga, pgb, pga_m_1, pga_m_2, pgb_n_1


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

    pg1 = PrimitiveGaussian(1.0,FCenter[0],CartAng[0],OrbCoeff[0,0])
    pg2 = PrimitiveGaussian(1.0,FCenter[6],CartAng[6],OrbCoeff[6,0])
    T = Kinetic()
    t17 = T(pg1,pg2)
    print(np.isclose(t17,0.00167343))
