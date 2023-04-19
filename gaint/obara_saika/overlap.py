import numpy as np
from gaint.gauss import PrimitiveGaussian

class Overlap(object):
    """The Obara-Saika scheme for three-dimensional kinetic energy integral over
    primitive Gaussian orbitals.

    Parameters
    ----------
    a : float 
        Gaussian exponent facotr.

    b : float 
        Gaussian exponent facotr.

    i : int
        Angular momentum quantum number.

    j : int
        Angular momentum quantum number.

    A : float
        Coordinate in on direction.

    B : float
        Coordinate in on direction.

    Returns
    -------
    result : float
        The non-normalizecd kinetic interals in one dimension.
    """
    def __init__(self):
        self.p = 0
        self.mu = 0
        self.P = ()

    def __call__(self, pga, pgb):
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
        result = 1
        for r in range(3):
            result *= self.S1d(r, pga, pgb)
        return result

    def S1d(self, r, pga, pgb):
        a = pga.exponent
        b = pgb.exponent
        p = a + b
        mu = (a*b)/(a+b)

        A = np.array(pga.origin)
        B = np.array(pgb.origin)
        XAB = A-B

        if pga.shell[r] > 0:
            return self.recursive(r, *self.gaussian_factory(r, pga, pgb))
        elif pgb.shell[r] > 0:
            return self.recursive(r, *self.gaussian_factory(r, pgb, pga))
        else:
            # Starting from the spherical Gaussians.
            S00 = np.power(np.pi/p,0.5)*np.exp(-mu*XAB[r]**2)
            return S00


    def recursive(self, r, pga, pgb, pga_1, pga_2, pgb_1):
        term1 = term2 = term3 = 0

        a = pga.exponent
        b = pgb.exponent
        p = a + b
        mu = (a*b)/(a+b)

        A = np.array(pga.origin)
        B = np.array(pgb.origin)
        P = (a*A+b*B)/p
        XPA = P-A

        if XPA[r] != 0:
            term1 = XPA[r] * self.S1d(r, pga_1, pgb)
        if pga_1.shell[r] >= 0:
            term2 = pga_1.shell[r] * (1 / (2 * p)) * self.S1d(r, pga_2, pgb)
        if pgb.shell[r] >= 0:
            term3 = pgb.shell[r] * (1 / (2 * p)) * self.S1d(r, pga_1, pgb_1)
        return term1 + term2 + term3

    def gaussian_factory(self, r, pga, pgb):
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
    S = Overlap()
    s17 = S(pg1,pg2)
    print(np.isclose(s17,-0.0000888019))
