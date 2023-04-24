import numpy as np

class Kinetic(object):
    """The McMurchie Davidson scheme for overlap integral over primitive Gaussian orbitals.

    Attributes
    ----------
    S1d : function
        One dimensional overlap function.
    """
    def __init__(self):
        """Initialize the instance.
        """
        from gaint.mcmurchie_davidson.overlap import Overlap
        overlap = Overlap()
        self.S1d = overlap.S1d

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
        Sij = self.S1d(0,pga,pgb)
        Skl = self.S1d(1,pga,pgb)
        Smn = self.S1d(2,pga,pgb)

        Tij = self.T1d(0,pga,pgb)
        Tkl = self.T1d(1,pga,pgb)
        Tmn = self.T1d(2,pga,pgb)

        Tab = Tij*Skl*Smn+Sij*Tkl*Smn+Sij*Skl*Tmn
        return Tab

    def T1d(self, r, pga, pgb):
        """The McMurchie Davidson scheme for one-dimensional kinetic energy integral over
        primitive Gaussian orbitals.

        Parameters
        ----------
        pga: PrimitiveGaussian
            The first primitive gaussian orbital.

        pgb: PrimitiveGaussian
            The second primitive gaussian orbital.

        Returns
        -------
        result : float
            The non-normalizecd kinetic interals in one dimension.
        """
        cb = pgb.coefficient
        B = pgb.origin
        b = pgb.exponent
        j = pgb.shell[0]
        l = pgb.shell[1]
        n = pgb.shell[2]

        if r==0:
            pgb_j_p2 = PrimitiveGaussian(cb, B, (j + 2, l, n), b)
            pgb_j_m2 = PrimitiveGaussian(cb, B, (j - 2, l, n), b)
            S1 = self.S1d(0,pga,pgb_j_p2)
            S2 = self.S1d(0,pga,pgb)
            S3 = self.S1d(0,pga,pgb_j_m2)
        if r==1:
            pgb_l_p2 = PrimitiveGaussian(cb, B, (j, l+2, n), b)
            pgb_l_m2 = PrimitiveGaussian(cb, B, (j, l-2, n), b)
            S1 = self.S1d(1,pga,pgb_l_p2)
            S2 = self.S1d(1,pga,pgb)
            S3 = self.S1d(1,pga,pgb_l_m2)
        if r==2:
            pgb_n_p2 = PrimitiveGaussian(cb, B, (j, l, n+2), b)
            pgb_n_m2 = PrimitiveGaussian(cb, B, (j, l, n-2), b)
            S1 = self.S1d(2,pga,pgb_n_p2)
            S2 = self.S1d(2,pga,pgb)
            S3 = self.S1d(2,pga,pgb_n_m2)

        result = -2*b**2*S1+b*(2*pgb.shell[r]+1)*S2-0.5*pgb.shell[r]*(pgb.shell[r]-1)*S3
        return result

if __name__ == '__main__':
    # Coordinate of H2O molecule
    R = [[0., 1.43233673, -0.96104039],
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
    FCenter = [R[0], R[1], R[2], R[2], R[2], R[2], R[2]]
    CartAng = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [1, 0, 0], [0, 1, 0], [0, 0, 1]]

    from gaint.gauss import PrimitiveGaussian
    pg1 = PrimitiveGaussian(1.0,FCenter[0],CartAng[0],OrbCoeff[0,0])
    pg2 = PrimitiveGaussian(1.0,FCenter[6],CartAng[6],OrbCoeff[6,0])
    T = Kinetic()
    t17 = T(pg1,pg2)
    print(np.isclose(t17,0.00167343))

