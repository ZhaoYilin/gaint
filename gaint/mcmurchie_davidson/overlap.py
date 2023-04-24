import numpy as np

class Overlap(object):
    """The McMurchie Davidson scheme for overlap integral over primitive Gaussian orbitals.

    Attributes
    ----------
    E1d : function
        One dimensional expansion coefficient function.
    """
    def __init__(self):
        """Initialize the instance.
        """
        from gaint.mcmurchie_davidson.expansion_coefficient import E1d
        self.E1d = E1d

    def __call__(self, pga, pgb):
        """Evaluates overlap integral over two primitive gaussian orbitals.

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
        result = 1
        for r in range(3):
            result *= self.S1d(r, pga, pgb)
        return result

    def S1d(self, r, pga, pgb):
        """Evaluates one dimensional overlap integral over two primitive gaussian orbitals.

        Parameters
        ----------
        r : int
            Cartesian index 0, 1, 2. 

        pga: PrimitiveGaussian
            The first primitive gaussian orbital.

        pgb: PrimitiveGaussian
            The second primitive gaussian orbital.
        """
        i = pga.shell[r]
        j = pgb.shell[r]
        A = pga.origin[r]
        B = pgb.origin[r]
        a = pga.exponent
        b = pgb.exponent
        p = a + b 
        return np.sqrt(np.pi/p)*self.E1d(i,j,0,A,B,a,b)
       
if __name__ == '__main__':
    # Coordinate of H2O molecule
    H2O = [[0., 1.43233673, -0.96104039],
    [0., -1.43233673, -0.96104039],
    [0., 0., 0.24026010]]

    # Primitive contraction coefficients
    PrimCoeff = np.array([[0.1543289673, 0.5353281423, 0.4446345422],
    [0.1543289673, 0.5353281423, 0.4446345422],
    [0.1543289673, 0.5353281423, 0.4446345422],
    [-0.09996722919, 0.3995128261, 0.7001154689],
    [0.155916275, 0.6076837186, 0.3919573931],
    [0.155916275, 0.6076837186, 0.3919573931],
    [0.155916275, 0.6076837186, 0.3919573931]])

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
    pga = PrimitiveGaussian(PrimCoeff[0,0],FCenter[0],CartAng[0],OrbCoeff[0,0])
    pgb = PrimitiveGaussian(PrimCoeff[6,0],FCenter[6],CartAng[6],OrbCoeff[6,0])

    S = Overlap()
    s17 = S(pga,pgb)
    print(np.isclose(s17,-0.0000888019)) 
