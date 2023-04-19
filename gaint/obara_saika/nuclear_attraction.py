import numpy as np
from gaint.boys import *
from gaint.gauss import PrimitiveGaussian

class NuclearAttraction(object):
    """The Obara-Saika scheme for three-dimensional nuclear attraction integral over
    primitive Gaussian orbitals.

    Attributes
    ----------
    p : float 
        The total exponent.

    mu : float
        The reduced exponent.

    P : List[float,float,float] 
        The centre of charge coordinate.

    C : List[float,float,float] 
        The coordinate of given nuclei.

    Kab : float
        The pre-exponential factor.

    Methods
    -------
    __init__(self)
        Initialize the instance.
    """
    def __init__(self):
        """Initialize the instance.
        """
        self.p = 0
        self.mu = 0
        self.Kab = 0
        self.P = []
        self.C = []
        self.boys_dict = {}

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
        l_total = sum(pga.shell) + sum(pgb.shell)

        a = pga.exponent
        b = pgb.exponent
        p = a + b
        mu = (a*b)/(a+b)

        A = np.array(pga.origin)
        B = np.array(pgb.origin)
        P = (a*A+b*B)/p

        RAB = np.linalg.norm(A-B)
        RPA = np.linalg.norm(P-A)
        RPB = np.linalg.norm(P-B)
        RPC = np.linalg.norm(P-C)

        Kab = exp(-mu*RAB**2)

        self.p = p
        self.mu = mu
        self.P = P
        self.C = C
        self.Kab = Kab

        # Build boys function F_{N}(x)
        N = l_total
        x = p*RPC**2
        boys_pre_factor = (2*np.pi)/p*Kab
        boys_function = boys(l_total, x)
        Theta_N_000000 = boys_pre_factor * boys_function
        self.boys_dict = {l_total: Theta_N_000000}

        while N >= 1:
            boys_function = boys_recursion(N, x, boys_function)
            N -= 1
            Theta_N_000000 = boys_pre_factor * boys_function
            self.boys_dict[N] = Theta_N_000000

        result = self.V(0, pga, pgb)
        return result

    def V(self, N, pga, pgb):
        """Evaluates nuclear attraction integral over two primitive gaussian orbitals.

        Parameters
        ----------
        N : int
            Order of the boys function F_{N}(x).

        pga : PrimitiveGaussian
            The first primitive gaussian orbital.

        pgb : PrimitiveGaussian
            The second primitive gaussian orbital.

        Return
        ------
        vlaue : float
            Integral value.
        """
        if pga.shell[0] > 0:
            return self.recursive(0, N, *self.gaussian_factory(0, pga, pgb))
        elif pga.shell[1] > 0:
            return self.recursive(1, N, *self.gaussian_factory(1, pga, pgb))
        elif pga.shell[2] > 0:
            return self.recursive(2, N, *self.gaussian_factory(2, pga, pgb))
        elif pgb.shell[0] > 0:
            return self.recursive(0, N, *self.gaussian_factory(0, pgb, pga))
        elif pgb.shell[1] > 0:
            return self.recursive(1, N, *self.gaussian_factory(1, pgb, pga))
        elif pgb.shell[2] > 0:
            result =  self.recursive(2, N, *self.gaussian_factory(2, pgb, pga))
            return result
        else:
            return self.boys_dict[N]

    def recursive(self, r, N, pga, pgb, pga_1, pga_2, pgb_1):
        """Evaluates nuclear attraction integral over two primitive gaussian orbitals.

        Parameters
        ----------
        r : int
            Cartesian index 0, 1, 2. 

        N : int
            Order of the boys function F_{N}(x).

        pga_1 : PrimitiveGaussian
            The primitive gaussian orbital.

        pgb : PrimitiveGaussian
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
        term1 = term2 = term3 = term4 = term5 = term6 = 0

        a = pga.exponent
        b = pgb.exponent
        p = a+b

        A = np.array(pga.origin)
        B = np.array(pgb.origin)
        P = (a*A+b*B)/p
        C = self.C

        XPA = np.array(P) - np.array(A)
        XPC = np.array(P) - np.array(C)

        if np.array_equal(P,A) is False:
            term1 = XPA[r] * self.V(N, pga_1, pgb)
        if pga_1.shell[r] > 0:
            term2 = pga_1.shell[r] * (1 / (2 * p)) * self.V(N, pga_2, pgb)
        if pgb.shell[r] > 0:
            term3 = pgb.shell[r] * (1 / (2 * p)) * self.V(N, pga_1, pgb_1)
        if np.array_equal(P,C) is False:
            term4 = XPC[r] * self.V(N+1, pga_1, pgb)
        if pga_1.shell[r] > 0:
            term5 = pga_1.shell[r] * (1 / (2 * p)) * self.V(N+1, pga_2, pgb)
        if pgb.shell[r] > 0:
            term6 = pgb.shell[r] * (1 / (2 * p)) * self.V(N+1, pga_1, pgb_1)

        result = term1+term2+term3-term4-term5-term6
        return result

    def gaussian_factory(self, r, pga, pgb):
        """Evaluates nuclear attraction integral over two primitive gaussian orbitals.

        Parameters
        ----------
        r : int
            Cartesian index 0, 1, 2. 

        N : int
            Order of the boys function F_{N}(x).

        pga : PrimitiveGaussian
            The primitive gaussian orbital.

        pgb : PrimitiveGaussian
            The primitive gaussian orbital.

        Return
        ------
        result : Tuple(pg,pg,pg,pg)
            Tuple of 4 PrimitiveGaussian orbital instance. 
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
    V = NuclearAttraction()
    v17 = V(pg1,pg2,FCenter[0])
    print(np.isclose(v17,-0.0000854386))
