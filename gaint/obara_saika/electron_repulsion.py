import numpy as np
from gaint.boys import *
from gaint.gauss import PrimitiveGaussian


class ElectronRepulsion:
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
        self.alpha = 0
        self.R = []
        self.boys_dict = {}

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
        l_total = sum(pga.shell) + sum(pgb.shell) + sum(pgc.shell) + sum(pgd.shell)

        a = pga.exponent
        b = pgb.exponent
        c = pgc.exponent
        d = pgd.exponent
        p = a + b
        q = c + d
        mu = (a*b)/(a+b)
        nu = (c*d)/(c+d)
        alpha = (p * q) / (p + q)
        self.alpha = alpha

        A = np.array(pga.origin)
        B = np.array(pgb.origin)
        C = np.array(pgc.origin)
        D = np.array(pgd.origin)
        P = (a*A+b*B)/(a+b)
        Q = (c*C+d*D)/(c+d)
        R = (p*P+q*Q)/(p+q)
        self.R = R

        RAB = np.linalg.norm(A-B)
        RCD = np.linalg.norm(C-D)
        RPQ = np.linalg.norm(P-Q)

        # Build boys function F_{N}(x)
        Kab = np.exp(-mu*RAB**2)
        Kcd = np.exp(-nu*RCD**2)
        boys_pre_factor = (2*np.pi**(5/2))/(p*q*np.sqrt(p+q))*Kab*Kcd
        N = l_total
        x = alpha*RPQ**2
        boys_function = boys(l_total, x)
        Theta_N_0000_0000_0000 = boys_pre_factor * boys_function
        self.boys_dict = {l_total: Theta_N_0000_0000_0000}

        while N >= 1:
            boys_function = boys_recursion(N, x, boys_function)
            N -= 1
            Theta_N_0000_0000_0000 = boys_pre_factor * boys_function
            self.boys_dict[N] = Theta_N_0000_0000_0000
        result = self.Eri(0, pga, pgb, pgc, pgd)
        return result

    def Eri(self, N, pga, pgb, pgc, pgd):
        """Evaluates nuclear attraction integral over two primitive gaussian orbitals.

        Parameters
        ----------
        N : int
            Order of the boys function F_{N}(x).

        pga : PrimitiveGaussian
            The primitive gaussian orbital.

        pgb : PrimitiveGaussian
            The primitive gaussian orbital.

        pgc : PrimitiveGaussian
            The primitive gaussian orbital.

        pgd : PrimitiveGaussian
            The primitive gaussian orbital.

        Return
        ------
        vlaue : float
            Integral value.
        """
        if pga.shell[0] > 0:
            return self.recursive(0, N, *self.gaussian_factory(0, pga, pgb, pgc, pgd))
        elif pga.shell[1] > 0:
            return self.recursive(1, N, *self.gaussian_factory(1, pga, pgb, pgc, pgd))
        elif pga.shell[2] > 0:
            return self.recursive(2, N, *self.gaussian_factory(2, pga, pgb, pgc, pgd))
        elif pgb.shell[0] > 0:
            return self.recursive(0, N, *self.gaussian_factory(0, pgb, pga, pgd, pgc))
        elif pgb.shell[1] > 0:
            return self.recursive(1, N, *self.gaussian_factory(1, pgb, pga, pgd, pgc))
        elif pgb.shell[2] > 0:
            return self.recursive(2, N, *self.gaussian_factory(2, pgb, pga, pgd, pgc))
        elif pgc.shell[0] > 0:
            return self.recursive(0, N, *self.gaussian_factory(0, pgc, pgd, pga, pgb))
        elif pgc.shell[1] > 0:
            return self.recursive(1, N, *self.gaussian_factory(1, pgc, pgd, pga, pgb))
        elif pgc.shell[2] > 0:
            return self.recursive(2, N, *self.gaussian_factory(2, pgc, pgd, pga, pgb))
        elif pgd.shell[0] > 0:
            return self.recursive(0, N, *self.gaussian_factory(0, pgd, pgc, pgb, pga))
        elif pgd.shell[1] > 0:
            return self.recursive(1, N, *self.gaussian_factory(1, pgd, pgc, pgb, pga))
        elif pgd.shell[2] > 0:
            return self.recursive(2, N, *self.gaussian_factory(2, pgd, pgc, pgb, pga))
        else:
            return self.boys_dict[N]

    def recursive(self, r, N, pga, pgb, pgc, pgd, pga_1, pga_2, pgb_1, pgc_1, pgd_1):
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
        term1 = term2 = term3 = term4 = term5 = term6 = term7 = term8 = 0

        a = pga.exponent
        b = pgb.exponent
        c = pgc.exponent
        d = pgd.exponent
        p = a + b
        q = c + d
        alpha = (p*q)/(p+q)
        #alpha = self.alpha
        
        A = np.array(pga.origin)
        B = np.array(pgb.origin)
        C = np.array(pgc.origin)
        D = np.array(pgd.origin)
        P = (a*A+b*B)/(a+b)
        Q = (c*C+d*D)/(c+d)

        XPA = P - A
        XPQ = P - Q

        if XPA[r] != 0:
            term1 = XPA[r] * self.Eri(N, pga_1, pgb, pgc, pgd)
        if XPQ[r] != 0:
            term2 = alpha/p*XPQ[r] * self.Eri(N+1, pga_1, pgb, pgc, pgd)
        if pga_1.shell[r] > 0:
            term3 = pga_1.shell[r] * (1 / (2 * p)) * self.Eri(N, pga_2, pgb, pgc, pgd)
            term4 = pga_1.shell[r] * (alpha / (2 * p ** 2)) * self.Eri(N+1, pga_2, pgb, pgc, pgd)
        if pgb.shell[r] > 0:
            term5 = pgb.shell[r] * (1 / (2 * p)) * self.Eri(N, pga_1, pgb_1, pgc, pgd)
            term6 = pgb.shell[r] * (alpha / (2 * p ** 2)) * self.Eri(N+1, pga_1, pgb_1, pgc, pgd)
        if pgc.shell[r] > 0:
            term7 = pgc.shell[r] * (1 / (2 * (p + q))) * self.Eri(N+1, pga_1, pgb, pgc_1, pgd)
        if pgd.shell[r] > 0:
            term8 = pgd.shell[r] * (1 / (2 * (p + q))) * self.Eri(N+1, pga_1, pgb, pgc, pgd_1)

        return term1 - term2 + term3 - term4 + term5 - term6 + term7 + term8

    def gaussian_factory(self, r, pga, pgb, pgc, pgd):
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
        cc = pgc.coefficient
        cd = pgd.coefficient

        a = pga.exponent
        b = pgb.exponent
        c = pgc.exponent
        d = pgd.exponent

        A = pga.origin
        B = pgb.origin
        C = pgc.origin
        D = pgd.origin

        ix,iy,iz = pga.shell
        jx,jy,jz = pgb.shell
        kx,ky,kz = pgc.shell
        lx,ly,lz = pgd.shell

        if r == 0:
            pga_1 = PrimitiveGaussian(ca, A, (ix - 1, iy, iz), a)
            pga_2 = PrimitiveGaussian(ca, A, (ix - 2, iy, iz), a)
            pgb_1 = PrimitiveGaussian(cb, B, (jx - 1, jy, jz), b)
            pgc_1 = PrimitiveGaussian(cc, C, (kx - 1, ky, kz), c)
            pgd_1 = PrimitiveGaussian(cd, D, (lx - 1, ly, lz), d)
            return pga, pgb, pgc, pgd, pga_1, pga_2, pgb_1, pgc_1, pgd_1
        elif r == 1:
            pga_1 = PrimitiveGaussian(ca, A, (ix, iy-1, iz), a)
            pga_2 = PrimitiveGaussian(ca, A, (ix, iy-2, iz), a)
            pgb_1 = PrimitiveGaussian(cb, B, (jx, jy-1, jz), b)
            pgc_1 = PrimitiveGaussian(cc, C, (kx, ky-1, kz), c)
            pgd_1 = PrimitiveGaussian(cd, D, (lx, ly-1, lz), d)
            return pga, pgb, pgc, pgd, pga_1, pga_2, pgb_1, pgc_1, pgd_1
        elif r == 2:
            pga_1 = PrimitiveGaussian(ca, A, (ix, iy, iz-1), a)
            pga_2 = PrimitiveGaussian(ca, A, (ix, iy, iz-2), a)
            pgb_1 = PrimitiveGaussian(cb, B, (jx, jy, jz-1), b)
            pgc_1 = PrimitiveGaussian(cc, C, (kx, ky, kz-1), c)
            pgd_1 = PrimitiveGaussian(cd, D, (lx, ly, lz-1), d)
            return pga, pgb, pgc, pgd, pga_1, pga_2, pgb_1, pgc_1, pgd_1


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

    g1 = PrimitiveGaussian(1.0,FCenter[0],CartAng[0],OrbCoeff[0,0])
    g2 = PrimitiveGaussian(1.0,FCenter[6],CartAng[6],OrbCoeff[6,0])
    Eri = ElectronRepulsion()
    eri1717 = Eri(g1,g2,g1,g2)
    print(eri1717)
