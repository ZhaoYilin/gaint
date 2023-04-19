import numpy as np

class Overlap(object):
    def __init__(self,scheme='md'):
        if scheme == 'md':
            from gaint.mcmurchie_davidson.overlap import S3d
        elif scheme == 'os':
            from gaint.obara_saika.overlap import S3d
        elif scheme == 'direct':
            from gaint.direct.overlap import S3d
        else:
            raise Exception("Option not in the scheme list.")
        self.scheme = scheme
        self.integral = S3d

    def __call__(self, g1, g2):
        a = g1.exponent
        b = g2.exponent
        ikm = g1.shell 
        jln = g2.shell 
        A = g1.origin
        B = g2.origin
        return self.integral(a, b, ikm, jln, A, B)

class Kinetic(object):
    def __init__(self,scheme='md'):
        if scheme == 'md':
            from gaint.mcmurchie_davidson.kinetic import T3d
        elif scheme == 'os':
            from gaint.obara_saika.kinetic import T3d
        elif scheme == 'direct':
            from gaint.direct.kinetic import T3d
        else:
            raise Exception("Option not in the scheme list.")
        self.scheme = scheme
        self.integral = T3d

    def __call__(self, g1, g2):
        a = g1.exponent
        b = g2.exponent
        ikm = g1.shell 
        jln = g2.shell 
        A = g1.origin
        B = g2.origin
        return self.integral(a, b, ikm, jln, A, B)

class NuclearAttraction(object):
    def __init__(self,scheme='md'):
        if scheme == 'md':
            from gaint.mcmurchie_davidson.nuclear_attraction import V3d
        #elif scheme == 'os':
        #    from gaint.obara_saika.nuclear_attraction import V3d
        #elif scheme == 'direct':
        #    from gaint.direct.nuclear_attraction import S3d
        else:
            raise Exception("Option not in the scheme list.")
        self.scheme = scheme
        self.integral = V3d

    def __call__(self, g1, g2, C):
        a = g1.exponent
        b = g2.exponent
        ikm = g1.shell 
        jln = g2.shell 
        A = g1.origin
        B = g2.origin
        return self.integral(a, b, ikm, jln, A, B, C)

class ElectronRepulsion(object):
    def __init__(self,scheme='md'):
        if scheme == 'md':
            from gaint.mcmurchie_davidson.electron_repulsion import Eri3d
        #elif scheme == 'os':
        #    from gaint.obara_saika.kinetic import T3d
        #elif scheme == 'direct':
        #    from gaint.direct.kinetic import T3d
        else:
            raise Exception("Option not in the scheme list.")
        self.scheme = scheme
        self.integral = Eri3d

    def __call__(self, g1, g2, g3, g4):
        a = g1.exponent
        b = g2.exponent
        ikm = g1.shell 
        jln = g2.shell 
        A = g1.origin
        B = g2.origin
        return Eri3d(a, b, ikm, jln, A, B, C)


if __name__ == '__main__':
    from gaint.gauss import PrimitiveGaussian
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
    O_overlap = Overlap('direct')
    O_kinetic = Kinetic('md')
    O_nuclear_attraction = NuclearAttraction('md')
    o_17 = O_overlap(g1,g2)
    t_17 = O_kinetic(g1,g2)
    v_17 = O_nuclear_attraction(g1,g2,FCenter[0])
    print(np.isclose(o_17,-0.0000888019))
    print(np.isclose(t_17,0.00167343))
    print(np.isclose(v_17,-0.0000854386))
