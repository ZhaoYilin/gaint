import numpy as np
from gaint.mcmurchie_davidson.expansion_coefficients import E1d
from gaint.mcmurchie_davidson.hermite_coulomb import R3d

def V3d(a, b, ikm, jln, A, B, C):
    ''' Evaluates kinetic energy integral between two Gaussians
         Returns a float.
         a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
         b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
         lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
               for Gaussian 'a'
         lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
         A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
         B:    list containing origin of Gaussian 'b'
         C:    list containing origin of nuclear center 'C'
     '''
    i,k,m = ikm
    j,l,n = jln
    # p the total exponent
    p = a + b
    # P the centre-of-charge coordinate
    A = np.array(A)
    B = np.array(B)
    P = (a*A + b*B)/p
    XPC = P-C
    RPC = np.linalg.norm(P-C)

    val = 0.0
    for t in range(i+j+1):
        Eijt = E1d(i,j,t,A[0],B[0],a,b)
        for u in range(k+l+1):
            Eklu = E1d(k,l,u,A[1],B[1],a,b)
            for v in range(m+n+1):
                Emnv = E1d(m,n,v,A[2],B[2],a,b)
                Rtuv = R3d(t,u,v,0,p,XPC[0],XPC[1],XPC[2],RPC)
                val += Eijt*Eklu*Emnv*Rtuv
    val *= 2*np.pi/p
    return val

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

    # H1s, H2s, O1s, O2s, O2px , O2py, O2pz
    FCenter = [H2O[0], H2O[1], H2O[2], H2O[2], H2O[2], H2O[2], H2O[2]]
    CartAng = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [1, 0, 0], [0, 1, 0], [0, 0, 1]]

    chi_17 = V3d(OrbCoeff[0,0], OrbCoeff[6,0], CartAng[0], CartAng[6], FCenter[0], FCenter[6],FCenter[0])
    print(chi_17)
    print(np.isclose(chi_17,-0.0000854386))

