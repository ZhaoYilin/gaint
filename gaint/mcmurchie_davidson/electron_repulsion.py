import numpy as np
from gaint.mcmurchie_davidson.expansion_coefficients import E1d
from gaint.mcmurchie_davidson.hermite_coulomb import R3d


def g3d(a,b,c,d,lmn1,lmn2,lmn3,lmn4,A,B,C,D):
    ''' Evaluates kinetic energy integral between two Gaussians
         Returns a float.
         a,b,c,d:   orbital exponent on Gaussian 'a','b','c','d'
         lmn1,lmn2
         lmn3,lmn4: int tuple containing orbital angular momentum
                    for Gaussian 'a','b','c','d', respectively
         A,B,C,D:   list containing origin of Gaussian 'a','b','c','d'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    l3,m3,n3 = lmn3
    l4,m4,n4 = lmn4
    p = a+b # composite exponent for P (from Gaussians 'a' and 'b')
    q = c+d # composite exponent for Q (from Gaussians 'c' and 'd')
    alpha = p*q/(p+q)

    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)
    P = (a*A + b*B)/p
    Q = (c*C + d*D)/q
    XPQ = P-Q
    RPQ = np.linalg.norm(P-Q)

    val = 0.0
    for t in range(l1+l2+1):
        Et = E1d(l1,l2,t,A[0],B[0],a,b)
        for u in range(m1+m2+1):
            Eu = E1d(m1,m2,u,A[1],B[1],a,b)
            for v in range(n1+n2+1):
                Ev = E1d(n1,n2,v,A[2],B[2],a,b)
                Eab = Et*Eu*Ev
                for tau in range(l3+l4+1):
                    Etau = E1d(l3,l4,tau,C[0],D[0],c,d)
                    for nu in range(m3+m4+1):
                        Enu = E1d(m3,m4,nu ,C[1],D[1],c,d)
                        for phi in range(n3+n4+1):
                            Ephi = E1d(n3,n4,phi,C[2],D[2],c,d)
                            Ecd = Etau*Enu*Ephi

                            R = R3d(t+tau,u+nu,v+phi,0,\
                                alpha,XPQ[0],XPQ[1],XPQ[2],RPQ)

                            val += np.power(-1,tau+nu+phi)*Eab*Ecd*R

    val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q))
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

    # H1s, H2s, O1s, O2s, O2px , O2py, O2p
    FCenter = [H2O[0], H2O[1], H2O[2], H2O[2], H2O[2], H2O[2], H2O[2]]
    CartAng = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # Test the electron repulsion integral over chi1 and chi7
    a = OrbCoeff[0,0]
    b = OrbCoeff[6,0]
    lmn1 = CartAng[0]
    lmn2 = CartAng[6]
    A = FCenter[0]
    B = FCenter[6]
    chi_1717 = g3d(a,b,a,b,lmn1,lmn2,lmn1,lmn2,A,B,A,B)
    print(chi_1717)

