import numpy as np
from gaint.direct.overlap import s1d

def t1d(a, b, i, j, A, B):
    term1 = 0.5*i*j*s1d(a,b,i-1,j-1,A,B)    
    term2 = -a*j*s1d(a,b,i+1,j-1,A,B)    
    term3 = -i*b*s1d(a,b,i-1,j+1,A,B)    
    term4 = 2*a*b*s1d(a,b,i+1,j+1,A,B)    
    result = term1+term2+term3+term4
    return result

def T3d(a, b, ikm, jln, A, B):
    """The Obara-Saika scheme for three-dimensional kinetic energy integral over
    primitive Gaussian orbitals.

    Parameters
    ----------
    a : float 
        Gaussian exponent facotr.

    b : float 
        Gaussian exponent facotr.

    ikm : List[int]
        Angular momentum quantum number.

    jln : List[int]
        Angular momentum quantum number.

    A : List[float]
        Coordinate at positon A.

    B : List[float]
        Coordinate at postion B.

    Returns
    -------
    result : float
        The non-normalizecd overlap interals in three dimension.
    """
    # p the total exponent
    p = a + b
    # P the centre-of-charge coordinate
    A = np.array(A)
    B = np.array(B)
    P = (a*A + b*B)/p
    # XPA, XPB and XAB the realative coordinate
    XAB = A-B
    RAB = np.linalg.norm(XAB)


    i,k,m = ikm
    j,l,n = jln
    
    # sij differ from Sij by a factor.
    tij = t1d(a,b,i,j,A[0],B[0])
    skl = s1d(a,b,k,l,A[1],B[1])
    smn = s1d(a,b,m,n,A[2],B[2])
    sij = s1d(a,b,i,j,A[0],B[0])
    tkl = t1d(a,b,k,l,A[1],B[1])
    smn = s1d(a,b,m,n,A[2],B[2])
    sij = s1d(a,b,i,j,A[0],B[0])
    skl = s1d(a,b,k,l,A[1],B[1])
    tmn = t1d(a,b,m,n,A[2],B[2])
    tab = tij*skl*smn+sij*tkl*smn+sij*skl*tmn
    result = (np.pi/p)**1.5*np.exp(-a*b*RAB**2/p)*tab
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

    chi_17 = T3d(OrbCoeff[0,0], OrbCoeff[6,0], CartAng[0], CartAng[6], FCenter[0], FCenter[6])
    print(chi_17)
    print(np.isclose(chi_17,0.00167343))

