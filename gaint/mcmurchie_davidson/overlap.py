import numpy as np
from gaint.mcmurchie_davidson.expansion_coefficients import E1d

        
def S1d(a, b, i, j, A, B):
    """The Mcmurchie Davidson scheme for one-dimensional overlap integrals over primitive
    Gaussian orbitals.

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
        The non-normalizecd overlap interals in one dimension.
    """
    # p the total exponent
    p = a + b
    result = E1d(i,j,0,A,B,a,b)*np.sqrt(np.pi/p)
    return result

def S3d(a, b, ikm, jln, A, B):
    """The Obara-Saika scheme for three-dimensional overlap integrals over primitive
    Gaussian orbitals.

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
    i,k,m = ikm
    j,l,n = jln
    Sij = S1d(a,b,i,j,A[0],B[0]) # X
    Skl = S1d(a,b,k,l,A[1],B[1]) # Y
    Smn = S1d(a,b,m,n,A[2],B[2]) # Z
    result = Sij*Skl*Smn
    return result

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

    chi_17 = S3d(OrbCoeff[0,0], OrbCoeff[6,0], CartAng[0], CartAng[6], FCenter[0], FCenter[6])
    print(np.isclose(chi_17,-0.0000888019))
