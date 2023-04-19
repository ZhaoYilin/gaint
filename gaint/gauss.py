import numpy as np

class PrimitiveGaussian(object):
    """Primitive Cartiesian Gaussian functions.
    
    Attributes
    ----------
    coefficent : float
        Contraction coefficent of Primitive Gaussian function.

    origin : List[float, float, float]
        Coordinate of the nuclei.
    
    shell : List[int, int, int]
        Angular momentum.

    exponent : float
        Primitive Gaussian exponent.
    
    Properties
    ----------
    norm: float
        Normalization factor.
    
    Methods
    -------
    __init__(self,coefficent, origin, shell, exponent)
        Initialize the instance.

    """
    def __init__(self, coefficient, origin, shell, exponent):
        """Initialize the instance.

        Parameters
        ----------
        coefficient : float
            Contraction coefficient of Primitive Gaussian function.

        origin : List[float, float, float]
            Coordinate of the nuclei.
    
        shell : List[int, int, int]
            Angular momentum.

        exponent : float
            Primitive Gaussian exponent.
        """
        self.coefficient = coefficient
        self.origin = origin
        self.shell = shell
        self.exponent  = exponent

    def __call__(self,x,y,z):
        """Returns the value of the function at point x, y, z.

        Parameters
        ----------
        x : float
        y : float
        z : float

        Returns
        -------
        result : float

        """
        X = x-self.origin[0]
        Y = y-self.origin[1]
        Z = z-self.origin[2]
        rr = X**2+Y**2+Z**2
        return np.power(X,self.shell[0])*np.power(Y,self.shell[1])*\
            np.power(Z,self.shell[2])*np.exp(-self.exponent*rr)

    @property
    def norm(self):
        """Normalization factors. 
        
        Return
        ------
        norm : list
            Normalization factors
        """
        from scipy.special import factorial2 as fact2
        i,j,k = self.shell
        norm = np.sqrt(np.power(2,2*(i+j+k)+1.5)*
                        np.power(self.exponent,i+j+k+1.5)/
                        fact2(2*i-1)/fact2(2*j-1)/
                        fact2(2*k-1)/np.power(np.pi,1.5))
        return norm
