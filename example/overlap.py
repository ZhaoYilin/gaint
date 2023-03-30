import numpy as np
from gaint.gauss import PrimitiveGaussian
from gaint.operator import Overlap 

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
O_overlap = Overlap('os')
o_17 = O_overlap(g1,g2)

print(np.isclose(o_17,-0.0000888019))
