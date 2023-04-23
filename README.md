# GaInt

GaInt is abbreviation of *Ga*ussian *Int*gral, it is a python package for the non-normalizecd molecular integrals over Primitive Cartesian Gaussian orbitals. This is a particularly naive implementation in python: little attempt is made to conserve memory or CPU time. Nevertheless, it is useful for small test calculations, in particular for investigating ideas quantum chemistry.




## Installation
* Prerequisties:

  - Python 3.5 or above
  - numpy 1.13.1
  - scipy 0.19.1

* Compile from source

      git clone https://github.com/ZhaoYilin/gaint.git
      cd gaint
      python setup.py install --user

* Using pip to install python package on GitHub

      pip install git+https://github.com/ZhaoYilin/gaint


## Documentation

### Hamiltonian
The core mechanical quantities of a chemistry system is the Hamiltonian. Hamiltonian operator should include the kinetic energy and potential energy terms of all atomic nuclei and all electrons. It is generally assumed that the molecule is in a vacuum and adiabatic state in isolation. At this time, the interaction potential energy between the nucleus and the electron in the molecule is only related to distance from each other and time independent. Its expression is:

$$
\begin{aligned}
\hat{H}_{elec}= &-\sum^N_{i=1}\frac{\hbar^2}{2m_i}{\nabla}_i^2
-\sum^N_{i=1}\sum^M_{\alpha=1} \frac{Z_a e^2}{\textbf{r}_{ia}}\\
&+\sum^N_{i=1}\sum^N_{j>i} \frac{e^2}{\textbf{r}_{ij}}
\end{aligned}
$$

Where $m_i$ is the mass of electron. $M_\alpha$ and $Z_\alpha$ refer to the mass and charge of atomic nucleus. $R_{\alpha\beta}$, $r_{i\alpha}$ and $r_{ij}$ is the distance between two nucleus, atomic nuclei and electron and two electrons respectively. The explicit representation of Laplacian operator is:

$$
\boldsymbol{\nabla}^2 = \frac{\partial^2}{\partial x^2} +\frac{\partial^2}{\partial y^2} 
+\frac{\partial^2}{\partial z^2}
$$

|  Name  | Operators |Symbol | Shape |
|:--------:|:--------:|:------:|:------:|
|Overlap| 1 |  S   | (nbasis,nbasis)   | 
|Kinetic| $-{\nabla}_i^2$ |  T   | (nbasis,nbasis) | 
|Nuclear Attraction| $-\frac{Z_a}{\textbf{r}_{ia}}$ |  V   | (nbasis,nbasis) | 
|Electron Repulsion| $\frac{1}{\textbf{r}_{ij}}$   |Eri   | (nbasis,nbasis,nbasis,nbasis) |


### Primitive Gaussian Type Orbital

Most quantum chemistry package supports Contracted Gaussian type orbtial(CGTO), it is an atomic orbital forming in linear combinations of primitive Gaussians type orbital(PGTO). Here in this note, only the integral over primitive Gaussian type orbital is to be considered which are defiend by an angular part which is homogeneous polynomial in the compoents x, y, and z of the position vector $\mathbf{r}$. That is,

$$
G_{ijk}(r_A,a) = x^i_A y^j_A z^k_A e^{-a r^2_A}
$$

- a>0 is the orbital exponent.  
- $r_A = r − A$ is the electronic coordinate.
- i ≥ 0, j ≥ 0, k ≥ 0 is the quantum number.
- Total angular-momentum quantum number l = i + j + k ≥ 0

```python 
from gaint.gauss import PrimitiveGaussian
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

pga = PrimitiveGaussian(1.0,FCenter[0],CartAng[0],OrbCoeff[0,0])
pgb = PrimitiveGaussian(1.0,FCenter[6],CartAng[6],OrbCoeff[6,0])
```

### Obara-Saika Scheme
```python
# The non-normalizecd integral between two primitive Gaussian type orbital
from gaint.obara_saikai.overlap import Overlap
S = Overlap()
s17 = S(pga,pgb)
print(np.isclose(s_17,-0.0000888019))

from gaint.obara_saikai.kinetic import Kinetic
T = Kinetic()
t17 = T(pga,pgb)
print(np.isclose(t17,0.00167343))

from gaint.obara_saikai.nuclear_attraction import NuclearAttraction
V = NuclearAttraction()
v17 = V(pga,pgb,FCenter[0])
print(np.isclose(v17,-0.0000854386))

from gaint.obara_saikai.electron_repulsion import ElectronRepulsion
Eri = ElectronRepulsion()
eri1717 = Eri(pga,pgb,pga,pgb)
print(np.isclose(eri1717,1.9060888184873294e-08))
```

### McMurchie Davidson Scheme
```python
# The non-normalizecd integral between two primitive Gaussian type orbital
from gaint.mcmurchie_davidson.overlap import Overlap
S = Overlap()
s17 = S(pga,pgb)
print(np.isclose(s_17,-0.0000888019))

from gaint.mcmurchie_davidson.kinetic import Kinetic
T = Kinetic()
t17 = T(pga,pgb)
print(np.isclose(t17,0.00167343))

from gaint.mcmurchie_davidson.nuclear_attraction import NuclearAttraction
V = NuclearAttraction()
v17 = V(pga,pgb,FCenter[0])
print(np.isclose(v17,-0.0000854386))

from gaint.mcmurchie_davidson.electron_repulsion import ElectronRepulsion
Eri = ElectronRepulsion()
eri1717 = Eri(pga,pgb,pga,pgb)
print(np.isclose(eri1717,1.9060888184873294e-08))
```