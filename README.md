GaInt
==================================

GaInt is abbreviation of *Ga*ussian&*Int*gral, it is a python package for molecular integrals over primitive Cartesian Gaussian orbitals. This is a particularly naive implementation in python: little attempt is made to conserve memory or CPU time. Nevertheless, it is useful for small test calculations, in particular for investigating ideas  quantum chemistry.




Installation
------------

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
