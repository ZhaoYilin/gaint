.. gaint documentation master file, created by
   sphinx-quickstart on Fri Apr 20 03:14:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GaInt |version|
###############
**GaInt** is the abbreviation of Gaussian Integral, it is a python package for the non-normalizecd molecular integrals over Primitive Cartesian Gaussian orbitals. This is a particularly naive implementation in python: little attempt is made to conserve memory or CPU time. Nevertheless, it is useful for small test calculations, in particular for investigating ideas quantum chemistry.

To begin a quantum chemistry calculation, first build the Hamiltonian of a system. In terms of second quantization operators, a time-independent non-relativistic Hamiltonian gives:

:math:`H = - \sum_{ij} t_{ij}\hat{c}^{\dagger}_{i}\hat{c}_{j} + \frac{1}{2} \sum_{ijkl} V_{ijkl}\hat{c}^{\dagger}_{i}\hat{c}^{\dagger}_{k}\hat{c}_{l}\hat{c}_{j}`

There are many integration schemes including McMurchie–Davidson, Obara–Saika and Rys schemes. Here Obara-Saika and McMurchie–Davidson scheme is shown in this note.

Contents
========

.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: USER DOCUMENTATION:

   user/installation
   user/molecule
   user/basis
   user/mcmurchie_davidson
   user/obara_saika


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

