===================
GaInt Documentation
===================

**GaInt** (Gaussian Integration) is a Python package for computing molecular integrals over Primitive Cartesian Gaussian orbitals. It provides implementations of various integration schemes including Obara-Saika and McMurchie-Davidson methods.

.. image:: https://img.shields.io/badge/License-MIT-blue.svg

Key Features
=============

- **Multiple Integration Schemes**: Support for Obara-Saika and McMurchie-Davidson recursive algorithms
- **Comprehensive Integral Types**: Overlap, kinetic, nuclear attraction, and electron repulsion integrals
- **Arbitrary Angular Momentum**: Support for s, p, d, and higher angular momentum orbitals
- **Pure Python Implementation**: Easy to install and use with standard scientific Python tools
- **Well-Documented API**: Clear documentation and examples for all functionality

Quick Start
===========

.. code-block:: python

    import numpy as np
    from gaint.gauss import PrimitiveGaussian
    from gaint.obara_saika.overlap import Overlap

    # Create two s-type Gaussian orbitals
    pg1 = PrimitiveGaussian(1.0, [0.0, 0.0, 0.0], [0, 0, 0], 1.0)
    pg2 = PrimitiveGaussian(1.0, [0.0, 0.0, 0.0], [0, 0, 0], 1.0)

    # Calculate overlap integral
    S = Overlap()
    overlap_value = S(pg1, pg2)
    print(f"Overlap integral: {overlap_value}")

About
======

GaInt was developed to provide researchers with an efficient, flexible tool for computing molecular integrals in quantum chemistry calculations. It emphasizes code clarity and educational value while maintaining computational accuracy for small to medium-sized systems.

Contents
=========

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorial Notebooks

   notebook/gauss
   notebook/mcmurchie_davidson
   notebook/obara_saika

.. toctree::
   :maxdepth: 2
   :caption: API
   :glob:

   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`