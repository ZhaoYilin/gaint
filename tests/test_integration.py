import pytest
import numpy as np
from gaint.gauss import PrimitiveGaussian

@pytest.fixture
def s_orbital():
    """Fixture for s-type Gaussian orbital"""
    return PrimitiveGaussian(
        coefficient=1.0,
        origin=[0.0, 0.0, 0.0],
        shell=[0, 0, 0],
        exponent=1.0
    )

@pytest.fixture
def p_orbital():
    """Fixture for p-type Gaussian orbital"""
    return PrimitiveGaussian(
        coefficient=1.0,
        origin=[0.0, 0.0, 0.0],
        shell=[1, 0, 0],
        exponent=1.0
    )

@pytest.fixture
def d_orbital():
    """Fixture for d-type Gaussian orbital"""
    return PrimitiveGaussian(
        coefficient=1.0,
        origin=[0.0, 0.0, 0.0],
        shell=[2, 0, 0],
        exponent=1.0
    )

@pytest.fixture
def water_molecule_coordinates():
    """Fixture for H2O molecule coordinates"""
    return [
        [0., 1.43233673, -0.96104039],  # H1
        [0., -1.43233673, -0.96104039], # H2
        [0., 0., 0.24026010]            # O
    ]
