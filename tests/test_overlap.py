import numpy as np
import pytest
from gaint.gauss import PrimitiveGaussian
from gaint.obara_saika.overlap import Overlap

class TestOverlapIntegral:
    """Test cases for overlap integral calculations"""
    
    def test_overlap_ss_same_center(self):
        """Test s-s overlap at same center"""
        # Create two s-type Gaussians at same center
        pg_a = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[0, 0, 0],
            exponent=1.0
        )
        
        pg_b = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[0, 0, 0],
            exponent=1.0
        )
        
        S = Overlap()
        result = S(pg_a, pg_b)
        # Overlap of two identical s-orbitals at same center with exponent=1.0
        # Should be (π/(a+b))^(3/2) = (π/2)^(3/2) ≈ 1.9687
        expected = (np.pi / 2.0) ** 1.5
        assert np.isclose(result, expected, rtol=1e-5)
    
    def test_overlap_ss_different_centers(self):
        """Test s-s overlap at different centers"""
        pg_a = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[0, 0, 0],
            exponent=1.0
        )
        
        pg_b = PrimitiveGaussian(
            coefficient=1.0,
            origin=[1.0, 0.0, 0.0],
            shell=[0, 0, 0],
            exponent=1.0
        )
        
        S = Overlap()
        result = S(pg_a, pg_b)
        # Overlap should be positive but less than same-center overlap
        same_center_result = (np.pi / 2.0) ** 1.5
        assert result > 0
        assert result < same_center_result
    
    def test_overlap_sp_orthogonality(self):
        """Test s-p overlap (should be zero due to symmetry)"""
        pg_s = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[0, 0, 0],
            exponent=1.0
        )
        
        pg_px = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[1, 0, 0],
            exponent=1.0
        )
        
        S = Overlap()
        S = Overlap()
        result = S(pg_s, pg_px)
        # s and p orbitals at same center should be orthogonal
        assert np.isclose(result, 0.0, atol=1e-10)
    
    def test_overlap_pp_same_center(self):
        """Test p-p overlap at same center"""
        pg_px1 = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[1, 0, 0],
            exponent=1.0
        )
        
        pg_px2 = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[1, 0, 0],
            exponent=1.0
        )
        
        S = Overlap()
        S = Overlap()
        result = S(pg_px1, pg_px2)
        # Overlap of two identical p-orbitals at same center with exponent=1.0
        # Should be (π/(a+b))^(3/2) * (1/(2p)) = (π/2)^(3/2) * (1/2) ≈ 0.4922
        expected = (np.pi / 2.0) ** 1.5 * (1.0 / (2 * 2.0))
        assert np.isclose(result, expected, rtol=1e-5)
    
    def test_overlap_different_exponents(self):
        """Test overlap with different Gaussian exponents"""
        pg_a = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[0, 0, 0],
            exponent=1.0
        )
        
        pg_b = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[0, 0, 0],
            exponent=2.0
        )
        
        S = Overlap()
        S = Overlap()
        result = S(pg_a, pg_b)
        # Overlap should be positive but less than same-exponent overlap
        same_exponent_result = (np.pi / 3.0) ** 1.5  # a=1.0, b=2.0, a+b=3.0
        assert result > 0
        # For different exponents, overlap should be less than geometric mean case
        assert result < (np.pi / 2.0) ** 1.5