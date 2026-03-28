import numpy as np
import pytest
from gaint.gauss import PrimitiveGaussian

class TestPrimitiveGaussian:
    """Test cases for PrimitiveGaussian class"""
    
    def test_initialization(self):
        """Test PrimitiveGaussian initialization"""
        # Test basic initialization
        pg = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[0, 0, 0],
            exponent=1.0
        )
        
        assert pg.coefficient == 1.0
        assert pg.origin == [0.0, 0.0, 0.0]
        assert pg.shell == [0, 0, 0]
        assert pg.exponent == 1.0
    
    def test_call_method(self):
        """Test the __call__ method"""
        pg = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[0, 0, 0],
            exponent=1.0
        )
        
        # Test at origin
        result = pg(0.0, 0.0, 0.0)
        assert result == 1.0
        
        # Test away from origin
        result = pg(1.0, 0.0, 0.0)
        expected = np.exp(-1.0)
        assert np.isclose(result, expected)
    
    def test_norm_property_s_orbital(self):
        """Test normalization for s-orbital (shell=[0,0,0])"""
        pg = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[0, 0, 0],
            exponent=1.0
        )
        
        norm = pg.norm
        # For s-orbital with exponent=1.0, norm should be (2/π)^(3/4)
        expected = (2.0 / np.pi) ** 0.75
        assert np.isclose(norm, expected)
    
    def test_norm_property_p_orbital(self):
        """Test normalization for p-orbital (shell=[1,0,0])"""
        pg = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[1, 0, 0],
            exponent=1.0
        )
        
        norm = pg.norm
        # For p-orbital with exponent=1.0
        expected = (128.0 / np.pi**3) ** 0.25
        assert np.isclose(norm, expected)
    
    def test_norm_property_d_orbital(self):
        """Test normalization for d-orbital (shell=[2,0,0])"""
        pg = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[2, 0, 0],
            exponent=1.0
        )
        
        norm = pg.norm
        # For d-orbital with exponent=1.0
        expected = (2048.0 / (9.0 * np.pi**3)) ** 0.25
        assert np.isclose(norm, expected)
    
    def test_norm_no_divide_by_zero(self):
        """Test that normalization doesn't cause divide by zero errors"""
        # Test various shell combinations that could cause issues
        test_shells = [
            [0, 0, 0],  # s-orbital
            [1, 0, 0],  # px-orbital
            [0, 1, 0],  # py-orbital
            [0, 0, 1],  # pz-orbital
            [2, 0, 0],  # dxx-orbital
            [1, 1, 0],  # dxy-orbital
            [0, 0, 2],  # dzz-orbital
        ]
        
        for shell in test_shells:
            pg = PrimitiveGaussian(
                coefficient=1.0,
                origin=[0.0, 0.0, 0.0],
                shell=shell,
                exponent=1.0
            )
            
            # This should not raise any division by zero warnings
            norm = pg.norm
            assert isinstance(norm, float)
            assert norm > 0  # Normalization should be positive
    
    def test_different_exponents(self):
        """Test with different Gaussian exponents"""
        exponents = [0.5, 1.0, 2.0, 5.0]
        
        for exp in exponents:
            pg = PrimitiveGaussian(
                coefficient=1.0,
                origin=[0.0, 0.0, 0.0],
                shell=[0, 0, 0],
                exponent=exp
            )
            
            norm = pg.norm
            # Higher exponents should give larger normalization factors
            assert norm > 0
    
    def test_different_origins(self):
        """Test with different origin coordinates"""
        origins = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ]
        
        for origin in origins:
            pg = PrimitiveGaussian(
                coefficient=1.0,
                origin=origin,
                shell=[0, 0, 0],
                exponent=1.0
            )
            
            # Origin should not affect normalization
            norm = pg.norm
            expected = (2.0 / np.pi) ** 0.75
            assert np.isclose(norm, expected)
    
    def test_gaussian_decay(self):
        """Test that Gaussian function decays properly"""
        pg = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[0, 0, 0],
            exponent=1.0
        )
        
        # Function should decay as we move away from origin
        values = [pg(x, 0.0, 0.0) for x in [0.0, 0.5, 1.0, 2.0]]
        
        # Values should be decreasing
        for i in range(len(values) - 1):
            assert values[i] > values[i + 1]
    
    def test_angular_dependence(self):
        """Test angular dependence for different orbitals"""
        # Test s-orbital (should be isotropic)
        pg_s = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[0, 0, 0],
            exponent=1.0
        )
        
        # s-orbital should be symmetric
        assert pg_s(1.0, 0.0, 0.0) == pg_s(0.0, 1.0, 0.0)
        assert pg_s(1.0, 0.0, 0.0) == pg_s(0.0, 0.0, 1.0)
        
        # Test p-orbital (should have directional dependence)
        pg_px = PrimitiveGaussian(
            coefficient=1.0,
            origin=[0.0, 0.0, 0.0],
            shell=[1, 0, 0],
            exponent=1.0
        )
        
        # px-orbital should be asymmetric
        assert pg_px(1.0, 0.0, 0.0) != pg_px(0.0, 1.0, 0.0)
        assert pg_px(0.0, 1.0, 0.0) == 0.0  # Should be zero along y-axis
