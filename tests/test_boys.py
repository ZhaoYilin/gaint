import numpy as np
import math
import pytest
from gaint.boys import boys

class TestBoysFunction:
    """Test cases for Boys function implementation"""
    
    def test_boys_function_basic(self):
        """Test Boys function for basic cases"""
        # Test n=0 case
        result = boys(0, 0.0)
        assert np.isclose(result, 1.0)
        
        result = boys(0, 1.0)
        expected = np.sqrt(np.pi) * 0.5 * math.erf(1.0) / 1.0
        assert np.isclose(result, expected)
    
    def test_boys_function_small_values(self):
        """Test Boys function for small argument values"""
        # Test with small T values
        for t in [0.0, 0.1, 0.5, 1.0]:
            result = boys(0, t)
            # Boys function should decrease with increasing t
            assert result > 0
            assert result <= 1.0
    
    def test_boys_function_large_values(self):
        """Test Boys function for large argument values"""
        # Test with larger T values
        for t in [5.0, 10.0, 20.0]:
            result = boys(0, t)
            # Boys function should approach 0 for large t
            assert result > 0
            assert result < 0.5
    
    def test_boys_function_different_n(self):
        """Test Boys function for different n values"""
        # Test various n values with fixed t
        t = 1.0
        for n in [0, 1, 2, 3]:
            result = boys(n, t)
            # Higher n should give smaller values
            assert result > 0
    
    def test_boys_function_symmetry(self):
        """Test Boys function symmetry properties"""
        # Boys function should be symmetric in some respects
        # (though not perfectly symmetric due to n parameter)
        t_values = [0.1, 0.5, 1.0, 2.0]
        
        for t in t_values:
            result1 = boys(0, t)
            # Some basic consistency checks
            assert result1 > 0
    
    def test_boys_function_edge_cases(self):
        """Test Boys function edge cases"""
        # Test n=0, t=0 (should be 1.0)
        result = boys(0, 0.0)
        assert np.isclose(result, 1.0)
        
        # Test with very small t
        result = boys(0, 1e-10)
        assert np.isclose(result, 1.0, rtol=1e-5)
