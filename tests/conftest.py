import numpy as np
import pytest
from gaint.gauss import PrimitiveGaussian

class TestIntegrationTests:
    """Integration tests combining multiple components"""
    
    def test_complete_overlap_calculation(self):
        """Test complete overlap calculation workflow"""
        # Create basis set for water molecule
        coordinates = [
            [0., 1.43233673, -0.96104039],  # H1
            [0., -1.43233673, -0.96104039], # H2
            [0., 0., 0.24026010]            # O
        ]
        
        # Create s-orbitals on each atom
        orbitals = []
        for coord in coordinates:
            pg = PrimitiveGaussian(
                coefficient=1.0,
                origin=coord,
                shell=[0, 0, 0],
                exponent=1.0
            )
            orbitals.append(pg)
        
        # Calculate all overlap integrals
        overlaps = []
        for i, pg_i in enumerate(orbitals):
            for j, pg_j in enumerate(orbitals):
                if i <= j:  # Only calculate upper triangle
                    try:
                        from gaint.obara_saika.overlap import overlap
                        ovlp = overlap(pg_i, pg_j)
                        overlaps.append((i, j, ovlp))
                    except ImportError:
                        # Skip if overlap module not available
                        continue
        
        # Basic validation
        assert len(overlaps) > 0
        for i, j, ovlp in overlaps:
            if i == j:
                # Diagonal elements should be close to 1.0 (normalized)
                assert np.isclose(ovlp, 1.0, atol=0.1)
            else:
                # Off-diagonal elements should be between 0 and 1
                assert ovlp > 0
                assert ovlp < 1.0
    
    def test_basis_set_consistency(self):
        """Test consistency of basis set calculations"""
        # Test that different calculation methods give consistent results
        # (This would be expanded when more integration methods are implemented)
        pass
