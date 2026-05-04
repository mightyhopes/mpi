import unittest
from unittest.mock import MagicMock
import sys

# Mock dependencies that might be missing in the environment
sys.modules['pandas'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Import the function to test
from scripts.plot_results import calculate_efficiency

class TestCalculateEfficiency(unittest.TestCase):
    def test_standard_case(self):
        """Test efficiency with 100% speedup"""
        self.assertEqual(calculate_efficiency(2, 2), 100.0)

    def test_sublinear_speedup(self):
        """Test efficiency with sub-linear speedup"""
        self.assertEqual(calculate_efficiency(1.5, 2), 75.0)

    def test_superlinear_speedup(self):
        """Test efficiency with super-linear speedup"""
        self.assertEqual(calculate_efficiency(2.5, 2), 125.0)

    def test_zero_speedup(self):
        """Test efficiency with zero speedup"""
        self.assertEqual(calculate_efficiency(0, 4), 0.0)

    def test_zero_processes(self):
        """Test efficiency with zero processes (should handle division by zero)"""
        self.assertEqual(calculate_efficiency(2, 0), 0.0)

    def test_negative_processes(self):
        """Test efficiency with negative processes"""
        # Current implementation: num_procs > 0 else 0
        self.assertEqual(calculate_efficiency(2, -1), 0.0)

    def test_negative_speedup(self):
        """Test efficiency with negative speedup"""
        self.assertEqual(calculate_efficiency(-1, 2), -50.0)

    def test_float_values(self):
        """Test with float inputs"""
        self.assertAlmostEqual(calculate_efficiency(1.23, 2.0), 61.5)

if __name__ == '__main__':
    unittest.main()
