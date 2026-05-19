import pytest
import math
from unittest.mock import MagicMock
import sys

# Mock numpy because it is not available in the environment
mock_np = MagicMock()
mock_np.sqrt = math.sqrt
sys.modules["numpy"] = mock_np

from scripts.run_experiment import calculate_matrix_size_for_weak_scaling

def test_calculate_matrix_size_base_case():
    # base_size=512, process_count=1 -> 512 * sqrt(1) = 512
    assert calculate_matrix_size_for_weak_scaling(512, 1) == 512

def test_calculate_matrix_size_two_procs():
    # base_size=256, process_count=2 -> int(256 * sqrt(2)) = int(256 * 1.4142...) = int(362.038...) = 362
    assert calculate_matrix_size_for_weak_scaling(256, 2) == 362

def test_calculate_matrix_size_four_procs():
    # base_size=256, process_count=4 -> 256 * sqrt(4) = 256 * 2 = 512
    assert calculate_matrix_size_for_weak_scaling(256, 4) == 512

def test_calculate_matrix_size_large_scaling():
    # base_size=2048, process_count=4 -> 2048 * sqrt(4) = 2048 * 2 = 4096
    assert calculate_matrix_size_for_weak_scaling(2048, 4) == 4096

def test_calculate_matrix_size_three_procs():
    # base_size=256, process_count=3 -> int(256 * sqrt(3)) = int(256 * 1.732...) = int(443.4...) = 443
    assert calculate_matrix_size_for_weak_scaling(256, 3) == 443

@pytest.mark.parametrize("base_size, procs, expected", [
    (100, 1, 100),
    (100, 4, 200),
    (100, 9, 300),
    (100, 16, 400),
])
def test_calculate_matrix_size_parameterized(base_size, procs, expected):
    assert calculate_matrix_size_for_weak_scaling(base_size, procs) == expected
