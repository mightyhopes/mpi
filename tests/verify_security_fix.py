import sys
from unittest.mock import MagicMock

# Mock numpy and matplotlib before they are imported by run_experiment
sys.modules['numpy'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

import unittest
from unittest.mock import patch, MagicMock
import os
import scripts.run_experiment as run_experiment

class TestSecurityFix(unittest.TestCase):

    @patch('scripts.run_experiment.subprocess.run')
    @patch('scripts.run_experiment.get_mpi_launcher')
    @patch('os.path.isfile')
    def test_secure_subprocess_call(self, mock_isfile, mock_get_mpi, mock_run):
        # Setup
        mock_isfile.return_value = True
        mock_get_mpi.return_value = ('mpirun', '-np')
        mock_run.return_value = MagicMock(stdout="Time: 1.23s", stderr="")

        # We need to make sure the path is considered "inside" mpi_programs
        program_path = "mpi_programs/matrix_standard.py"
        num_procs = 4
        matrix_size = 512

        # Execute
        run_experiment.run_experiment(program_path, num_procs, matrix_size)

        # Verify
        self.assertTrue(mock_run.called)
        args, kwargs = mock_run.call_args

        # Verify shell=False is explicitly passed
        self.assertIn('shell', kwargs)
        self.assertFalse(kwargs['shell'])

        # Verify command is a list
        cmd = args[0]
        self.assertIsInstance(cmd, list)
        self.assertEqual(cmd[0], 'mpirun')
        self.assertEqual(cmd[2], '4')
        self.assertIn('--test-n', cmd)
        self.assertIn('512', cmd)

    def test_invalid_num_procs(self):
        # Test with non-integer string
        res_time, res_out = run_experiment.run_experiment("mpi_programs/test.py", "abc", 512)
        self.assertIsNone(res_time)
        self.assertIn("Invalid num_procs", res_out)

        # Test with negative number
        res_time, res_out = run_experiment.run_experiment("mpi_programs/test.py", -1, 512)
        self.assertIsNone(res_time)
        self.assertIn("Invalid num_procs", res_out)

    def test_invalid_matrix_size(self):
        # Test with non-integer string
        res_time, res_out = run_experiment.run_experiment("mpi_programs/test.py", 4, "xyz")
        self.assertIsNone(res_time)
        self.assertIn("Invalid matrix_size", res_out)

    def test_path_traversal_violation(self):
        # Test with path outside mpi_programs
        res_time, res_out = run_experiment.run_experiment("scripts/run_experiment.py", 4, 512)
        self.assertIsNone(res_time)
        self.assertIn("Security Violation", res_out)

        # Test with directory traversal attempt
        res_time, res_out = run_experiment.run_experiment("mpi_programs/../scripts/run_experiment.py", 4, 512)
        self.assertIsNone(res_time)
        self.assertIn("Security Violation", res_out)

if __name__ == '__main__':
    unittest.main()
