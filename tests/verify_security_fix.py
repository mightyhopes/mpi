import unittest
from unittest.mock import patch, MagicMock
import subprocess
import os
import sys

# Mock numpy before importing run_experiment
sys.modules['numpy'] = MagicMock()

# Add the scripts directory to the path so we can import run_experiment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from run_experiment import run_experiment

class TestSecurityFix(unittest.TestCase):
    @patch('subprocess.run')
    def test_run_experiment_leaks_no_info_on_oserror(self, mock_run):
        # Setup mock to raise OSError with sensitive info
        mock_run.side_effect = OSError("Secret path: /home/user/secret_file not found")

        # Run experiment
        # Redirect stderr to avoid cluttering test output
        with patch('sys.stderr', new=MagicMock()):
            exec_time, error_msg = run_experiment("dummy_path", 1, 100)

        # Verify
        self.assertIsNone(exec_time)
        self.assertEqual(error_msg, "System error during execution")
        self.assertNotIn("Secret path", error_msg)
        self.assertNotIn("/home/user/secret_file", error_msg)

    @patch('subprocess.run')
    def test_run_experiment_leaks_no_info_on_generic_exception(self, mock_run):
        # Setup mock to raise a generic Exception with sensitive info
        mock_run.side_effect = Exception("Database connection failed: user=admin, pass=password123")

        # Run experiment
        with patch('sys.stderr', new=MagicMock()):
            exec_time, error_msg = run_experiment("dummy_path", 1, 100)

        # Verify
        self.assertIsNone(exec_time)
        self.assertEqual(error_msg, "Internal execution error")
        self.assertNotIn("user=admin", error_msg)
        self.assertNotIn("password123", error_msg)

    @patch('subprocess.run')
    def test_run_experiment_timeout_is_still_handled(self, mock_run):
        # Setup mock to raise TimeoutExpired
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=120)

        # Run experiment
        exec_time, error_msg = run_experiment("dummy_path", 1, 100, timeout_sec=120)

        # Verify
        self.assertIsNone(exec_time)
        self.assertEqual(error_msg, "TIMEOUT after 120s")

if __name__ == '__main__':
    unittest.main()
