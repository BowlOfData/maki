"""
Unit tests for SSRF protection in Maki framework
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to Python path so imports work properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maki.utils import Utils
from maki.connector import Connector

class TestSSRFProtection(unittest.TestCase):

    def test_validate_domain_with_localhost(self):
        """Test that localhost is allowed"""
        # This should not raise an exception
        Utils._validate_domain("localhost")
        Utils._validate_domain("127.0.0.1")
        Utils._validate_domain("::1")

    def test_validate_domain_with_private_ips(self):
        """Test that private IPs are blocked"""
        # These should raise ValueError
        with self.assertRaises(ValueError):
            Utils._validate_domain("10.0.0.1")

        with self.assertRaises(ValueError):
            Utils._validate_domain("172.16.0.1")

        with self.assertRaises(ValueError):
            Utils._validate_domain("192.168.1.1")

    def test_validate_domain_with_public_ips(self):
        """Test that public IPs are allowed"""
        # These should not raise exceptions
        Utils._validate_domain("8.8.8.8")
        Utils._validate_domain("1.1.1.1")
        Utils._validate_domain("2001:4860:4860::8888")

    def test_validate_domain_with_blacklisted_domains(self):
        """Test that blacklisted domains are blocked"""
        with self.assertRaises(ValueError):
            Utils._validate_domain("0.0.0.0")

        with self.assertRaises(ValueError):
            Utils._validate_domain("255.255.255.255")

    def test_validate_domain_with_allocated_public_space(self):
        """128.0.0.0/16 is allocated public space and must not be blocked"""
        Utils._validate_domain("128.0.0.1")
        Utils._validate_domain("128.0.255.254")

    def test_validate_domain_blocks_benchmarking_range(self):
        """198.18.0.0/15 (RFC 2544 benchmarking) stays blocked"""
        with self.assertRaises(ValueError):
            Utils._validate_domain("198.18.0.1")
        with self.assertRaises(ValueError):
            Utils._validate_domain("198.19.255.254")

    def test_validate_domain_allow_private(self):
        """allow_private permits private IPs for operator-configured endpoints"""
        Utils._validate_domain("192.168.1.10", allow_private=True)
        Utils._validate_domain("10.0.0.5", allow_private=True)
        # Blacklisted literals stay blocked even with allow_private
        with self.assertRaises(ValueError):
            Utils._validate_domain("0.0.0.0", allow_private=True)
        # Format checks still apply
        with self.assertRaises(ValueError):
            Utils._validate_domain("bad domain!", allow_private=True)

    def test_convert64_with_absolute_paths(self):
        """Test that convert64 accepts absolute paths"""
        # This should work without raising an exception
        # We'll mock the file system operations to avoid actual file access
        with patch('maki.utils.os.path.exists') as mock_exists, \
             patch('maki.utils.os.path.isfile') as mock_isfile, \
             patch('maki.utils.open', unittest.mock.mock_open(read_data=b'test data')) as mock_file, \
             patch('maki.utils.os.getcwd') as mock_getcwd:

            # Mock that the current working directory is the parent of our test path
            # This allows the absolute path to be considered valid (within working directory)
            mock_getcwd.return_value = "/absolute/path/to"
            mock_exists.return_value = True
            mock_isfile.return_value = True
            result = Utils.convert64("/absolute/path/to/image.jpg")
            self.assertIsNotNone(result)

    def test_convert64_with_relative_paths(self):
        """Test that convert64 accepts relative paths"""
        # This should work without raising an exception
        with patch('maki.utils.os.path.exists') as mock_exists, \
             patch('maki.utils.os.path.isfile') as mock_isfile, \
             patch('maki.utils.open', unittest.mock.mock_open(read_data=b'test data')) as mock_file:

            # Mock that the file exists and is a file (not a directory)
            mock_exists.return_value = True
            mock_isfile.return_value = True
            result = Utils.convert64("relative/path/to/image.jpg")
            self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
