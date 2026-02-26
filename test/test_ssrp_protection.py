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
from maki.maki import Maki

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

    def test_compose_url_with_protocols(self):
        """Test that compose_url handles URLs with protocols properly"""
        # This should work - localhost with protocol
        url = Utils.compose_url("http://localhost", "11434", "generate")
        self.assertTrue(url.startswith("http://"))

        # This should work - public domain with protocol
        url = Utils.compose_url("https://api.example.com", "443", "generate")
        self.assertTrue(url.startswith("http://"))  # Note: protocol gets added by compose_url

    def test_compose_url_with_ipv6(self):
        """Test that compose_url handles IPv6 addresses"""
        # This should work
        url = Utils.compose_url("::1", "11434", "generate")
        self.assertTrue(url.startswith("http://"))

        # This should work with IPv6 address
        url = Utils.compose_url("[2001:db8::1]", "11434", "generate")
        self.assertTrue(url.startswith("http://"))

    def test_compose_url_with_invalid_inputs(self):
        """Test that compose_url validates inputs properly"""
        # Test with invalid action
        with self.assertRaises(ValueError):
            Utils.compose_url("localhost", "11434", "")

        # Test with invalid port
        with self.assertRaises(ValueError):
            Utils.compose_url("localhost", "abc", "generate")

    def test_convert64_with_absolute_paths(self):
        """Test that convert64 accepts absolute paths"""
        # This should work without raising an exception
        # We'll mock the file system operations to avoid actual file access
        with patch('maki.utils.os.path.exists') as mock_exists, \
             patch('maki.utils.open', unittest.mock.mock_open(read_data=b'test data')) as mock_file:

            mock_exists.return_value = True
            result = Utils.convert64("/absolute/path/to/image.jpg")
            self.assertIsNotNone(result)

    def test_convert64_with_relative_paths(self):
        """Test that convert64 accepts relative paths"""
        # This should work without raising an exception
        with patch('maki.utils.os.path.exists') as mock_exists, \
             patch('maki.utils.open', unittest.mock.mock_open(read_data=b'test data')) as mock_file:

            mock_exists.return_value = True
            result = Utils.convert64("relative/path/to/image.jpg")
            self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()