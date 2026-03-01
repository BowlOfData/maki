"""
Tests for the FTPClient plugin

This file contains unit tests for the FTPClient plugin functionality.
"""

import unittest
from unittest.mock import Mock, patch
import os

# Import the FTPClient plugin
from maki.plugins.ftp_client.ftp_client import FTPClient


class TestFTPClient(unittest.TestCase):
    """Test cases for the FTPClient plugin"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock Maki instance
        self.mock_maki = Mock()
        self.ftp_client = FTPClient(self.mock_maki)

    def test_connect_missing_requirements(self):
        """Test connection when libraries are not available"""
        # Mock HAS_FTP_LIBS to False to test the library availability check
        with patch('maki.plugins.ftp_client.ftp_client.HAS_FTP_LIBS', False):
            result = self.ftp_client.connect(
                host="ftp.example.com",
                username="user",
                password="pass",
                protocol="ftp"
            )

            self.assertFalse(result['success'])
            self.assertIn("FTP/SFTP libraries not available", result['error'])

    def test_connect_invalid_protocol_when_libraries_available(self):
        """Test connection with invalid protocol when libraries are available"""
        # Mock HAS_FTP_LIBS to True to test the protocol validation
        with patch('maki.plugins.ftp_client.ftp_client.HAS_FTP_LIBS', True):
            result = self.ftp_client.connect(
                host="ftp.example.com",
                username="user",
                password="pass",
                protocol="invalid"
            )

            self.assertFalse(result['success'])
            self.assertIn("Unsupported protocol", result['error'])

    def test_disconnect_without_connection(self):
        """Test disconnect when not connected"""
        result = self.ftp_client.disconnect()

        # Should not fail even when not connected
        self.assertTrue(result['success'])

    def test_upload_file_without_connection(self):
        """Test upload file when not connected"""
        result = self.ftp_client.upload_file("/local/file.txt", "/remote/file.txt")

        self.assertFalse(result['success'])
        self.assertIn("Not connected", result['error'])

    def test_download_file_without_connection(self):
        """Test download file when not connected"""
        result = self.ftp_client.download_file("/remote/file.txt", "/local/file.txt")

        self.assertFalse(result['success'])
        self.assertIn("Not connected", result['error'])

    def test_list_directory_without_connection(self):
        """Test list directory when not connected"""
        result = self.ftp_client.list_directory()

        self.assertFalse(result['success'])
        self.assertIn("Not connected", result['error'])

    def test_create_directory_without_connection(self):
        """Test create directory when not connected"""
        result = self.ftp_client.create_directory("/remote/dir")

        self.assertFalse(result['success'])
        self.assertIn("Not connected", result['error'])

    def test_remove_directory_without_connection(self):
        """Test remove directory when not connected"""
        result = self.ftp_client.remove_directory("/remote/dir")

        self.assertFalse(result['success'])
        self.assertIn("Not connected", result['error'])

    def test_get_file_info_without_connection(self):
        """Test get file info when not connected"""
        result = self.ftp_client.get_file_info("/remote/file.txt")

        self.assertFalse(result['success'])
        self.assertIn("Not connected", result['error'])


if __name__ == '__main__':
    unittest.main()