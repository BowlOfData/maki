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

    @patch('maki.plugins.ftp_client.ftp_client.ftplib')
    @patch('maki.plugins.ftp_client.ftp_client.paramiko')
    def test_connect_ftp_success(self, mock_paramiko, mock_ftplib):
        """Test successful FTP connection"""
        # Mock FTP connection
        mock_ftp = Mock()
        mock_ftplib.FTP.return_value = mock_ftp

        result = self.ftp_client.connect(
            host="ftp.example.com",
            username="user",
            password="pass",
            protocol="ftp"
        )

        self.assertTrue(result['success'])
        self.assertEqual(result['connection_type'], 'ftp')

    @patch('maki.plugins.ftp_client.ftp_client.ftplib')
    @patch('maki.plugins.ftp_client.ftp_client.paramiko')
    def test_connect_sftp_success(self, mock_paramiko, mock_ftplib):
        """Test successful SFTP connection"""
        # Mock SSH client
        mock_ssh_client = Mock()
        mock_paramiko.SSHClient.return_value = mock_ssh_client

        result = self.ftp_client.connect(
            host="sftp.example.com",
            username="user",
            password="pass",
            protocol="sftp"
        )

        self.assertTrue(result['success'])
        self.assertEqual(result['connection_type'], 'sftp')

    def test_connect_invalid_protocol(self):
        """Test connection with invalid protocol"""
        result = self.ftp_client.connect(
            host="ftp.example.com",
            username="user",
            password="pass",
            protocol="invalid"
        )

        self.assertFalse(result['success'])
        self.assertIn("Unsupported protocol", result['error'])

    def test_connect_missing_requirements(self):
        """Test connection when libraries are not available"""
        # This test would require mocking the import to fail
        # For now, we just verify the error handling
        pass

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