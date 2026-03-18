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


class TestRemotePathTraversalValidation(unittest.TestCase):
    """Tests verifying that remote path traversal is blocked in all operations (CVE-2.1 / CVE-2.2)."""

    TRAVERSAL_PATHS = [
        "../etc/passwd",
        "../../etc/shadow",
        "uploads/../../etc/cron.d",
        "foo/../../../root/.ssh/id_rsa",
        "..",
    ]

    def setUp(self):
        self.client = FTPClient()
        # Simulate a connected FTP client
        self.client.connected = True
        self.client.ftp_connection = Mock()
        self.client.connection_type = 'ftp'

        # Simulate a connected SFTP client
        self.sftp_client = FTPClient()
        self.sftp_client.connected = True
        self.sftp_client.sftp_connection = Mock()
        self.sftp_client.connection_type = 'sftp'
        mock_sftp = Mock()
        self.sftp_client.sftp_connection.open_sftp = Mock(return_value=mock_sftp)

    def _assert_traversal_blocked(self, result):
        self.assertFalse(result['success'], f"Expected traversal to be blocked, got: {result}")
        self.assertIn("Invalid remote path", result.get('error', ''))

    # --- upload_file ---

    def test_upload_ftp_remote_path_traversal(self):
        for path in self.TRAVERSAL_PATHS:
            with self.subTest(path=path):
                result = self.client.upload_file("local.txt", path)
                self._assert_traversal_blocked(result)
                self.client.ftp_connection.storbinary.assert_not_called()

    def test_upload_sftp_remote_path_traversal(self):
        for path in self.TRAVERSAL_PATHS:
            with self.subTest(path=path):
                result = self.sftp_client.upload_file("local.txt", path)
                self._assert_traversal_blocked(result)
                self.sftp_client.sftp_connection.open_sftp.assert_not_called()

    # --- download_file ---

    def test_download_ftp_remote_path_traversal(self):
        for path in self.TRAVERSAL_PATHS:
            with self.subTest(path=path):
                result = self.client.download_file(path, "local.txt")
                self._assert_traversal_blocked(result)
                self.client.ftp_connection.retrbinary.assert_not_called()

    def test_download_sftp_remote_path_traversal(self):
        for path in self.TRAVERSAL_PATHS:
            with self.subTest(path=path):
                result = self.sftp_client.download_file(path, "local.txt")
                self._assert_traversal_blocked(result)
                self.sftp_client.sftp_connection.open_sftp.assert_not_called()

    # --- list_directory ---

    def test_list_directory_ftp_remote_path_traversal(self):
        for path in self.TRAVERSAL_PATHS:
            with self.subTest(path=path):
                result = self.client.list_directory(path)
                self._assert_traversal_blocked(result)
                self.client.ftp_connection.cwd.assert_not_called()

    def test_list_directory_sftp_remote_path_traversal(self):
        for path in self.TRAVERSAL_PATHS:
            with self.subTest(path=path):
                result = self.sftp_client.list_directory(path)
                self._assert_traversal_blocked(result)
                self.sftp_client.sftp_connection.open_sftp.assert_not_called()

    # --- create_directory ---

    def test_create_directory_ftp_remote_path_traversal(self):
        for path in self.TRAVERSAL_PATHS:
            with self.subTest(path=path):
                result = self.client.create_directory(path)
                self._assert_traversal_blocked(result)
                self.client.ftp_connection.mkd.assert_not_called()

    def test_create_directory_sftp_remote_path_traversal(self):
        for path in self.TRAVERSAL_PATHS:
            with self.subTest(path=path):
                result = self.sftp_client.create_directory(path)
                self._assert_traversal_blocked(result)
                self.sftp_client.sftp_connection.open_sftp.assert_not_called()

    # --- remove_directory ---

    def test_remove_directory_ftp_remote_path_traversal(self):
        for path in self.TRAVERSAL_PATHS:
            with self.subTest(path=path):
                result = self.client.remove_directory(path)
                self._assert_traversal_blocked(result)
                self.client.ftp_connection.rmd.assert_not_called()

    def test_remove_directory_sftp_remote_path_traversal(self):
        for path in self.TRAVERSAL_PATHS:
            with self.subTest(path=path):
                result = self.sftp_client.remove_directory(path)
                self._assert_traversal_blocked(result)
                self.sftp_client.sftp_connection.open_sftp.assert_not_called()

    # --- get_file_info ---

    def test_get_file_info_ftp_remote_path_traversal(self):
        for path in self.TRAVERSAL_PATHS:
            with self.subTest(path=path):
                result = self.client.get_file_info(path)
                self._assert_traversal_blocked(result)
                self.client.ftp_connection.size.assert_not_called()

    def test_get_file_info_sftp_remote_path_traversal(self):
        for path in self.TRAVERSAL_PATHS:
            with self.subTest(path=path):
                result = self.sftp_client.get_file_info(path)
                self._assert_traversal_blocked(result)
                self.sftp_client.sftp_connection.open_sftp.assert_not_called()

    # --- null byte injection ---

    def test_null_byte_in_remote_path_is_blocked(self):
        for method, kwargs in [
            (self.client.upload_file, {"local_path": "f.txt", "remote_path": "up\x00load.txt"}),
            (self.client.download_file, {"remote_path": "down\x00load.txt", "local_path": "f.txt"}),
            (self.client.list_directory, {"remote_path": "di\x00r"}),
            (self.client.create_directory, {"remote_path": "di\x00r"}),
            (self.client.remove_directory, {"remote_path": "di\x00r"}),
            (self.client.get_file_info, {"remote_path": "fi\x00le.txt"}),
        ]:
            with self.subTest(method=method.__name__):
                result = method(**kwargs)
                self.assertFalse(result['success'])
                self.assertIn("Invalid remote path", result.get('error', ''))


if __name__ == '__main__':
    unittest.main()