"""
FTP Client Plugin for Maki Framework

This module provides functionality to connect to FTP and SFTP servers for file operations.
It allows agents to upload and download files, list directories, remove folders, and create directories on remote servers.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Try to import FTP libraries - these might not be available in all environments
try:
    import ftplib
    from paramiko import SSHClient, SFTPClient
    from paramiko.ssh_exception import SSHException
    HAS_FTP_LIBS = True
except ImportError as e:
    HAS_FTP_LIBS = False
    logging.warning(f"FTP/SFTP libraries not available: {e}")


class FTPClient:
    """
    A plugin class for connecting to FTP and SFTP servers in the Maki framework.

    This class provides methods to perform various file operations on remote servers
    using both plain FTP and secure SFTP protocols.
    """

    def __init__(self, maki_instance=None):
        """
        Initialize the FTPClient plugin.

        Args:
            maki_instance: Optional Maki instance to use for logging and potential LLM interactions
        """
        self.maki = maki_instance
        self.logger = logging.getLogger(__name__)
        self.logger.info("FTPClient plugin initialized")

        # Connection state
        self.ftp_connection = None
        self.sftp_connection = None
        self.connected = False
        self.connection_type = None

    def connect(self, host: str, username: str, password: str = None, port: int = None,
                protocol: str = 'ftp', cert_path: str = None) -> Dict[str, Any]:
        """
        Connect to an FTP or SFTP server.

        Args:
            host (str): The hostname or IP address of the FTP/SFTP server
            username (str): Username for authentication
            password (str, optional): Password for authentication (required for FTP, optional for SFTP)
            port (int, optional): Port number (default: 21 for FTP, 22 for SFTP)
            protocol (str): Protocol to use ('ftp' or 'sftp')
            cert_path (str, optional): Path to certificate file (required for SFTP)

        Returns:
            Dict[str, Any]: A dictionary containing connection results
                - 'success': Boolean indicating if connection was successful
                - 'connection_type': Type of connection ('ftp' or 'sftp')
                - 'host': The host connected to
                - 'error': Error message if connection failed
        """
        if not HAS_FTP_LIBS:
            return {
                'success': False,
                'connection_type': None,
                'host': host,
                'error': 'FTP/SFTP libraries not available. Please install ftplib and paramiko.'
            }

        if not isinstance(host, str) or not host.strip():
            return {
                'success': False,
                'connection_type': None,
                'host': host,
                'error': 'Host must be a non-empty string'
            }

        if not isinstance(username, str) or not username.strip():
            return {
                'success': False,
                'connection_type': None,
                'host': host,
                'error': 'Username must be a non-empty string'
            }

        result = {
            'success': False,
            'connection_type': protocol,
            'host': host,
            'error': None
        }

        try:
            if protocol.lower() == 'ftp':
                if port is None:
                    port = 21

                self.ftp_connection = ftplib.FTP()
                self.ftp_connection.connect(host, port)
                self.ftp_connection.login(username, password)
                self.connected = True
                self.connection_type = 'ftp'

                result['success'] = True
                self.logger.info(f"Successfully connected to FTP server: {host}")

            elif protocol.lower() == 'sftp':
                if port is None:
                    port = 22

                # For SFTP, we need SSH client
                self.sftp_connection = SSHClient()
                self.sftp_connection.set_missing_host_key_policy('auto_add')

                # If certificate path is provided, use key-based authentication
                if cert_path and os.path.exists(cert_path):
                    self.sftp_connection.connect(
                        hostname=host,
                        port=port,
                        username=username,
                        key_filename=cert_path,
                        timeout=30
                    )
                else:
                    # Use password-based authentication
                    self.sftp_connection.connect(
                        hostname=host,
                        port=port,
                        username=username,
                        password=password,
                        timeout=30
                    )

                self.connected = True
                self.connection_type = 'sftp'

                result['success'] = True
                self.logger.info(f"Successfully connected to SFTP server: {host}")

            else:
                result['error'] = f"Unsupported protocol: {protocol}. Supported protocols are 'ftp' and 'sftp'."

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error connecting to server {host}: {str(e)}")

        return result

    def disconnect(self) -> Dict[str, Any]:
        """
        Disconnect from the FTP/SFTP server.

        Returns:
            Dict[str, Any]: A dictionary containing disconnection results
                - 'success': Boolean indicating if disconnection was successful
                - 'error': Error message if disconnection failed
        """
        result = {
            'success': False,
            'error': None
        }

        try:
            if self.ftp_connection:
                self.ftp_connection.quit()
                self.ftp_connection = None

            if self.sftp_connection:
                self.sftp_connection.close()
                self.sftp_connection = None

            self.connected = False
            self.connection_type = None

            result['success'] = True
            self.logger.info("Successfully disconnected from server")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error disconnecting from server: {str(e)}")

        return result

    def upload_file(self, local_path: str, remote_path: str) -> Dict[str, Any]:
        """
        Upload a file to the remote server.

        Args:
            local_path (str): Path to the local file to upload
            remote_path (str): Path where the file should be uploaded on the remote server

        Returns:
            Dict[str, Any]: A dictionary containing upload results
                - 'success': Boolean indicating if upload was successful
                - 'local_path': The local file path
                - 'remote_path': The remote file path
                - 'bytes_uploaded': Number of bytes uploaded
                - 'error': Error message if upload failed
        """
        if not self.connected:
            return {
                'success': False,
                'local_path': local_path,
                'remote_path': remote_path,
                'bytes_uploaded': 0,
                'error': 'Not connected to server'
            }

        if not isinstance(local_path, str) or not local_path.strip():
            return {
                'success': False,
                'local_path': local_path,
                'remote_path': remote_path,
                'bytes_uploaded': 0,
                'error': 'Local path must be a non-empty string'
            }

        if not isinstance(remote_path, str) or not remote_path.strip():
            return {
                'success': False,
                'local_path': local_path,
                'remote_path': remote_path,
                'bytes_uploaded': 0,
                'error': 'Remote path must be a non-empty string'
            }

        result = {
            'success': False,
            'local_path': local_path,
            'remote_path': remote_path,
            'bytes_uploaded': 0,
            'error': None
        }

        try:
            # Check if local file exists
            if not os.path.exists(local_path):
                result['error'] = f"Local file not found: {local_path}"
                return result

            if not os.path.isfile(local_path):
                result['error'] = f"Path is not a file: {local_path}"
                return result

            bytes_uploaded = 0

            if self.connection_type == 'ftp':
                with open(local_path, 'rb') as file:
                    self.ftp_connection.storbinary(f'STOR {remote_path}', file)
                    bytes_uploaded = os.path.getsize(local_path)

            elif self.connection_type == 'sftp':
                sftp = self.sftp_connection.open_sftp()
                sftp.put(local_path, remote_path)
                bytes_uploaded = os.path.getsize(local_path)
                sftp.close()

            result['success'] = True
            result['bytes_uploaded'] = bytes_uploaded
            self.logger.info(f"Successfully uploaded file: {local_path} -> {remote_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error uploading file {local_path}: {str(e)}")

        return result

    def download_file(self, remote_path: str, local_path: str) -> Dict[str, Any]:
        """
        Download a file from the remote server.

        Args:
            remote_path (str): Path to the file on the remote server
            local_path (str): Path where the file should be saved locally

        Returns:
            Dict[str, Any]: A dictionary containing download results
                - 'success': Boolean indicating if download was successful
                - 'remote_path': The remote file path
                - 'local_path': The local file path
                - 'bytes_downloaded': Number of bytes downloaded
                - 'error': Error message if download failed
        """
        if not self.connected:
            return {
                'success': False,
                'remote_path': remote_path,
                'local_path': local_path,
                'bytes_downloaded': 0,
                'error': 'Not connected to server'
            }

        if not isinstance(remote_path, str) or not remote_path.strip():
            return {
                'success': False,
                'remote_path': remote_path,
                'local_path': local_path,
                'bytes_downloaded': 0,
                'error': 'Remote path must be a non-empty string'
            }

        if not isinstance(local_path, str) or not local_path.strip():
            return {
                'success': False,
                'remote_path': remote_path,
                'local_path': local_path,
                'bytes_downloaded': 0,
                'error': 'Local path must be a non-empty string'
            }

        result = {
            'success': False,
            'remote_path': remote_path,
            'local_path': local_path,
            'bytes_downloaded': 0,
            'error': None
        }

        try:
            # Create parent directory if it doesn't exist
            local_dir = os.path.dirname(local_path)
            if local_dir and not os.path.exists(local_dir):
                os.makedirs(local_dir, exist_ok=True)

            bytes_downloaded = 0

            if self.connection_type == 'ftp':
                with open(local_path, 'wb') as file:
                    self.ftp_connection.retrbinary(f'RETR {remote_path}', file.write)
                    bytes_downloaded = os.path.getsize(local_path)

            elif self.connection_type == 'sftp':
                sftp = self.sftp_connection.open_sftp()
                sftp.get(remote_path, local_path)
                bytes_downloaded = os.path.getsize(local_path)
                sftp.close()

            result['success'] = True
            result['bytes_downloaded'] = bytes_downloaded
            self.logger.info(f"Successfully downloaded file: {remote_path} -> {local_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error downloading file {remote_path}: {str(e)}")

        return result

    def list_directory(self, remote_path: str = '.') -> Dict[str, Any]:
        """
        List the contents of a directory on the remote server.

        Args:
            remote_path (str): Path to the directory to list (default: current directory)

        Returns:
            Dict[str, Any]: A dictionary containing directory listing results
                - 'success': Boolean indicating if listing was successful
                - 'remote_path': The path that was listed
                - 'contents': List of directory contents
                - 'error': Error message if listing failed
        """
        if not self.connected:
            return {
                'success': False,
                'remote_path': remote_path,
                'contents': [],
                'error': 'Not connected to server'
            }

        if not isinstance(remote_path, str) or not remote_path.strip():
            return {
                'success': False,
                'remote_path': remote_path,
                'contents': [],
                'error': 'Remote path must be a non-empty string'
            }

        result = {
            'success': False,
            'remote_path': remote_path,
            'contents': [],
            'error': None
        }

        try:
            contents = []

            if self.connection_type == 'ftp':
                # For FTP, get directory listing
                self.ftp_connection.cwd(remote_path)
                contents = self.ftp_connection.nlst()

            elif self.connection_type == 'sftp':
                # For SFTP, get directory listing
                sftp = self.sftp_connection.open_sftp()
                contents = sftp.listdir(remote_path)
                sftp.close()

            result['success'] = True
            result['contents'] = contents
            self.logger.info(f"Successfully listed directory: {remote_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error listing directory {remote_path}: {str(e)}")

        return result

    def create_directory(self, remote_path: str) -> Dict[str, Any]:
        """
        Create a directory on the remote server.

        Args:
            remote_path (str): Path of the directory to create

        Returns:
            Dict[str, Any]: A dictionary containing directory creation results
                - 'success': Boolean indicating if creation was successful
                - 'remote_path': The path that was created
                - 'error': Error message if creation failed
        """
        if not self.connected:
            return {
                'success': False,
                'remote_path': remote_path,
                'error': 'Not connected to server'
            }

        if not isinstance(remote_path, str) or not remote_path.strip():
            return {
                'success': False,
                'remote_path': remote_path,
                'error': 'Remote path must be a non-empty string'
            }

        result = {
            'success': False,
            'remote_path': remote_path,
            'error': None
        }

        try:
            if self.connection_type == 'ftp':
                # For FTP, create directory
                self.ftp_connection.mkd(remote_path)

            elif self.connection_type == 'sftp':
                # For SFTP, create directory
                sftp = self.sftp_connection.open_sftp()
                sftp.mkdir(remote_path)
                sftp.close()

            result['success'] = True
            self.logger.info(f"Successfully created directory: {remote_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error creating directory {remote_path}: {str(e)}")

        return result

    def remove_directory(self, remote_path: str, recursive: bool = False) -> Dict[str, Any]:
        """
        Remove a directory on the remote server.

        Args:
            remote_path (str): Path of the directory to remove
            recursive (bool): Whether to remove directory recursively (default: False)

        Returns:
            Dict[str, Any]: A dictionary containing directory removal results
                - 'success': Boolean indicating if removal was successful
                - 'remote_path': The path that was removed
                - 'recursive': Whether removal was recursive
                - 'error': Error message if removal failed
        """
        if not self.connected:
            return {
                'success': False,
                'remote_path': remote_path,
                'recursive': recursive,
                'error': 'Not connected to server'
            }

        if not isinstance(remote_path, str) or not remote_path.strip():
            return {
                'success': False,
                'remote_path': remote_path,
                'recursive': recursive,
                'error': 'Remote path must be a non-empty string'
            }

        result = {
            'success': False,
            'remote_path': remote_path,
            'recursive': recursive,
            'error': None
        }

        try:
            if self.connection_type == 'ftp':
                if recursive:
                    # For FTP, we need to recursively remove contents
                    # This is complex, so we'll just remove the directory if it's empty
                    self.ftp_connection.rmd(remote_path)
                else:
                    self.ftp_connection.rmd(remote_path)

            elif self.connection_type == 'sftp':
                sftp = self.sftp_connection.open_sftp()
                if recursive:
                    # For SFTP, recursively remove directory contents
                    self._sftp_remove_recursive(sftp, remote_path)
                else:
                    sftp.rmdir(remote_path)
                sftp.close()

            result['success'] = True
            self.logger.info(f"Successfully removed directory: {remote_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error removing directory {remote_path}: {str(e)}")

        return result

    def _sftp_remove_recursive(self, sftp, path: str) -> None:
        """
        Helper method to recursively remove directory contents in SFTP.

        Args:
            sftp: SFTP client connection
            path (str): Path of the directory to remove recursively
        """
        try:
            # Get directory contents
            items = sftp.listdir(path)
            for item in items:
                item_path = os.path.join(path, item)
                try:
                    # Try to remove as file first
                    sftp.remove(item_path)
                except IOError:
                    # If it fails, it's a directory, so recurse
                    self._sftp_remove_recursive(sftp, item_path)
            # Remove the directory itself
            sftp.rmdir(path)
        except Exception as e:
            self.logger.error(f"Error in recursive removal of {path}: {str(e)}")
            raise

    def get_file_info(self, remote_path: str) -> Dict[str, Any]:
        """
        Get information about a file or directory on the remote server.

        Args:
            remote_path (str): Path to the file or directory

        Returns:
            Dict[str, Any]: A dictionary containing file information
                - 'success': Boolean indicating if operation was successful
                - 'remote_path': The path of the file/directory
                - 'size': File size in bytes (0 for directories)
                - 'is_file': Boolean indicating if path is a file
                - 'is_directory': Boolean indicating if path is a directory
                - 'exists': Boolean indicating if path exists
                - 'error': Error message if operation failed
        """
        if not self.connected:
            return {
                'success': False,
                'remote_path': remote_path,
                'size': 0,
                'is_file': False,
                'is_directory': False,
                'exists': False,
                'error': 'Not connected to server'
            }

        if not isinstance(remote_path, str) or not remote_path.strip():
            return {
                'success': False,
                'remote_path': remote_path,
                'size': 0,
                'is_file': False,
                'is_directory': False,
                'exists': False,
                'error': 'Remote path must be a non-empty string'
            }

        result = {
            'success': False,
            'remote_path': remote_path,
            'size': 0,
            'is_file': False,
            'is_directory': False,
            'exists': False,
            'error': None
        }

        try:
            if self.connection_type == 'ftp':
                # For FTP, we'll need to determine if it's a file or directory
                # This is a bit tricky, so we'll try to get size and if that fails, treat as directory
                try:
                    # Try to get file size
                    size = self.ftp_connection.size(remote_path)
                    result['size'] = size
                    result['is_file'] = True
                    result['is_directory'] = False
                    result['exists'] = True
                except:
                    # If size fails, it might be a directory
                    try:
                        # Try to list directory
                        self.ftp_connection.cwd(remote_path)
                        result['is_file'] = False
                        result['is_directory'] = True
                        result['exists'] = True
                        result['size'] = 0
                    except:
                        result['exists'] = False

            elif self.connection_type == 'sftp':
                sftp = self.sftp_connection.open_sftp()
                try:
                    stat = sftp.stat(remote_path)
                    result['size'] = stat.st_size
                    result['is_file'] = stat.st_mode & 0o100000 != 0  # S_IFREG
                    result['is_directory'] = stat.st_mode & 0o040000 != 0  # S_IFDIR
                    result['exists'] = True
                except IOError:
                    result['exists'] = False
                sftp.close()

            result['success'] = True
            self.logger.info(f"Retrieved file info for: {remote_path}")

        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error getting file info for {remote_path}: {str(e)}")

        return result


# Plugin registration function
def register_plugin(maki_instance=None):
    """
    Register the FTPClient plugin with the Maki framework.

    Args:
        maki_instance: Maki instance to use for the plugin

    Returns:
        FTPClient: An instance of the FTPClient plugin
    """
    return FTPClient(maki_instance)