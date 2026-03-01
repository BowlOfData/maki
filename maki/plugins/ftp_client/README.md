# FTP Client Plugin for Maki Framework

This plugin provides functionality to connect to FTP and SFTP servers for file operations. It allows agents to upload and download files, list directories, remove folders, and create directories on remote servers.

## Features

- **FTP Support**: Connect to plain FTP servers
- **SFTP Support**: Connect to secure SFTP servers using SSH keys or passwords
- **File Operations**: Upload and download files
- **Directory Operations**: List, create, and remove directories
- **Error Handling**: Comprehensive error handling and logging
- **Flexible Authentication**: Support for both password and certificate-based authentication for SFTP

## Installation

The plugin requires the following Python packages to be installed:

```bash
pip install ftplib paramiko
```

## Usage

### Basic Usage

```python
from maki.maki import Maki
from maki.plugins.ftp_client.ftp_client import FTPClient

# Initialize Maki
maki = Maki("http://localhost", 11434, "llama3")

# Initialize the FTP client plugin
ftp_client = FTPClient(maki)

# Connect to FTP server
connect_result = ftp_client.connect(
    host="ftp.example.com",
    username="user",
    password="password",
    protocol="ftp"
)

# Upload a file
upload_result = ftp_client.upload_file("/local/file.txt", "/remote/file.txt")

# Download a file
download_result = ftp_client.download_file("/remote/file.txt", "/local/file.txt")

# List directory contents
list_result = ftp_client.list_directory("/remote/dir")

# Disconnect
disconnect_result = ftp_client.disconnect()
```

### SFTP Usage with Certificate

```python
# Connect to SFTP server with certificate
connect_result = ftp_client.connect(
    host="sftp.example.com",
    username="user",
    protocol="sftp",
    cert_path="/path/to/private_key.pem"
)
```

## Methods

### `connect(host, username, password=None, port=None, protocol='ftp', cert_path=None)`

Connect to an FTP or SFTP server.

### `disconnect()`

Disconnect from the server.

### `upload_file(local_path, remote_path)`

Upload a file to the remote server.

### `download_file(remote_path, local_path)`

Download a file from the remote server.

### `list_directory(remote_path='.')`

List the contents of a directory on the remote server.

### `create_directory(remote_path)`

Create a directory on the remote server.

### `remove_directory(remote_path, recursive=False)`

Remove a directory on the remote server.

### `get_file_info(remote_path)`

Get information about a file or directory on the remote server.

## Response Format

All methods return a dictionary with the following structure:
- `success`: Boolean indicating if the operation was successful
- `error`: Error message if operation failed (None if successful)
- Additional fields specific to each method

## Supported Protocols

- `ftp`: Plain FTP connection
- `sftp`: Secure SFTP connection using SSH

## Authentication

### FTP
- Username and password required

### SFTP
- Username required
- Password or certificate path required (certificate path takes precedence)