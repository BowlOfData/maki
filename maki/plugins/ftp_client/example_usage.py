"""
Example usage of the FTPClient plugin with Maki agents

This file demonstrates how to use the FTPClient plugin to connect to FTP/SFTP servers
and perform various file operations.
"""

from maki.maki import Maki
from maki.plugins.ftp_client.ftp_client import FTPClient

# Initialize Maki
maki = Maki("http://localhost", 11434, "llama3")

# Initialize the FTP client plugin
ftp_client = FTPClient(maki)

def example_ftp_operations():
    """Demonstrate FTP operations"""
    print("FTP Client Plugin Example Usage")
    print("================================")

    # Example 1: Connect to FTP server
    print("\n1. Connecting to FTP server...")
    connect_result = ftp_client.connect(
        host="ftp.example.com",
        username="user",
        password="password",
        protocol="ftp"
    )

    if connect_result['success']:
        print("✓ Successfully connected to FTP server")

        # Example 2: List directory contents
        print("\n2. Listing directory contents...")
        list_result = ftp_client.list_directory()
        if list_result['success']:
            print(f"✓ Directory contents: {list_result['contents']}")
        else:
            print(f"✗ Failed to list directory: {list_result['error']}")

        # Example 3: Create a directory
        print("\n3. Creating directory...")
        mkdir_result = ftp_client.create_directory("/test_dir")
        if mkdir_result['success']:
            print("✓ Directory created successfully")
        else:
            print(f"✗ Failed to create directory: {mkdir_result['error']}")

        # Example 4: Disconnect
        print("\n4. Disconnecting from server...")
        disconnect_result = ftp_client.disconnect()
        if disconnect_result['success']:
            print("✓ Successfully disconnected")
        else:
            print(f"✗ Failed to disconnect: {disconnect_result['error']}")

    else:
        print(f"✗ Failed to connect to FTP server: {connect_result['error']}")

def example_sftp_operations():
    """Demonstrate SFTP operations"""
    print("\n\nSFTP Client Plugin Example Usage")
    print("================================")

    # Example 1: Connect to SFTP server with certificate
    print("\n1. Connecting to SFTP server with certificate...")
    connect_result = ftp_client.connect(
        host="sftp.example.com",
        username="user",
        password="password",  # Optional if using certificate
        protocol="sftp",
        cert_path="/path/to/certificate.pem"  # Required for certificate-based auth
    )

    if connect_result['success']:
        print("✓ Successfully connected to SFTP server")

        # Example 2: List directory contents
        print("\n2. Listing directory contents...")
        list_result = ftp_client.list_directory()
        if list_result['success']:
            print(f"✓ Directory contents: {list_result['contents']}")
        else:
            print(f"✗ Failed to list directory: {list_result['error']}")

        # Example 3: Create a directory
        print("\n3. Creating directory...")
        mkdir_result = ftp_client.create_directory("/test_dir")
        if mkdir_result['success']:
            print("✓ Directory created successfully")
        else:
            print(f"✗ Failed to create directory: {mkdir_result['error']}")

        # Example 4: Disconnect
        print("\n4. Disconnecting from server...")
        disconnect_result = ftp_client.disconnect()
        if disconnect_result['success']:
            print("✓ Successfully disconnected")
        else:
            print(f"✗ Failed to disconnect: {disconnect_result['error']}")

    else:
        print(f"✗ Failed to connect to SFTP server: {connect_result['error']}")

if __name__ == "__main__":
    print("FTPClient plugin example usage")
    print("================================")

    # Run FTP example
    example_ftp_operations()

    # Run SFTP example
    example_sftp_operations()

    print("\nPlugin is ready to be used with Maki agents")