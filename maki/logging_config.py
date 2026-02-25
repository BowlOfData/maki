"""
Logging configuration for Maki framework
"""
import logging
import os

def configure_logging(log_level=logging.INFO, log_file_path=None):
    """Configure logging for the Maki framework

    Args:
        log_level: The logging level to use (default: INFO)
        log_file_path: Optional path to log file. If None, only StreamHandler is used.
    """
    # Setup logging configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Add file handler if log_file_path is provided
    if log_file_path:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file_path)
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path),
                logging.StreamHandler()
            ]
        )

    # Set log level for requests and urllib3
    logging.getLogger('requests').setLevel(log_level)
    logging.getLogger('urllib3').setLevel(log_level)