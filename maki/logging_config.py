"""
Logging configuration for Maki framework
"""
import logging
import os

def configure_logging(log_level=logging.INFO):
    """Configure logging for the Maki framework

    Args:
        log_level: The logging level to use (default: INFO)
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Setup logging configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'maki.log')),
            logging.StreamHandler()
        ]
    )

    # Set log level for requests and urllib3
    logging.getLogger('requests').setLevel(log_level)
    logging.getLogger('urllib3').setLevel(log_level)

# Setup default logging
configure_logging()