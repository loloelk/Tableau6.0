# utils/logging_config.py
import logging
import os
from datetime import datetime

def configure_logging():
    """
    Configure logging for the application.
    Creates a log directory if it doesn't exist and sets up logging.
    """
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Create a log file with current date
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f'app_{today}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log application startup
    logging.info("Application started")

def setup_logger(name):
    """
    Set up a logger with the given name.
    Args:
        name: Name of the logger
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Create a log file with current date
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(log_dir, f'app_{today}.log')

    # Create logger
    logger = logging.getLogger(name)
    
    # Only configure if handlers are not already added
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger