import logging
import sys
import os
from datetime import datetime
from typing import Optional
import json

class AgriculturalAssistantLogger:
    """Enhanced logging configuration for the Agricultural Assistant"""
    
    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize logging configuration
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
        """
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_file = log_file or f"logs/assistant_{datetime.now().strftime('%Y%m%d')}.log"
        self.setup_logging()
    
    def setup_logging(self):
        """Setup comprehensive logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler
        try:
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create file handler: {e}")
        
        # Suppress noisy third-party loggers
        noisy_loggers = [
            'transformers',
            'langchain',
            'sentence_transformers',
            'urllib3',
            'requests',
            'gradio'
        ]
        
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    @staticmethod
    def log_system_info():
        """Log system information for debugging"""
        try:
            import platform
            import torch
            
            system_info = {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'torch_version': torch.__version__ if torch else 'Not installed',
                'cuda_available': torch.cuda.is_available() if torch else False,
                'cuda_device_count': torch.cuda.device_count() if torch and torch.cuda.is_available() else 0
            }
            
            logger = logging.getLogger(__name__)
            logger.info(f"System Info: {json.dumps(system_info, indent=2)}")
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not log system info: {e}")

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging for the entire application
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    logger_config = AgriculturalAssistantLogger(log_level, log_file)
    logger_config.log_system_info()
    return logger_config
