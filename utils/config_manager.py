# utils/config_manager.py
import os
import yaml

def load_config():
    """
    Load configuration from YAML file.
    Uses environment-specific config if available.
    
    Returns:
    --------
    dict
        Configuration dictionary
    """
    # Get environment
    env = os.environ.get('ENVIRONMENT', 'development')
    
    # Try to load environment-specific config
    config_path = f"config/config.{env}.yaml"
    
    # If environment-specific config doesn't exist, use default
    if not os.path.exists(config_path):
        config_path = "config/config.yaml"
        
    # Load and return config
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


class ConfigManager:
    """
    Singleton class to manage application configuration.
    This ensures configuration is loaded once and reused.
    """
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from YAML file."""
        self._config = load_config()
    
    def get_config(self):
        """
        Get the configuration dictionary.
        
        Returns:
        --------
        dict
            Configuration dictionary
        """
        return self._config
    
    def get_value(self, key, default=None):
        """
        Get a configuration value by key.
        
        Parameters:
        -----------
        key : str
            Configuration key
        default : any
            Default value if key doesn't exist
            
        Returns:
        --------
        any
            Configuration value or default
        """
        return self._config.get(key, default)