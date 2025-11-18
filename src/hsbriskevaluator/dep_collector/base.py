"""
collect dependencies recursively for a specific package
"""
from typing import Dict

class BaseDepCollector():
    platform: str 
    def __init__(self):
        pass

    def collect(self, package_name: str) -> Dict:
        """
        Collect dependency information for a package
        
        Args:
            package_name: Name of the package to collect dependencies for
            
        Returns:
            Dict containing dependency information
        """
        raise NotImplementedError("Subclasses must implement collect method")
