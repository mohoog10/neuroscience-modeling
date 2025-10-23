"""
Abstract Interface Module
Defines the base interface for input/output operations
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class Interface(ABC):
    """Abstract base class for interface implementations"""
    
    def __init__(self):
        self.input_data = None
        self.output_data = None
    
    @abstractmethod
    def setup(self) -> None:
        """Initialize the interface"""
        pass
    
    @abstractmethod
    def run(self, name: str) -> bool:
        """
        Run the interface with given name
        
        Args:
            name: Name of the run
            
        Returns:
            bool: Success status
        """
        pass
