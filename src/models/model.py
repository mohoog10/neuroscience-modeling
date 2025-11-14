"""
Abstract Model Module
Defines the base model class for neuroscience modeling
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class Model(ABC):
    """Abstract base class for model implementations"""
    
    def __init__(self):
        self.config = {}
        self.model = None
        self.is_built = False
        self.is_validated = False
        self.is_predicted = False
        self.is_tested = False
    
    @abstractmethod
    def build(self, model: Any,return_estimator: bool = False) -> bool:
        """
        Build the model
        
        Args:
            model: Model configuration or instance
            
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    def train(self) -> Dict:
        """
        Train the model
        
        Returns:
            dict: Training results
        """
        pass
    
    @abstractmethod
    def validate(self) -> Dict:
        """
        Validate the model
        
        Returns:
            dict: Validation results
        """
        pass
    
    @abstractmethod
    def predict(self) -> Dict:
        """
        Make predictions with the model
        
        Returns:
            dict: Prediction results
        """
        pass
    
    @abstractmethod
    def test(self) -> Dict:
        """
        Test the model
        
        Returns:
            dict: Test results
        """
        pass
    
    @abstractmethod
    def run_optuna_search(self,n_trials:int):
        pass