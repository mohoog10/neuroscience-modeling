"""
Model Registry Module
Enables model selection for training, validation, test, etc.
"""
from typing import Dict, Type, Optional
from ..models.model import Model


class ModelRegistry:
    """Registry for managing available models"""
    
    def __init__(self):
        self._models: Dict[str, Type[Model]] = {}
        self._instances: Dict[str, Model] = {}
    
    def register_model(self, name: str, model_class: Type[Model]) -> None:
        """
        Register a model class
        
        Args:
            name: Name identifier for the model
            model_class: Model class to register
        """
        if name in self._models:
            print(f"Warning: Model '{name}' already registered. Overwriting...")
        
        self._models[name] = model_class
        print(f"Model '{name}' registered successfully")
    
    def get_model(self, name: str) -> Optional[Model]:
        """
        Get a model instance by name
        
        Args:
            name: Name of the model to retrieve
            
        Returns:
            Model instance or None if not found
        """
        if name not in self._models:
            print(f"Error: Model '{name}' not found in registry")
            return None
        
        # Create new instance if not cached
        if name not in self._instances:
            self._instances[name] = self._models[name]()
        
        return self._instances[name]
    
    def list_models(self) -> list:
        """
        List all registered models
        
        Returns:
            List of registered model names
        """
        return list(self._models.keys())
    
    def model_exists(self, name: str) -> bool:
        """
        Check if a model exists in registry
        
        Args:
            name: Model name to check
            
        Returns:
            bool: True if model exists
        """
        return name in self._models
    
    def clear_instances(self) -> None:
        """Clear all cached model instances"""
        self._instances.clear()
        print("All model instances cleared")
