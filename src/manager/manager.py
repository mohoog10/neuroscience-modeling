"""
Manager Module
Manages interface and model registry coordination
"""
from typing import Optional, Dict, Any
from ..interface.interface import Interface
from ..registry.model_registry import ModelRegistry
from ..models.model import Model


class Manager:
    """
    Manager class that coordinates between interface and model registry
    """
    
    def __init__(self, interface: Interface, model_registry: ModelRegistry):
        """
        Initialize Manager
        
        Args:
            interface: Interface implementation (e.g., CLI)
            model_registry: Model registry instance
        """
        self.interface = interface
        self.model_registry = model_registry
        self.current_model: Optional[Model] = None
        
        # Configuration storage
        self.train_config: Dict[str, Any] = {}
        self.model_instance_name: Optional[str] = None
        self.model_instance_config: Dict[str, Any] = {}
        self.train_config: Dict[str, Any] = {}
        self.model_maintenance_name: Optional[str] = None
        self.model_maintenance_config: Dict[str, Any] = {}
        self.model_with_model_instance_name: Optional[str] = None
        self.model_with_model_instance_config: Dict[str, Any] = {}
        self.train_model_instance_name: Optional[str] = None
        self.train_model_instance_config: Dict[str, Any] = {}
        self.store_model_instance_name: Optional[str] = None
        self.store_model_instance_config: Dict[str, Any] = {}
        self.host_model_instance_name: Optional[str] = None
        
        self.new_model_name: Optional[str] = None
    
    def select_model(self, model_name: str, config: Optional[Dict] = None) -> bool:
        """
        Select and initialize a model
        
        Args:
            model_name: Name of the model to select
            config: Optional configuration dictionary
            
        Returns:
            bool: Success status
        """
        print(f"Selecting model: {model_name}")
        
        if not self.model_registry.model_exists(model_name):
            print(f"Error: Model '{model_name}' not found")
            return False
        
        self.current_model = self.model_registry.get_model(model_name)
        
        if config:
            self.model_instance_config = config
        
        print(f"Model '{model_name}' selected successfully")
        return True
    
    def build_model(self, config: Optional[Dict] = None) -> bool:
        """
        Build the currently selected model
        
        Args:
            config: Optional build configuration
            
        Returns:
            bool: Success status
        """
        if not self.current_model:
            print("Error: No model selected")
            return False
        
        build_config = config if config else self.model_instance_config
        return self.current_model.build(build_config)
    
    def train_model(self) -> Dict:
        """
        Train the current model
        
        Returns:
            dict: Training results
        """
        if not self.current_model:
            print("Error: No model selected")
            return {"error": "No model selected"}
        
        return self.current_model.train()
    
    def validate_model(self) -> Dict:
        """
        Validate the current model
        
        Returns:
            dict: Validation results
        """
        if not self.current_model:
            print("Error: No model selected")
            return {"error": "No model selected"}
        
        return self.current_model.validate()
    
    def test_model(self) -> Dict:
        """
        Test the current model
        
        Returns:
            dict: Test results
        """
        if not self.current_model:
            print("Error: No model selected")
            return {"error": "No model selected"}
        
        return self.current_model.test()
    
    def predict_with_model(self) -> Dict:
        """
        Make predictions with the current model
        
        Returns:
            dict: Prediction results
        """
        if not self.current_model:
            print("Error: No model selected")
            return {"error": "No model selected"}
        
        return self.current_model.predict()
    
    def run_pipeline(self, model_name: str, mode: str = 'train', config: Optional[Dict] = None) -> Dict:
        """
        Run a complete pipeline: select, build, and execute
        
        Args:
            model_name: Name of the model to use
            mode: Operation mode ('train', 'validate', 'test', 'predict')
            config: Optional configuration
            
        Returns:
            dict: Pipeline results
        """
        print(f"\n{'='*50}")
        print(f"Running Pipeline: {model_name} in {mode} mode")
        print(f"{'='*50}\n")
        
        # Select and build model
        if not self.select_model(model_name, config):
            return {"error": "Model selection failed"}
        
        if not self.build_model():
            return {"error": "Model build failed"}
        
        # Execute based on mode
        results = {}
        if mode == 'train':
            results = self.train_model()
        elif mode == 'validate':
            results = self.validate_model()
        elif mode == 'test':
            results = self.test_model()
        elif mode == 'predict':
            results = self.predict_with_model()
        else:
            results = {"error": f"Unknown mode: {mode}"}
        
        print(f"\n{'='*50}")
        print(f"Pipeline completed for {model_name}")
        print(f"{'='*50}\n")
        
        return results
    
    def get_available_models(self) -> list:
        """
        Get list of available models
        
        Returns:
            list: Available model names
        """
        return self.model_registry.list_models()
