"""
Main Application Entry Point
Neuroscience Modeling Framework
"""
import sys
from src.interface.interface_cli import InterfaceCLI
from src.manager.manager import Manager
from src.registry.model_registry import ModelRegistry
from src.models.model1 import Model1
from src.models.model2 import Model2


def initialize_registry():
    """Initialize and populate the model registry"""
    registry = ModelRegistry()
    
    # Register available models
    registry.register_model('Model1', Model1)
    registry.register_model('Model2', Model2)
    
    return registry


def main():
    """Main application function"""
    print("\n" + "="*60)
    print("Neuroscience Modeling Framework")
    print("="*60 + "\n")
    
    # Initialize components
    interface = InterfaceCLI()
    interface.setup()
    
    registry = initialize_registry()
    
    manager = Manager(interface, registry)
    
    print("\nAvailable models:", manager.get_available_models())
    print("\n" + "="*60 + "\n")
    
    # Example usage - Run different pipelines
    if len(sys.argv) > 1:
        # CLI mode with arguments
        interface.run("CLI Run")
    else:
        # Demo mode - showcase both models
        print("Running in DEMO mode...\n")
        
        # Demo Model1
        config1 = {
            'learning_rate': 0.01,
            'epochs': 50
        }
        results1 = manager.run_pipeline('Model1', mode='train', config=config1)
        
        # Validate Model1
        manager.validate_model()
        
        # Test Model1
        manager.test_model()
        
        print("\n" + "-"*60 + "\n")
        
        # Demo Model2
        config2 = {
            'learning_rate': 0.001,
            'epochs': 30,
            'batch_size': 32
        }
        results2 = manager.run_pipeline('Model2', mode='train', config=config2)
        
        # Validate Model2
        manager.validate_model()
        
        # Make predictions with Model2
        manager.predict_with_model()
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
