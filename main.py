"""
Main Application Entry Point
Neuroscience Modeling Framework
"""
import tkinter as tk
import sys
from src.interface.interface_cli import InterfaceCLI
from src.interface.interface_tkinter import InterFaceTkinter
from src.manager.manager import Manager
from src.registry.model_registry import ModelRegistry
from src.models.model1 import Model1
from src.models.model2 import Model2
from src.models.kmeansmodel import KMeansModel


def initialize_registry():
    """Initialize and populate the model registry"""
    registry = ModelRegistry()
    
    # Register available models
    #registry.register_model('Model1', Model1)
    #registry.register_model('Model2', Model2)
    registry.register_model('KMeans', KMeansModel)
    
    return registry


def main():
    """Main application function"""
    print("\n" + "="*60)
    print("Neuroscience Modeling Framework")
    print("="*60 + "\n")
    
    registry = initialize_registry()


    # Initialize components
    root = tk.Tk()
    #app = (root)
    interface = InterFaceTkinter(root)
    interface.setup()

    if interface.result is None:
        print('Training stopped, No config')
        sys.exit(0)   
    print('YO')
    
    manager = Manager(interface, registry)
    results = manager.run_pipeline(interface.model_type.get(),mode='train',config=interface.result)
    print(results)

if __name__ == "__main__":
    main()
