"""
Test Script for Neuroscience Modeling Framework
Verifies that all components are working correctly
"""
import sys
sys.path.insert(0, '.')

from src.interface.interface_cli import InterfaceCLI
from src.manager.manager import Manager
from src.registry.model_registry import ModelRegistry
from src.models.model1 import Model1
from src.models.model2 import Model2


def test_registry():
    """Test the model registry"""
    print("Testing Model Registry...")
    registry = ModelRegistry()
    
    # Register models
    registry.register_model('Model1', Model1)
    registry.register_model('Model2', Model2)
    
    # Check registration
    assert registry.model_exists('Model1'), "Model1 not found"
    assert registry.model_exists('Model2'), "Model2 not found"
    assert len(registry.list_models()) == 2, "Wrong number of models"
    
    print("✓ Model Registry tests passed\n")


def test_models():
    """Test model instantiation and basic operations"""
    print("Testing Models...")
    
    # Test Model1
    model1 = Model1()
    config1 = {'learning_rate': 0.01, 'epochs': 10}
    
    assert model1.build(config1), "Model1 build failed"
    assert model1.is_built, "Model1 not marked as built"
    
    train_results = model1.train()
    assert 'final_loss' in train_results, "Training results incomplete"
    
    print("✓ Model1 tests passed")
    
    # Test Model2
    model2 = Model2()
    config2 = {'learning_rate': 0.001, 'epochs': 10, 'batch_size': 16}
    
    assert model2.build(config2), "Model2 build failed"
    assert model2.is_built, "Model2 not marked as built"
    
    train_results = model2.train()
    assert 'final_accuracy' in train_results, "Training results incomplete"
    
    print("✓ Model2 tests passed\n")


def test_manager():
    """Test the manager functionality"""
    print("Testing Manager...")
    
    interface = InterfaceCLI()
    registry = ModelRegistry()
    registry.register_model('Model1', Model1)
    registry.register_model('Model2', Model2)
    
    manager = Manager(interface, registry)
    
    # Test model selection
    assert manager.select_model('Model1'), "Model selection failed"
    assert manager.current_model is not None, "No model selected"
    
    # Test pipeline
    results = manager.run_pipeline('Model1', mode='train', config={'epochs': 5})
    assert 'error' not in results, f"Pipeline failed: {results}"
    
    print("✓ Manager tests passed\n")


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("Running Neuroscience Modeling Framework Tests")
    print("="*60 + "\n")
    
    try:
        test_registry()
        test_models()
        test_manager()
        
        print("="*60)
        print("✓ All tests passed successfully!")
        print("="*60)
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
