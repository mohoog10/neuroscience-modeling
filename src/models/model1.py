"""
Model1 Implementation
Example neuroscience model using simple neural network
"""
from .model import Model
import numpy as np
from typing import Dict, Any


class Model1(Model):
    """First example model implementation - Simple Neural Network"""
    
    def __init__(self):
        super().__init__()
        self.name = "Model1"
        self.weights = None
        self.biases = None
        self.learning_rate = 0.01
        self.epochs = 100
    
    def build(self, model: Any) -> bool:
        """
        Build Model1 with simple architecture
        
        Args:
            model: Model configuration dictionary
            
        Returns:
            bool: Success status
        """
        print(f"Building {self.name}...")
        
        if isinstance(model, dict):
            self.config = model
            self.learning_rate = model.get('learning_rate', 0.01)
            self.epochs = model.get('epochs', 100)
        
        # Initialize simple weights and biases
        self.weights = np.random.randn(10, 5) * 0.01
        self.biases = np.zeros((1, 5))
        
        self.is_built = True
        print(f"{self.name} built successfully with shape {self.weights.shape}")
        return True
    
    def train(self) -> Dict:
        """
        Train Model1
        
        Returns:
            dict: Training metrics
        """
        print(f"Training {self.name}...")
        
        if not self.is_built:
            print("Error: Model not built. Call build() first.")
            return {"error": "Model not built"}
        
        # Simulate training
        losses = []
        for epoch in range(self.epochs):
            # Simulate loss decrease
            loss = 1.0 / (epoch + 1)
            losses.append(loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss:.4f}")
        
        results = {
            "model": self.name,
            "epochs": self.epochs,
            "final_loss": losses[-1],
            "losses": losses
        }
        
        print(f"{self.name} training completed!")
        return results
    
    def validate(self) -> Dict:
        """
        Validate Model1
        
        Returns:
            dict: Validation metrics
        """
        print(f"Validating {self.name}...")
        
        if not self.is_built:
            return {"error": "Model not built"}
        
        # Simulate validation
        accuracy = np.random.uniform(0.85, 0.95)
        
        results = {
            "model": self.name,
            "accuracy": accuracy,
            "validation_loss": 0.15
        }
        
        self.is_validated = True
        print(f"{self.name} validation completed! Accuracy: {accuracy:.4f}")
        return results
    
    def predict(self) -> Dict:
        """
        Make predictions with Model1
        
        Returns:
            dict: Prediction results
        """
        print(f"Making predictions with {self.name}...")
        
        if not self.is_built:
            return {"error": "Model not built"}
        
        # Simulate predictions
        predictions = np.random.randn(10, 5)
        
        results = {
            "model": self.name,
            "predictions": predictions.tolist(),
            "num_predictions": len(predictions)
        }
        
        self.is_predicted = True
        print(f"{self.name} predictions completed!")
        return results
    
    def test(self) -> Dict:
        """
        Test Model1
        
        Returns:
            dict: Test metrics
        """
        print(f"Testing {self.name}...")
        
        if not self.is_built:
            return {"error": "Model not built"}
        
        # Simulate testing
        test_accuracy = np.random.uniform(0.80, 0.90)
        test_loss = np.random.uniform(0.10, 0.20)
        
        results = {
            "model": self.name,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss
        }
        
        self.is_tested = True
        print(f"{self.name} testing completed! Test Accuracy: {test_accuracy:.4f}")
        return results
