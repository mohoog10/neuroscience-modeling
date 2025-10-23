# Neuroscience Modeling Framework

A Docker-based neuroscience modeling framework with clean architecture for model training, validation, and testing.

## 🏗️ Architecture

The framework follows a clean architecture pattern with the following components:

```
neuroscience-modeling/
│
├── src/
│   ├── interface/          # Interface layer
│   │   ├── interface.py    # Abstract interface base class
│   │   └── interface_cli.py # CLI implementation
│   │
│   ├── manager/            # Manager layer
│   │   └── manager.py      # Coordinates interface and registry
│   │
│   ├── registry/           # Model registry
│   │   └── model_registry.py # Model selection and management
│   │
│   └── models/             # Model implementations
│       ├── model.py        # Abstract model base class
│       ├── model1.py       # Simple neural network model
│       └── model2.py       # Convolutional network model
│
├── main.py                 # Application entry point
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
└── requirements.txt        # Python dependencies
```

## 🚀 Features

- **Clean Architecture**: Separation of concerns with abstract interfaces and implementations
- **Model Registry**: Easy registration and selection of different models
- **CLI Interface**: Command-line interface for model operations
- **Docker Support**: Containerized deployment for consistency
- **Multiple Models**: Includes two example models (Model1 and Model2)
- **Complete Pipeline**: Support for training, validation, testing, and prediction

## 📋 Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (version 1.29 or higher)

OR

- Python 3.11+
- pip

## 🔧 Installation

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/mohoog10/neuroscience-modeling.git
cd neuroscience-modeling
```

2. Build and run with Docker Compose:
```bash
docker-compose up --build
```

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/mohoog10/neuroscience-modeling.git
cd neuroscience-modeling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## 💻 Usage

### Demo Mode

Run without arguments to see a demonstration of both models:

```bash
python main.py
```

### CLI Mode

Use command-line arguments to specify model and operation mode:

```bash
# Train Model1
python main.py --model Model1 --mode train

# Validate Model2
python main.py --model Model2 --mode validate

# Test a model
python main.py --model Model1 --mode test
```

### Available Options

- `--model`: Model to use (Model1, Model2)
- `--mode`: Operation mode (train, validate, test)
- `--config`: Path to configuration file (optional)

### Docker Usage

Run the container:
```bash
docker-compose up
```

Run with custom command:
```bash
docker-compose run neuroscience-modeling python main.py --model Model1 --mode train
```

Stop the container:
```bash
docker-compose down
```

## 🧪 Models

### Model1 - Simple Neural Network
- Basic neural network implementation
- Suitable for simple classification tasks
- Fast training and inference

**Configuration Options:**
- `learning_rate`: Learning rate (default: 0.01)
- `epochs`: Number of training epochs (default: 100)

### Model2 - Convolutional Network
- Convolutional neural network architecture
- Better for complex pattern recognition
- Includes multiple metrics (accuracy, precision, recall, F1-score)

**Configuration Options:**
- `learning_rate`: Learning rate (default: 0.001)
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Batch size for training (default: 32)

## 📊 Example Output

```
============================================================
Neuroscience Modeling Framework
============================================================

Model 'Model1' registered successfully
Model 'Model2' registered successfully

Available models: ['Model1', 'Model2']

============================================================

Running in DEMO mode...

==================================================
Running Pipeline: Model1 in train mode
==================================================

Selecting model: Model1
Model 'Model1' selected successfully
Building Model1...
Model1 built successfully with shape (10, 5)
Training Model1...
Epoch 0/50, Loss: 1.0000
Epoch 20/50, Loss: 0.0476
Epoch 40/50, Loss: 0.0244
Model1 training completed!
...
```

## 🛠️ Extending the Framework

### Adding a New Model

1. Create a new model class inheriting from `Model`:

```python
from src.models.model import Model

class Model3(Model):
    def __init__(self):
        super().__init__()
        self.name = "Model3"
    
    def build(self, model):
        # Implementation
        pass
    
    def train(self):
        # Implementation
        pass
    
    # Implement other required methods...
```

2. Register the model in `main.py`:

```python
from src.models.model3 import Model3

def initialize_registry():
    registry = ModelRegistry()
    registry.register_model('Model1', Model1)
    registry.register_model('Model2', Model2)
    registry.register_model('Model3', Model3)  # Add new model
    return registry
```

### Adding a New Interface

1. Create a new interface class inheriting from `Interface`:

```python
from src.interface.interface import Interface

class InterfaceGUI(Interface):
    def setup(self):
        # Implementation
        pass
    
    def run(self, name):
        # Implementation
        pass
```

2. Use it in the Manager initialization.

## 📝 Project Structure Details

### Interface Layer
- **interface.py**: Abstract base class defining the interface contract
- **interface_cli.py**: Command-line interface implementation using argparse

### Manager Layer
- **manager.py**: Orchestrates the interaction between interface and models
- Handles model selection, building, and execution of operations

### Registry Layer
- **model_registry.py**: Manages model registration and retrieval
- Provides a centralized way to access available models

### Models Layer
- **model.py**: Abstract base class for all models
- **model1.py**: Simple neural network implementation
- **model2.py**: Convolutional network implementation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

mohoog10

## 🔗 Links

- GitHub: [mohoog10](https://github.com/mohoog10)
- Repository: [neuroscience-modeling](https://github.com/mohoog10/neuroscience-modeling)

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Python Documentation](https://docs.python.org/3/)
- [NumPy Documentation](https://numpy.org/doc/)

## 🐛 Known Issues

None at this time. Please report any issues on the GitHub issue tracker.

## 🎯 Future Enhancements

- [ ] Add GUI interface
- [ ] Implement data loading utilities
- [ ] Add model checkpointing
- [ ] Include visualization tools
- [ ] Add more model architectures
- [ ] Implement hyperparameter tuning
- [ ] Add logging and monitoring
- [ ] Create comprehensive test suite
