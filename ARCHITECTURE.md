# Architecture Documentation

## System Architecture Overview

The Neuroscience Modeling Framework follows a clean architecture pattern with clear separation of concerns and dependency inversion.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI Instance                           │
│                                                               │
│  ┌──────────────┐            ┌─────────────────────────┐   │
│  │ InterfaceCLI │───────────▶│    <<interface>>        │   │
│  │              │            │      Interface          │   │
│  │ -registry:   │            │                         │   │
│  │  ModelReg    │            │ +input: str             │   │
│  │ -modelReg:   │            │ +output: int            │   │
│  │  ModelReg    │            │                         │   │
│  │              │            │ +setup(): None          │   │
│  └──────────────┘            │ +run(name): bool        │   │
│         │                     └─────────────────────────┘   │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────┐      │
│  │              Architecture                        │      │
│  │                                                  │      │
│  │  ┌────────────────────────────────────────┐    │      │
│  │  │           Manager                      │    │      │
│  │  │                                        │    │      │
│  │  │ -interface: InterfaceCLI              │    │      │
│  │  │ -registry: ModelRegistry               │    │      │
│  │  │                                        │    │      │
│  │  │ +register_model(name: str, model_class: TypeModel): None  │
│  │  │ +get_model(name: str, model_class: TypeModel): None       │
│  │  │ -train_model(instance_name: str, epochs: str): None       │
│  │  │ +model_with_model_instance_name: str): None               │
│  │  │ +run_model(instance_name: str): None                      │
│  │  │ +store_model(instance_name: str): None                    │
│  │  │ +host_model(instance_name: str): None                     │
│  │  │                                        │    │      │
│  │  │ +new(model): None                      │    │      │
│  │  └──────────────────┬─────────────────────┘    │      │
│  │                     │                           │      │
│  │                     ▼                           │      │
│  │  ┌────────────────────────────────────────┐    │      │
│  │  │       ModelRegistry                    │    │      │
│  │  │                                        │    │      │
│  │  │ Enable the model selection for         │    │      │
│  │  │ training, validation, test, use, etc.  │    │      │
│  │  │                                        │    │      │
│  │  │ -model_classes: dict[str, TypeModel]  │    │      │
│  │  │ -model_instances: dict[str, Model]    │    │      │
│  │  │                                        │    │      │
│  │  │ +register_model(name: str, model_class: TypeModel): None │
│  │  │ +get_instance_model(name: str, model_class: TypeModel): None │
│  │  │ +train_model(instance_name: str, epochs: str): None │
│  │  │ +store_model(instance_name: str): None │
│  │  │ +host_model(instance_name: str): None │
│  │  └──────────────────┬─────────────────────┘    │      │
│  │                     │                           │      │
│  │                     ▼                           │      │
│  │  ┌────────────────────────────────────────┐    │      │
│  │  │   Neuron Modeling PackageClasses       │    │      │
│  │  │                                        │    │      │
│  │  │         provided Models                │    │      │
│  │  │         -Model1: Model2                │    │      │
│  │  └──────────────────┬─────────────────────┘    │      │
│  │                     │                           │      │
│  │                     ▼                           │      │
│  │  ┌────────────────────────────────────────┐    │      │
│  │  │       <<abstract>>                     │    │      │
│  │  │          Model                         │    │      │
│  │  │                                        │    │      │
│  │  │ +config: dict                          │    │      │
│  │  │ +model: None                           │    │      │
│  │  │                                        │    │      │
│  │  │ +build(model: t): bool                 │    │      │
│  │  │ +train(): dict                         │    │      │
│  │  │ +validate(): dict                      │    │      │
│  │  │ +predict(): dict                       │    │      │
│  │  │ +test(): dict                          │    │      │
│  │  └────────────────────────────────────────┘    │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### 1. Interface Layer

**Purpose**: Provides the entry point for user interaction

**Components**:
- `Interface` (Abstract): Defines the contract for all interfaces
- `InterfaceCLI` (Concrete): Command-line interface implementation

**Responsibilities**:
- Parse user input
- Display output
- Handle user commands
- Coordinate with Manager

### 2. Manager Layer

**Purpose**: Orchestrates the interaction between interface and models

**Components**:
- `Manager`: Central coordinator class

**Responsibilities**:
- Model selection and initialization
- Pipeline execution (train, validate, test, predict)
- Configuration management
- Coordination between interface and registry

**Key Methods**:
- `select_model()`: Choose a model from registry
- `build_model()`: Initialize the selected model
- `train_model()`: Execute training
- `validate_model()`: Execute validation
- `test_model()`: Execute testing
- `predict_with_model()`: Execute predictions
- `run_pipeline()`: Execute complete workflow

### 3. Registry Layer

**Purpose**: Manages available models and their instantiation

**Components**:
- `ModelRegistry`: Model registration and retrieval system

**Responsibilities**:
- Register new model classes
- Retrieve model instances
- Cache model instances
- List available models

**Key Methods**:
- `register_model()`: Add a model class to registry
- `get_model()`: Retrieve or instantiate a model
- `list_models()`: Get all available models
- `model_exists()`: Check if a model is registered

### 4. Models Layer

**Purpose**: Implements the actual neuroscience models

**Components**:
- `Model` (Abstract): Base class for all models
- `Model1`: Simple neural network implementation
- `Model2`: Convolutional network implementation

**Responsibilities**:
- Model building and initialization
- Training logic
- Validation logic
- Testing logic
- Prediction logic

**Key Methods** (all models must implement):
- `build()`: Construct the model architecture
- `train()`: Train the model
- `validate()`: Validate model performance
- `predict()`: Make predictions
- `test()`: Test model performance

## Design Patterns

### 1. Abstract Factory Pattern
- `ModelRegistry` acts as a factory for creating model instances
- Allows easy addition of new models without modifying existing code

### 2. Strategy Pattern
- Different models implement the same interface (`Model`)
- Models can be swapped at runtime

### 3. Template Method Pattern
- `Model` abstract class defines the workflow structure
- Concrete models implement specific steps

### 4. Dependency Inversion Principle
- High-level modules (Manager) depend on abstractions (Interface, Model)
- Low-level modules (InterfaceCLI, Model1, Model2) implement abstractions

## Data Flow

1. **User Input** → `InterfaceCLI`
2. `InterfaceCLI` → `Manager` (command execution)
3. `Manager` → `ModelRegistry` (model selection)
4. `ModelRegistry` → `Model` (instance creation)
5. `Manager` → `Model` (operations: build, train, validate, etc.)
6. `Model` → `Manager` (results)
7. `Manager` → `InterfaceCLI` (formatted output)
8. `InterfaceCLI` → **User Output**

## Extension Points

### Adding a New Model

1. Create a new class inheriting from `Model`
2. Implement all abstract methods
3. Register in `ModelRegistry` via `main.py`

```python
class Model3(Model):
    def build(self, model): ...
    def train(self): ...
    def validate(self): ...
    def predict(self): ...
    def test(self): ...
```

### Adding a New Interface

1. Create a new class inheriting from `Interface`
2. Implement `setup()` and `run()` methods
3. Pass to `Manager` constructor

```python
class InterfaceGUI(Interface):
    def setup(self): ...
    def run(self, name): ...
```

### Adding New Manager Functionality

1. Add methods to `Manager` class
2. Use existing `ModelRegistry` and `Interface` infrastructure
3. Maintain separation of concerns

## Configuration Management

Configuration flows through the system:

1. **Interface Level**: Parses user config (CLI args, files)
2. **Manager Level**: Stores and forwards config
3. **Model Level**: Applies config to model building/training

Example config structure:
```python
{
    'learning_rate': 0.01,
    'epochs': 100,
    'batch_size': 32,
    # Model-specific parameters
}
```

## Error Handling

Each layer handles errors at its level:

- **Interface**: User input validation
- **Manager**: Workflow errors, model selection issues
- **Registry**: Model not found, registration errors
- **Models**: Training failures, invalid configurations

## Testing Strategy

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **System Tests**: Test complete workflows

See `test.py` for examples.

## Performance Considerations

1. **Model Caching**: Registry caches model instances
2. **Lazy Loading**: Models instantiated only when needed
3. **Configuration Reuse**: Configs stored at manager level

## Future Enhancements

1. **Persistence Layer**: Save/load models
2. **Monitoring**: Training metrics tracking
3. **Distributed Training**: Multi-GPU support
4. **API Layer**: REST API interface
5. **Web Interface**: Browser-based GUI
