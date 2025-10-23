# Quick Start Guide

Get up and running with the Neuroscience Modeling Framework in 5 minutes!

## Option 1: Docker (Easiest)

```bash
# Clone the repository
git clone https://github.com/mohoog10/neuroscience-modeling.git
cd neuroscience-modeling

# Run with Docker Compose
docker-compose up --build
```

That's it! The demo will run automatically.

## Option 2: Local Python

```bash
# Clone the repository
git clone https://github.com/mohoog10/neuroscience-modeling.git
cd neuroscience-modeling

# Install dependencies
pip install -r requirements.txt

# Run the demo
python main.py
```

## Quick Examples

### Run Demo Mode
```bash
python main.py
```

### Train Model1
```bash
python main.py --model Model1 --mode train
```

### Validate Model2
```bash
python main.py --model Model2 --mode validate
```

### Test the Framework
```bash
python test.py
```

## Understanding the Output

When you run the demo, you'll see:

1. **Initialization**: Registry setup and model registration
2. **Model1 Pipeline**: Building, training, validation, and testing
3. **Model2 Pipeline**: Building, training, validation, and prediction
4. **Results**: Metrics and performance data for each model

## Next Steps

1. Check out `README.md` for detailed documentation
2. Explore the code in `src/` directory
3. Try modifying the configuration in `main.py`
4. Add your own model by following the examples in `src/models/`

## Common Issues

**Import Error**: Make sure you're in the project root directory

**Docker Error**: Ensure Docker is running and you have proper permissions

**Missing Dependencies**: Run `pip install -r requirements.txt`

## Need Help?

- Check the full README.md
- Open an issue on GitHub
- Review the example code in the src/ directory
