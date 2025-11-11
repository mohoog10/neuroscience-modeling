import json
from pathlib import Path
import sys
from src.interface.interface_cli import InterfaceCLI  # adjust import to your package
from src.registry.model_registry import ModelRegistry  # adjust import
from src.manager.manager import Manager              # adjust import
from src.processing.datapreprocessor import DataPreprocessor  # if you want to reuse elsewhere
from src.models.kmeansmodel import KMeansModel
def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = p.read_text(encoding="utf-8").strip()
    # try JSON load
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Config file must contain JSON. Error: {e}")

def main():
    cli = InterfaceCLI()
    cli.setup()
    args = cli.run("runner")
    if args is None:
        print("No args parsed. Exiting.")
        return
    default_path = 'test_config.json'
    model_name = args.model or "KMeans"   # default model name if not passed
    mode = args.mode or "train"
    config_path = args.config
    combined_config = {}
    if config_path:
        combined_config = load_config(config_path)
    else:
        print("No config path provided; using defaults.")
        #combined_config = {
        #    "data": {
        #        "filepath": "data/advertising.csv",
        #        "target_column": None,
        #        "dummy": True,
        #        "scaler": "standard",
        #        "test_size": 0.2,
        #        "val_size": 0.1,
        #        "random_state": 42,
        #        "return_type": "numpy"
        #    },
        #    "model": {
        #        "n_clusters": 8,
        #        "max_iter": 300,
        #        "tol": 1e-4,
        #        "random_state": 0
        #    }
        #}
        combined_config = load_config(default_path)
    # create registry and manager (make sure ModelRegistry has your Kmeans model registered)
    registry = ModelRegistry()
    registry.register_model('KMeans', KMeansModel)
    pre = DataPreprocessor(combined_config['data'])
    # ensure your models are registered here; e.g., registry.register("Kmeans", KMeansModelClass)
    manager = Manager(interface=cli, model_registry=registry, preprocessor = pre)
    


    X_train, y_train, X_val, y_val, X_test, y_test = pre.fit_split()
    combined_config['model']['X_train'] = X_train
    combined_config['model']['X_test'] = X_test
    combined_config['model']['X_val'] = X_val
    combined_config['model']['y_train'] = y_train
    combined_config['model']['y_test'] = y_test
    combined_config['model']['y_val'] = y_val

    results = manager.run_pipeline(model_name, mode=mode, config=combined_config['model'])
    print("Pipeline results:", results)
    #manager.current_model.predict()
    test = manager.run_pipeline(model_name, mode='predict', config=combined_config['model'])
    val = manager.run_pipeline(model_name, mode='validate', config=combined_config['model'])
    #print(test)
    print(val)
    # while model instance and preprocessor are available in Manager
    model = manager.current_model
    # call plotting, using Manager's preprocessor
    fig = model.plot_low_dim_using_config(split="test", method="pca", preprocessor=manager.preprocessor, sample=2000, title="Test set PCA")

if __name__ == "__main__":
    main()
