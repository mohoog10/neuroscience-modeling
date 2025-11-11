import json
from pathlib import Path
import sys

from src.interface.interface_cli import InterfaceCLI
from src.registry.model_registry import ModelRegistry
from src.manager.manager import Manager
from src.processing.datapreprocessor import DataPreprocessor
from src.models.kmeansmodel import KMeansModel
from src.models.logistic_model import LogisticRegressionModel
from src.models.rfc_model import RandomForestClassifierModel
from src.models.keras_model import KerasClassifierModel
from util_functions import *

def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = p.read_text(encoding="utf-8").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Config file must contain JSON. Error: {e}")


def inject_splits_and_preprocessor(model_cfg: dict, pre: DataPreprocessor, splits: dict) -> dict:
    """
    Return a shallow copy of model_cfg with runtime arrays and preprocessor injected.
    Models expect keys: X_train, y_train, X_val, y_val, X_test, y_test, _internal_scaler, _feature_columns
    """
    cfg = dict(model_cfg)  # shallow copy, keep original intact
    cfg["X_train"] = splits.get("X_train")
    cfg["y_train"] = splits.get("y_train")
    cfg["X_val"] = splits.get("X_val")
    cfg["y_val"] = splits.get("y_val")
    cfg["X_test"] = splits.get("X_test")
    cfg["y_test"] = splits.get("y_test")
    # Preprocessor may expose an internal scaler or full transformer
    cfg["_internal_scaler"] = getattr(pre, "_internal_scaler", None)
    cfg["_feature_columns"] = getattr(pre, "_feature_columns", None)
    cfg["target_column"] = pre.config.get("target_column") if getattr(pre, "config", None) else None
    return cfg


def main():
    cli = InterfaceCLI()
    cli.setup()
    args = cli.run("runner")

    if args is None:
        print("No valid CLI arguments; falling back to defaults.")
        #return
    model_name = getattr(args, "model", None)# or "KMeans"
    mode = getattr(args, "mode", None) or "train"
    config_path = config_path = getattr(args, "config", None)
    if config_path is None:
        config_path = "default_config.json"
        print('No config included. Using default')

    try:
        combined_config = load_config(config_path)
    except Exception as exc:
        print("Failed to load config:", exc)
        return

    # -- Registry and model registration
    registry = ModelRegistry()
    registry.register_model("kmeans", KMeansModel)
    # register supervised models if their modules exist
    try:
        registry.register_model("logistic", LogisticRegressionModel)
    except Exception:
        pass
    try:
        registry.register_model("random_forest", RandomForestClassifierModel)
    except Exception:
        pass
    try:
        registry.register_model("keras_classifier", KerasClassifierModel)
    except Exception:
        pass
    # -- Preprocessor (single instance)
    try:
        pre = DataPreprocessor(combined_config.get("data", {}))
    except Exception as exc:
        print("Failed to create DataPreprocessor:", exc)
        return

    # -- Manager
    manager = Manager(interface=cli, model_registry=registry, preprocessor=pre)

    # -- Fit / split once, attach splits
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = pre.fit_split()
    except Exception as exc:
        print("Data preprocessing / split failed:", exc)
        return

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }
    grid = combined_config.get('gridsearch', None)
    if grid:
        if grid.get('enabled'):
            gridsearch_run(combined_config,mode,
                        manager,pre,splits)
        else:
            one_model_run(combined_config,model_name,
                        mode,manager,pre,splits)
    else:
        one_model_run(combined_config,model_name,
                        mode,manager,pre,splits)


if __name__ == "__main__":
    main()