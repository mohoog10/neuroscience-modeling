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
    model_name = getattr(args, "model", None) or "KMeans"
    mode = getattr(args, "mode", None) or "train"
    config_path = config_path = getattr(args, "config", None)
    if config_path is None:
        config_path = "test4_config.json"
        print('No config included. Using default')

    try:
        combined_config = load_config(config_path)
    except Exception as exc:
        print("Failed to load config:", exc)
        return

    # -- Registry and model registration
    registry = ModelRegistry()
    registry.register_model("KMeans", KMeansModel)
    # register supervised models if their modules exist
    try:
        registry.register_model("LogisticRegression", LogisticRegressionModel)
    except Exception:
        pass
    try:
        registry.register_model("RandomForest", RandomForestClassifierModel)
    except Exception:
        pass
    try:
        registry.register_model("KerasClassifier", KerasClassifierModel)
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

    # Select the model config dict by name from combined_config["models"] if present,
    # otherwise use combined_config["model"] for backward compatibility.
    models_cfg_list = combined_config.get("models")
    if models_cfg_list:
        selected_cfg = None
        for m in models_cfg_list:
            if m.get("name") == model_name:
                selected_cfg = m
                break
        if selected_cfg is None:
            print(f"Model '{model_name}' not found in combined_config['models']. Available:", [m.get("name") for m in models_cfg_list])
            return
    else:
        # backward compat: single model under "model" key
        selected_cfg = combined_config.get("model", {})
        # if no name present, ensure manager can still select via model_name
        if "name" not in selected_cfg:
            selected_cfg = dict(selected_cfg)
            selected_cfg["name"] = model_name

    # inject runtime splits and preprocessor
    runtime_cfg = inject_splits_and_preprocessor(selected_cfg, pre, splits)

    # run pipeline
    try:
        results = manager.run_pipeline(model_name, mode=mode, config=runtime_cfg)
    except Exception as exc:
        print("Pipeline run failed:", exc)
        return

    print("Training results:", results)

    # run predict/validate/test in a safe sequence if requested or by default
    #try:
        # Predict (if model supports it)
    pred_res = manager.run_pipeline(model_name, mode="predict", config=runtime_cfg)
    print("Prediction results:", pred_res)
    #except Exception as exc:
    #    print("Predict step failed (continuing):", exc)

   #try:
    val_res = manager.run_pipeline(model_name, mode="validate", config=runtime_cfg)
    print("Validation results:", val_res)
    #except Exception as exc:
    #    print("Validate step failed (continuing):", exc)

    # Plot test set using model method if available
    #model = manager.current_model
    #if hasattr(model, "plot_low_dim_using_config") and getattr(manager, "preprocessor", None) is not None:
    #    try:
    #        fig = model.plot_low_dim_using_config(split="test",
    #                                             method="pca",
    #                                             preprocessor=manager.preprocessor,
    #                                             sample=2000,
    #                                             title="Test set PCA")
    #    except Exception as exc:
    #        print("Plotting failed:", exc)

if __name__ == "__main__":
    main()
