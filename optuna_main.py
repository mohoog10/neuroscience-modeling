import json
from pathlib import Path
import sys,time

from src.interface.interface_cli import InterfaceCLI
from src.registry.model_registry import ModelRegistry
from src.manager.manager import Manager
from src.processing.datapreprocessor import DataPreprocessor
from src.models.kmeansmodel import KMeansModel
from src.models.logistic_model import LogisticRegressionModel
from src.models.rfc_model import RandomForestClassifierModel
from src.models.keras_model import KerasClassifierModel

def progress_bar(current, total, bar_length=30):
    percent = float(current) / total
    arrow = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f"\r[{arrow}{spaces}] {int(percent*100)}%")
    sys.stdout.flush()

def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = p.read_text(encoding="utf-8").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Config file must contain JSON. Error: {e}")

def print_shapes(X_train, y_train, X_val, y_val, X_test, y_test):
    print("Dataset shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    
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
    mode = getattr(args, "mode", None) or "Tuning"
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
    #registry.register_model("kmeans", KMeansModel)
    # register supervised models if their modules exist
    #try:
    #    registry.register_model("logistic", LogisticRegressionModel)
    #except Exception:
    #    pass
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
    #try:
    X,y = pre.load_all_csvs()
    X_train, y_train, X_val, y_val, X_test, y_test = pre.split_files(X,y)
    #except Exception as exc:
        #print("Data preprocessing / split failed:", exc)
        #return

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }
    print_shapes(X_train, y_train, X_val, y_val, X_test, y_test)
    # --- Loop over models ---
    models_cfg_list = combined_config.get("models")
    if models_cfg_list:
        total_models = len(models_cfg_list)
        total_start = time.time()  # start total timer

        for idx, selected_cfg in enumerate(models_cfg_list, start=1):
            model_start = time.time()  # start model timer
            registry.clear_instances()
            # inject runtime splits and preprocessor
            runtime_cfg = inject_splits_and_preprocessor(selected_cfg, pre, splits)
            print(selected_cfg['name'],runtime_cfg['name'],selected_cfg['type'])
            if selected_cfg.get('build_mode') == 'grid':
                manager.run_tuning(
                    selected_cfg['type'],
                    mode=mode,
                    config=runtime_cfg,
                    n_trials=30
                )

            # end model timer
            model_end = time.time()
            elapsed_model = model_end - model_start

            # update progress bar
            progress_bar(idx, total_models)
            print(f"  → Model '{selected_cfg['name']}' finished in {elapsed_model:.2f} seconds")
            

        # end total timer
        total_end = time.time()
        elapsed_total = total_end - total_start
        print(f"\nAll {total_models} models finished in {elapsed_total:.2f} seconds.")
    else:
        # backward compat: single model under "model" key
        selected_cfg = combined_config.get("model", {})
        if "name" not in selected_cfg:
            selected_cfg = dict(selected_cfg)
            selected_cfg["name"] = model_name

        runtime_cfg = inject_splits_and_preprocessor(selected_cfg, pre, splits)

        if selected_cfg.get('build_mode') == 'grid':
            start = time.time()
            manager.run_tuning(selected_cfg['type'], mode=mode, config=runtime_cfg, n_trials=8)
            end = time.time()
            print(f"Model '{selected_cfg['name']}' finished in {end-start:.2f} seconds.")

if __name__ == "__main__":
    main()