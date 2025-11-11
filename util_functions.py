
def inject_splits_and_preprocessor(model_cfg: dict, pre, splits: dict) -> dict:
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

def inject_all_models(config: dict, pre, splits: dict) -> list[dict]:
    """
    Return a new list of model configs with splits and preprocessor info injected.
    """
    return [inject_splits_and_preprocessor(m, pre, splits) for m in config.get("models", [])]

def one_model_run(combined_config,
                  model_name,
                  mode,
                  manager,pre,splits):
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
            print(f"Model '{model_name}' not found amongst config models. Available:", [m.get("name") for m in models_cfg_list])
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

    results = manager.run_pipeline(selected_cfg['type'], mode=mode, config=runtime_cfg)
    print("Training results:", results)
    pred_res = manager.run_pipeline(selected_cfg['type'], mode="predict", config=runtime_cfg)
    print("Prediction results:", pred_res)

    val_res = manager.run_pipeline(selected_cfg['type'], mode="validate", config=runtime_cfg)
    print("Validation results:", val_res)


def gridsearch_run(combined_config,
                  mode,
                  manager,pre,splits):
    runtime_cfg = inject_all_models(combined_config, pre, splits)
    results = manager.run_gridsearch_pipeline(combined_config,runtime_cfg)
    print("Training results:", results)

