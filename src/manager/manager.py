"""
Manager Module
Manages interface and model registry coordination
"""
import joblib
import json
from typing import Optional, Dict, Any
from ..interface.interface import Interface
from ..registry.model_registry import ModelRegistry
from ..models.model import Model
from ..processing.datapreprocessor import DataPreprocessor
#from sklearn.model_selection import GridSearchCV
import json
from copy import deepcopy

class Manager:
    """
    Manager class that coordinates between interface and model registry
    """
    
    def __init__(self, interface: Interface, model_registry: ModelRegistry, 
                 preprocessor: DataPreprocessor):
        """
        Initialize Manager
        
        Args:
            interface: Interface implementation (e.g., CLI)
            model_registry: Model registry instance
        """
        self.interface = interface
        self.model_registry = model_registry
        self.preprocessor = preprocessor
        self.current_model: Optional[Model] = None
        
        # Configuration storage
        self.train_config: Dict[str, Any] = {}
        self.model_instance_name: Optional[str] = None
        self.model_instance_config: Dict[str, Any] = {}
        self.train_config: Dict[str, Any] = {}
        self.model_maintenance_name: Optional[str] = None
        self.model_maintenance_config: Dict[str, Any] = {}
        self.model_with_model_instance_name: Optional[str] = None
        self.model_with_model_instance_config: Dict[str, Any] = {}
        self.train_model_instance_name: Optional[str] = None
        self.train_model_instance_config: Dict[str, Any] = {}
        self.store_model_instance_name: Optional[str] = None
        self.store_model_instance_config: Dict[str, Any] = {}
        self.host_model_instance_name: Optional[str] = None
        
        self.new_model_name: Optional[str] = None
    
    def select_model(self, model_name: str, config: Optional[Dict] = None) -> bool:
        """
        Select and initialize a model
        
        Args:
            model_name: Name of the model to select
            config: Optional configuration dictionary
            
        Returns:
            bool: Success status
        """
        print(f"Selecting model: {model_name}")
        
        # reuse existing built instance if same name
        if getattr(self, "model_selected_name", None) == model_name and self.current_model is not None:
            if getattr(self.current_model, "is_built", False):
                print(f"Reusing already-built model instance for '{model_name}'")
                if config:
                    self.model_instance_config = config
                return True

        if not self.model_registry.model_exists(model_name):
            print(f"Error: Model '{model_name}' not found")
            return False
        
        self.current_model = self.model_registry.get_model(model_name)
        
        if config:
            self.model_instance_config = config
        
        print(f"Model '{model_name}' selected successfully")
        return True
    
    def build_model(self, config: Optional[Dict] = None,return_estimator=False) -> bool:
        """
        Build the currently selected model
        
        Args:
            config: Optional build configuration
            
        Returns:
            bool: Success status
        """
        if not self.current_model:
            print("Error: No model selected")
            return False
        
        # early skip if already built
        if getattr(self.current_model, "is_built", False):
            #print("Model already built; skipping build step")
            if config:
                self.model_instance_config = config
            return True


        build_config = config if config else self.model_instance_config

        return self.current_model.build(build_config,return_estimator)
    
    def train_model(self) -> Dict:
        """
        Train the current model
        
        Returns:
            dict: Training results
        """
        if not self.current_model:
            print("Error: No model selected")
            return {"error": "No model selected"}
        
        return self.current_model.train()
    
    def validate_model(self) -> Dict:
        """
        Validate the current model
        
        Returns:
            dict: Validation results
        """
        if not self.current_model:
            print("Error: No model selected")
            return {"error": "No model selected"}
        
        return self.current_model.validate()
    
    def test_model(self) -> Dict:
        """
        Test the current model
        
        Returns:
            dict: Test results
        """
        if not self.current_model:
            print("Error: No model selected")
            return {"error": "No model selected"}
        
        return self.current_model.test()
    
    def predict_with_model(self) -> Dict:
        """
        Make predictions with the current model
        
        Returns:
            dict: Prediction results
        """
        if not self.current_model:
            print("Error: No model selected")
            return {"error": "No model selected"}
        
        return self.current_model.predict()
    
    def run_pipeline(self, model_name: str = None, mode: str = 'train', config: Optional[Dict] = None) -> Dict:
        """
        Run a complete pipeline: either gridsearch across all models or a single model.
        """

        # === existing single-model path ===
        print(f"\n{'='*50}")
        print(f"Running Pipeline: {model_name} in {mode} mode")
        print(f"{'='*50}\n")

        if not self.select_model(model_name, config):
            return {"error": "Model selection failed"}
        if not self.build_model():
            return {"error": "Model build failed"}

        if mode == 'train':
            results = self.train_model()
        elif mode == 'validate':
            results = self.validate_model()
        elif mode == 'test':
            results = self.test_model()
        elif mode == 'predict':
            results = self.predict_with_model()
        else:
            results = {"error": f"Unknown mode: {mode}"}

        print(f"\n{'='*50}")
        print(f"Pipeline completed for {model_name}")
        print(f"{'='*50}\n")
        return results

    def run_gridsearch_pipeline(self, cfg: Dict,models_cfg: list[Dict]) -> Dict:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
        from sklearn.model_selection import GridSearchCV
        gs_cfg = cfg.get('gridsearch',{})
        cv = gs_cfg.get("cv", 3)
        scoring = gs_cfg.get("scoring", "accuracy")
        n_jobs = gs_cfg.get("n_jobs", -1)
        verbose = gs_cfg.get("verbose", 1)

        pre_grid = gs_cfg.get("preprocessing_grid", {})
        scaler_opts = pre_grid.get("scaler", [False])
        poly_opts = pre_grid.get("poly_degree", [False])

        results = {}
        for model_cfg in models_cfg:
            grid = model_cfg.get("param_grid")
            if not grid:
                continue

            name = model_cfg["type"]
            print(f"\n{'='*50}\nGridSearchCV for {name}\n{'='*50}\n")
            model_cfg["input_dim"] = model_cfg.get('X_train').shape[1]
            self.select_model(name, model_cfg)   # select this model
            self.build_model(return_estimator=True)  # adapt build_model to optionally return raw estimator
            estimator = self.current_model.model

            # Build pipeline
            pipe = Pipeline([
                ("scaler", "passthrough"),
                ("poly", "passthrough"),
                ("model", estimator)
            ])

            # Map string scalers to actual objects
            scaler_map = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler(),
                "robust": RobustScaler()
            }

            # Build combined param grid
            combined_grid = []

            for sc in scaler_opts:
                for deg in poly_opts:
                    base = {}
                    # scaler choice
                    base["scaler"] = [scaler_map[sc]] if sc else ["passthrough"]
                    # polynomial choice
                    base["poly"] = [PolynomialFeatures(degree=deg, include_bias=False)] if deg else ["passthrough"]

                    # model params (prefix with "model__")
                    model_grid = {f"model__{k}": v for k, v in grid.items()}

                    # merge
                    combined_grid.append({**base, **model_grid})
            #print(combined_grid)
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=combined_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,error_score='raise')
            gs.fit(model_cfg.get('X_train'), model_cfg.get('y_train'))

            results[name] = {
                "best_score": gs.best_score_,
                "best_params": gs.best_params_,
                "cv_results": gs.cv_results_
            }

        return results


    
    def get_available_models(self) -> list:
        """
        Get list of available models
        
        Returns:
            list: Available model names
        """
        return self.model_registry.list_models()

    def save_model_artifact(self,
                            save_dir: str,
                            model_name: str,
                            model_obj=None,
                            preprocessor=None,
                            metadata: dict = None,
                            overwrite: bool = False) -> str:
        """
        Save model_obj and preprocessor to save_dir. Returns metadata path.
        """
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        base = f"{model_name}_{ts}"

        model_path = p / f"{base}.joblib"
        prep_path = p / f"{base}_preproc.joblib"
        meta_path = p / f"{base}_meta.json"

        if not overwrite and (model_path.exists() or prep_path.exists() or meta_path.exists()):
            raise FileExistsError("Artifact already exists; set overwrite=True to replace.")

        # choose the model to save: passed model_obj or current_model instance with attribute .model
        to_save_model = model_obj or getattr(self.current_model, "model", None) or self.current_model
        if to_save_model is None:
            raise RuntimeError("No model instance available to save")

        # use joblib (works for sklearn and many simple Python objects)
        joblib.dump(to_save_model, model_path)

        if preprocessor is not None:
            joblib.dump(preprocessor, prep_path)
        else:
            # try to retrieve preprocessor attribute from Manager if stored earlier
            if hasattr(self, "preprocessor") and self.preprocessor is not None:
                joblib.dump(self.preprocessor, prep_path)
            else:
                prep_path = None

        meta = metadata or {}
        meta.update({
            "model_name": model_name,
            "model_path": str(model_path),
            "preprocessor_path": str(prep_path) if prep_path else None,
            "saved_at": ts
        })
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return str(meta_path)

    def load_model_artifact(self, meta_path: str):
        """
        Load model and preprocessor given metadata JSON path. Returns (model, preprocessor, meta).
        """
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        model_path = meta.get("model_path")
        preproc_path = meta.get("preprocessor_path")
        model = joblib.load(model_path)
        preproc = joblib.load(preproc_path) if preproc_path else None
        return model, preproc, meta

    def predict_with_saved(self, meta_path: str, X_raw):
        """
        Load artifacts and run preprocessing + predict. X_raw may be DataFrame or path.
        """
        model, preproc, meta = self.load_model_artifact(meta_path)

        # if preproc exists, use it; otherwise assume X_raw already preprocessed numeric
        if preproc is not None:
            # ensure preproc has a transform method or DataPreprocessor.transform_new
            if hasattr(preproc, "transform_new"):
                X = preproc.transform_new(X_raw)
            elif hasattr(preproc, "transform"):
                X = preproc.transform(X_raw)
            else:
                raise RuntimeError("Preprocessor loaded has no transform_transform_new method")
        else:
            X = X_raw

        # call model predict (adapt for model API)
        if hasattr(model, "predict"):
            preds = model.predict(X)
        elif hasattr(model, "labels_"):  # a fitted clusterer without predict method
            preds = getattr(model, "labels_", None)
        else:
            raise RuntimeError("Loaded model has no predict method")

        return {"predictions": preds, "meta": meta}
    
    def run_tuning(self, model_name: str = None, mode: str = 'train', config: Optional[Dict] = None) -> Dict:
        """
        Run a complete pipeline: either gridsearch across all models or a single model.
        """

        # === existing single-model path ===
        print(f"\n{'='*50}")
        print(f"Running Keras hyperparameter tuning")
        print(f"{'='*50}\n")

        if not self.select_model(model_name, config):
            return {"error": "Model selection failed"}
        if not self.build_model():
            return {"error": "Model build failed"}

        self.current_model.run_optuna_search()

        print(f"\n{'='*50}")
        print(f"Tuning complete for {model_name}: {config.get('name','')}")
        print(f"{'='*50}\n")
        #return results