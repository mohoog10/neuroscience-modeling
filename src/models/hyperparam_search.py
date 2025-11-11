from sklearn.model_selection import GridSearchCV
import json
from copy import deepcopy

class HyperParamSearch:
    def __init__(self, config, manager_cls, X_train, y_train):
        self.config = config
        self.manager_cls = manager_cls
        self.X_train = X_train
        self.y_train = y_train
        self.results = {}

    def run(self):
        gs_cfg = self.config.get("gridsearch", {})
        cv = gs_cfg.get("cv", 3)
        scoring = gs_cfg.get("scoring", "accuracy")
        n_jobs = gs_cfg.get("n_jobs", -1)
        verbose = gs_cfg.get("verbose", 1)

        for model_cfg in self.config.get("models", []):
            grid = model_cfg.get("param_grid")
            if not grid:
                continue

            name = model_cfg["name"]
            print(f"GridSearchCV for {name}...")

            # Ask Manager to build the model object
            manager = self.manager_cls(deepcopy(self.config))
            estimator = manager.build_model(model_cfg)  # <-- your existing factory

            gs = GridSearchCV(
                estimator=estimator,
                param_grid=grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose
            )
            gs.fit(self.X_train, self.y_train)

            self.results[name] = {
                "best_score": gs.best_score_,
                "best_params": gs.best_params_,
                "cv_results": gs.cv_results_
            }

        return self.results
    

