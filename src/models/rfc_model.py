from typing import Any, Dict, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
from sklearn.exceptions import NotFittedError
from ..models.model import Model

class RandomForestClassifierModel(Model):
    def __init__(self):
        super().__init__()
        self._saved_split_labels = {"train": None, "val": None, "test": None}
        self._saved_split_predictions = {"train": None, "val": None, "test": None}
        self._label_to_index = None
        self._index_to_label = None
        self.is_built = False

    def build(self, model: Any, return_estimator:bool=False) -> bool:
        if return_estimator:
            self.model = RandomForestClassifier()
            self.is_built = True
            return True
        
        if isinstance(model, dict):
            self.config.update(model)
            n_estimators = int(self.config.get("n_estimators", 100))
            max_depth = self.config.get("max_depth", None)
            random_state = int(self.config.get("random_state", 0))
            self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
            # RandomForest is tree-based; scaling optional but not required
            self.is_built = True
            return True
        if isinstance(model, RandomForestClassifier):
            self.model = model
            self.is_built = True
            return True
        raise ValueError("build expects config dict or RandomForestClassifier instance")

    def _get_data(self, key: str) -> Optional[np.ndarray]:
        arr = self.config.get(key, None)
        return None if arr is None else np.asarray(arr)

    def _maybe_scale(self, X: np.ndarray) -> np.ndarray:
        if not self.config.get("scaler", False):
            return X
        scaler = self.config.get("_internal_scaler", None)
        if scaler is None:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            self.config["_internal_scaler"] = scaler
            return scaler.fit_transform(X)
        return scaler.transform(X)
    
    def _prepare_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Encode labels to integer indices and keep mapping for inverse transform.
        If y is already numeric, keep as-is.
        """
        y = np.asarray(y)
        # If numeric and integer-like, return as ints
        if np.issubdtype(y.dtype, np.number):
            return y.astype(int)

        # otherwise create mapping
        unique = np.unique(y)
        self._index_to_label = [str(u) for u in unique]
        self._label_to_index = {str(lbl): idx for idx, lbl in enumerate(self._index_to_label)}
        y_idx = np.array([self._label_to_index[str(v)] for v in y], dtype=int)
        return y_idx

    def train(self) -> Dict:
        if not self.is_built:
            raise RuntimeError("Model not built. Call build(...) first.")
        X = self._get_data("X_train"); y = self.config.get("y_train", None)
        if X is None or y is None:
            raise ValueError("X_train and y_train must be provided in config before train()")
        y = np.asarray(y)
        Xs = self._maybe_scale(X)
        self.model.fit(Xs, y)
        preds = self.model.predict(Xs)
        self._saved_split_labels["train"] = y
        self._saved_split_predictions["train"] = preds
        res = {"n_samples": int(Xs.shape[0]), "train_accuracy": float(accuracy_score(y, preds))}
        try:
            res["train_f1"] = float(f1_score(y, preds, average="weighted"))
        except Exception:
            pass
        return res

    def validate(self) -> Dict:
        if not self.is_built:
            raise RuntimeError("Model not built. Call build(...) first.")
        X = self._get_data("X_val"); y = self.config.get("y_val", None)
        if X is None or y is None:
            raise ValueError("X_val and y_val must be provided in config before validate()")
        y = np.asarray(y)
        Xs = self._maybe_scale(X)
        try:
            preds = self.model.predict(Xs)
        except NotFittedError:
            raise RuntimeError("Model must be trained (train()) before validate()")
        self._saved_split_labels["val"] = y
        self._saved_split_predictions["val"] = preds
        res = {"n_samples": int(Xs.shape[0]), "val_accuracy": float(accuracy_score(y, preds))}
        try:
            res["val_f1"] = float(f1_score(y, preds, average="weighted"))
            print("*"*40)
            print("\nConfusion matrix: \n")
            print(confusion_matrix(y_true=y,y_pred=preds))
        except Exception:
            pass
        return res

    def predict(self) -> Dict:
        if not self.is_built:
            raise RuntimeError("Model not built. Call build(...) first.")
        target = self.config.get("fit_predict_on", "val")
        key_map = {"train": "X_train", "val": "X_val", "test": "X_test"}
        if target not in key_map:
            raise ValueError("fit_predict_on must be one of 'train','val','test'")
        X = self._get_data(key_map[target])
        if X is None:
            raise ValueError(f"{key_map[target]} must be provided in config before predict()")
        Xs = self._maybe_scale(X)
        try:
            preds = self.model.predict(Xs)
        except NotFittedError:
            raise RuntimeError("Model must be trained (train()) before predict()")
        self._saved_split_predictions[target] = preds
        #print(target)
        unique, counts = np.unique(preds, return_counts=True)
        #print(unique)
        #print(counts)
        return {"counts": {str(k): int(v) for k, v in zip(unique, counts)}, "n_samples": int(Xs.shape[0])}

    def test(self) -> Dict:
        if not self.is_built:
            raise RuntimeError("Model not built. Call build(...) first.")
        X = self._get_data("X_test"); y = self.config.get("y_test", None)
        if X is None:
            raise ValueError("X_test must be provided in config before test()")
        Xs = self._maybe_scale(X)
        try:
            preds = self.model.predict(Xs)
        except NotFittedError:
            raise RuntimeError("Model must be trained (train()) before test()")
        res = {"n_samples": int(Xs.shape[0]), "labels": preds}
        if y is not None:
            y = np.asarray(y)
            if y.shape[0] == preds.shape[0]:
                res["test_accuracy"] = float(accuracy_score(y, preds))
                try:
                    res["test_f1"] = float(f1_score(y, preds, average="weighted"))
                except Exception:
                    pass
        self._saved_split_predictions["test"] = preds
        self._saved_split_labels["test"] = np.asarray(y) if y is not None else None
        return res

    def run_optuna_search(self, n_trials=10):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        import optuna
        import numpy as np
        import pandas as pd
        import os
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        X = self._get_data("X_train")
        y = self.config.get("y_train", None)
        if X is None or y is None:
            raise ValueError("X_train and y_train must be provided")

        # Flatten to 2D
        Xs = np.asarray(X, dtype="float32").reshape(X.shape[0], -1)
        y_idx = self._prepare_labels(y)

        # Validation/test
        X_val = self._get_data("X_val"); y_val = self.config.get("y_val", None)
        X_test = self._get_data("X_test"); y_test = self.config.get("y_test", None)
        validation_data = None; test_data = None
        if X_val is not None and y_val is not None:
            Xv = np.asarray(X_val, dtype="float32").reshape(X_val.shape[0], -1)
            yv = self._prepare_labels(y_val)
            Xt = np.asarray(X_test, dtype="float32").reshape(X_test.shape[0], -1)
            yt = self._prepare_labels(y_test)
            validation_data = (Xv, yv)
            test_data = (Xt, yt)

        results = []
        model_type = "RandomForest"
        trial_dir = os.path.join("results", "weights", model_type)
        os.makedirs(trial_dir, exist_ok=True)

        def objective(trial):
            hp = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 500, step=5),
                "max_depth": trial.suggest_int("max_depth", 5, 50, step=1),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }

            clf = RandomForestClassifier(**hp, random_state=42, n_jobs=-1)
            clf.fit(Xs, y_idx)

            if validation_data:
                val_acc = accuracy_score(validation_data[1], clf.predict(validation_data[0]))
            else:
                val_acc = accuracy_score(y_idx, clf.predict(Xs))

            train_acc = accuracy_score(y_idx, clf.predict(Xs))

            trial.set_user_attr("train_acc", float(train_acc))
            trial.set_user_attr("val_acc", float(val_acc))
            trial.set_user_attr("params", hp)

            results.append({
                "trial": trial.number,
                **hp,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            })

            return val_acc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best = study.best_trial
        best_params = best.user_attrs["params"]
        best_model = RandomForestClassifier(**best_params, random_state=self.config.get("random_state",0), n_jobs=-1)
        best_model.fit(Xs, y_idx)

        if test_data:
            self._save_classification_report_and_confusion(best_model, test_data[0], test_data[1], model_type)

        # Save results
        trials_csv = os.path.join("results", f"{model_type}_trials_results.csv")
        results_csv = os.path.join("results", "best_results.csv")

        df_trials = pd.DataFrame(results)
        if os.path.exists(trials_csv):
            old = pd.read_csv(trials_csv)
            df_trials = pd.concat([old, df_trials], ignore_index=True)
        df_trials = df_trials.sort_values(by=["train_accuracy", "val_accuracy"], ascending=False)
        df_trials.to_csv(trials_csv, index=False)

        best = study.best_trial
        df_best = pd.DataFrame([{
            "model_type": model_type,
            "best_trial_number": best.number,
            "best_train_acc": best.user_attrs.get("train_acc"),
            "best_val_acc": best.user_attrs.get("val_acc"),
            "params": best.user_attrs.get("params"),
        }])
        if os.path.exists(results_csv):
            old = pd.read_csv(results_csv)
            df_best = pd.concat([old, df_best], ignore_index=True)
        df_best.to_csv(results_csv, index=False)

    def _suggest_hparams(self, trial, model_type):
        """
        Suggest hyperparameters depending on model type.
        """

        return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 50, step=5),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),}
    

    def _save_classification_report_and_confusion(self, model, X_val, y_val, model_type):
        """
        Evaluate the best model on validation data, save classification report and confusion matrix heatmap.
        """
        import os
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import classification_report, confusion_matrix
        # Predict
        y_pred_probs = model.predict_proba(X_val)
        y_pred = y_pred_probs.argmax(axis=1)
        y_true = y_val.argmax(axis=1) if y_val.ndim > 1 else y_val
        print(self._index_to_label) 
        if self._index_to_label is None:
            label_names = [str(i) for i in np.unique(y_true)]
        else:
            label_names = self._index_to_label

        # Classification report
        report = classification_report(y_true, y_pred, target_names=label_names, digits=4)

        # Save report to text file
        outdir = os.path.join("results", "reports", model_type)
        os.makedirs(outdir, exist_ok=True)
        report_path = os.path.join(outdir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{model_type} Confusion Matrix")
        plt.tight_layout()

        cm_path = os.path.join(outdir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()

        print(f"Saved classification report to {report_path}")
        print(f"Saved confusion matrix heatmap to {cm_path}")
