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
        self.is_built = False

    def build(self, model: Any) -> bool:
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
