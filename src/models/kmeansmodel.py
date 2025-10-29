
from __future__ import annotations
from .model import Model
from typing import Any, Dict, Optional
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError


class KMeansModel(Model):
    """
    KMeans wrapper implementing the Model interface.

    Configuration options (passed to build as dict or set on self.config):
    - n_clusters: int (default 4)
    - random_state: int | None (default 0)
    - scaler: bool (default True) whether to StandardScale inputs
    - max_iter: int (default 300)
    - tol: float (default 1e-4)
    - X_train, X_val, X_test: numpy arrays of shape (n_samples, n_features)
    - fit_predict_on: str in {"train","val","test"} for predict() default "val"
    """

    def build(self, model: Any) -> bool:
        """
        Accepts either:
        - a dict with config keys (recommended), or
        - an sklearn-like KMeans instance

        Returns True on successful build.
        """
        if isinstance(model, dict):
            # merge provided config
            self.config.update(model)
            n_clusters = int(self.config.get("n_clusters", 4))
            random_state = self.config.get("random_state", 0)
            max_iter = int(self.config.get("max_iter", 300))
            tol = float(self.config.get("tol", 1e-4))
            self.model = KMeans(n_clusters=n_clusters,
                                random_state=random_state,
                                max_iter=max_iter,
                                tol=tol)
            self.is_built = True
            return True

        # accept prebuilt estimator
        if isinstance(model, KMeans):
            self.model = model
            self.is_built = True
            return True

        raise ValueError("build expects a config dict or a sklearn.cluster.KMeans instance")

    def _get_data(self, key: str) -> Optional[np.ndarray]:
        arr = self.config.get(key, None)
        if arr is None:
            return None
        return np.asarray(arr)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if not self.config.get("scaler", True):
            return X
        scaler = self.config.get("_internal_scaler", None)
        if scaler is None:
            scaler = StandardScaler()
            self.config["_internal_scaler"] = scaler
            Xs = scaler.fit_transform(X)
        else:
            Xs = scaler.transform(X)
        return Xs

    def train(self) -> Dict:
        """
        Fit KMeans on X_train provided in config.
        Returns training results with labels and basic metrics.
        """
        if not self.is_built:
            raise RuntimeError("Model not built. Call build(...) first.")

        X_train = self._get_data("X_train")
        if X_train is None:
            raise ValueError("X_train must be provided in config before train()")

        X_train = self._scale(X_train)
        self.model.fit(X_train)

        labels = self.model.labels_
        results = {
            "n_clusters": int(getattr(self.model, "n_clusters", np.unique(labels).size)),
            "inertia": float(getattr(self.model, "inertia_", np.nan)),
            "labels": labels,
        }

        # Silhouette requires at least 2 clusters and less than n_samples clusters
        try:
            if len(np.unique(labels)) > 1 and len(np.unique(labels)) < X_train.shape[0]:
                results["silhouette_score"] = float(silhouette_score(X_train, labels))
                results["calinski_harabasz"] = float(calinski_harabasz_score(X_train, labels))
                results["davies_bouldin"] = float(davies_bouldin_score(X_train, labels))
        except Exception:
            pass

        self.is_validated = False
        self.is_predicted = False
        self.is_tested = False
        return results

    def validate(self) -> Dict:
        """
        Validate on X_val and return clustering labels and metrics.
        If no X_val present, raises ValueError.
        """
        if not self.is_built:
            raise RuntimeError("Model not built. Call build(...) first.")

        X_val = self._get_data("X_val")
        if X_val is None:
            raise ValueError("X_val must be provided in config before validate()")

        X_val = self._scale(X_val)
        try:
            labels = self.model.predict(X_val)
        except NotFittedError:
            raise RuntimeError("Model must be trained (train()) before validate()")

        results = {
            "n_samples": int(X_val.shape[0]),
            "labels": labels,
        }
        try:
            if len(np.unique(labels)) > 1 and len(np.unique(labels)) < X_val.shape[0]:
                results["silhouette_score"] = float(silhouette_score(X_val, labels))
                results["calinski_harabasz"] = float(calinski_harabasz_score(X_val, labels))
                results["davies_bouldin"] = float(davies_bouldin_score(X_val, labels))
        except Exception:
            pass

        self.is_validated = True
        return results

    def predict(self) -> Dict:
        """
        Predict cluster assignments on dataset specified by 'fit_predict_on' config key.
        Allowed values: "train", "val", "test". Default: "val".
        Returns labels and counts per cluster.
        """
        if not self.is_built:
            raise RuntimeError("Model not built. Call build(...) first.")

        target = self.config.get("fit_predict_on", "val")
        key_map = {"train": "X_train", "val": "X_val", "test": "X_test"}
        if target not in key_map:
            raise ValueError("fit_predict_on must be one of 'train', 'val', 'test'")

        X = self._get_data(key_map[target])
        if X is None:
            raise ValueError(f"{key_map[target]} must be provided in config before predict()")

        X = self._scale(X)
        try:
            labels = self.model.predict(X)
        except NotFittedError:
            raise RuntimeError("Model must be trained (train()) before predict()")

        unique, counts = np.unique(labels, return_counts=True)
        counts_dict = {int(k): int(v) for k, v in zip(unique, counts)}
        self.is_predicted = True
        return {"labels": labels, "counts": counts_dict, "n_samples": int(X.shape[0])}

    def test(self) -> Dict:
        """
        Test uses X_test in config. Returns the same structure as validate/predict
        plus optionally external test metrics if 'y_test' with ground truth cluster ids is provided.
        """
        if not self.is_built:
            raise RuntimeError("Model not built. Call build(...) first.")

        X_test = self._get_data("X_test")
        if X_test is None:
            raise ValueError("X_test must be provided in config before test()")

        X_test = self._scale(X_test)
        try:
            labels = self.model.predict(X_test)
        except NotFittedError:
            raise RuntimeError("Model must be trained (train()) before test()")

        results = {
            "n_samples": int(X_test.shape[0]),
            "labels": labels,
        }

        # optional external labels to compute supervised metrics (if available)
        y_test = self.config.get("y_test", None)
        if y_test is not None:
            y_test = np.asarray(y_test)
            if y_test.shape[0] == labels.shape[0]:
                # use adjusted rand if available without import overhead
                try:
                    from sklearn.metrics import adjusted_rand_score
                    results["adjusted_rand_score"] = float(adjusted_rand_score(y_test, labels))
                except Exception:
                    pass

        try:
            if len(np.unique(labels)) > 1 and len(np.unique(labels)) < X_test.shape[0]:
                results["silhouette_score"] = float(silhouette_score(X_test, labels))
                results["calinski_harabasz"] = float(calinski_harabasz_score(X_test, labels))
                results["davies_bouldin"] = float(davies_bouldin_score(X_test, labels))
        except Exception:
            pass

        self.is_tested = True
        return results