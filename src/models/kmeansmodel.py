
from __future__ import annotations
from .model import Model
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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

    def __init__(self):
        # within __init__
        self._saved_split_labels = {"train": None, "val": None, "test": None}
        self._saved_split_predictions = {"train": None, "val": None, "test": None}  
        super().__init__()

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
        #centers = np.vstack([self.model.cluster_centers_[labels == i].mean(axis=0) for i in range(labels)])

        results = {
            "n_clusters": int(getattr(self.model, "n_clusters", np.unique(labels).size)),
            "inertia": float(getattr(self.model, "inertia_", np.nan)),
            "labels": labels,
        }
        results["cluster_centers"] = np.vstack([ X_train[labels == i].mean(axis=0) if np.any(labels == i
                        ) else self.model.cluster_centers_[i] for i in range(
                                        self.model.cluster_centers_.shape[0])])
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
    

    def plot_low_dim_using_config(self,
                              split: Optional[str] = "val",
                              method: str = "pca",
                              n_components: int = 2,
                              preprocessor=None,
                              y: Optional[np.ndarray] = None,
                              preds: Optional[np.ndarray] = None,
                              sample: Optional[int] = None,
                              figsize=(7, 5),
                              title: Optional[str] = None,
                              cmap="tab10",
                              save_path: Optional[str] = None,
                              random_state: int = 0):
        """
        Plot low-dimensional projection for data referenced in model config.

        Behavior and inputs:
        - split: 'train','val','test' or None. If provided, the method looks for
            X_{split} and optional y_{split} and preds_{split} inside self.config or
            as attributes on the model (self._X_train, self._y_train, self._preds_train, ...).
        - preprocessor: fitted transformer (Manager should pass the scaler/preprocessor).
                        If it has transform_new, it will be used; otherwise transform.
        - method: 'pca' (default), 'tsne', or 'umap' (if installed).
        - If sample is set, a random subset of rows is used for projection.
        - preds and y override values loaded from config/attributes.
        """
        # helper to load arrays from config or attributes
        def _load(name):
            # 1) explicit attribute on model (preferred)
            attr = getattr(self, name, None)
            if attr is not None:
                return attr
            # 2) from config dict if present
            cfg = getattr(self, "config", {}) or {}
            return cfg.get(name, None)

        if split is not None:
            if split not in ("train", "val", "test"):
                raise ValueError("split must be one of 'train','val','test' or None")
            X = _load(f"_X_{split}") or _load(f"X_{split}")
            y_loaded = _load(f"_y_{split}") or _load(f"y_{split}")
            preds_loaded = _load(f"_preds_{split}") or _load(f"preds_{split}")
        else:
            X = None
            y_loaded = None
            preds_loaded = None

        # allow explicit y/preds override
        y_use = np.asarray(y) if y is not None else (np.asarray(y_loaded) if y_loaded is not None else None)
        preds_use = np.asarray(preds) if preds is not None else (np.asarray(preds_loaded) if preds_loaded is not None else None)

        if X is None:
            raise RuntimeError("No data found for plotting. Ensure X_train/X_val/X_test are in model config or set as attributes.")

        # convert DataFrame to numpy, but keep DataFrame when preprocessor expects it
        is_df = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        X_in = X

            # apply preprocessor if provided
        if preprocessor is not None:
            # helper to obtain feature names if available
            def _feature_names_from(preproc, model):
                # common places we might have column order recorded
                candidates = []
                candidates.append(getattr(model, "_feature_columns", None))
                candidates.append(getattr(preproc, "feature_names_in_", None))
                candidates.append(getattr(preproc, "columns_", None))
                candidates.append(getattr(preproc, "_feature_names", None))
                for c in candidates:
                    if c is None:
                        continue
                    try:
                        if isinstance(c, (list, tuple)) and len(c) > 0:
                            return list(c)
                    except Exception:
                        continue
                return None

            # if X is ndarray but preprocessor expects DataFrame ops, convert to DataFrame
            need_df = False
            X_in = X
            if isinstance(X, (np.ndarray,)) and hasattr(preprocessor, "transform_new"):
                # assume transform_new expects DataFrame (your DataPreprocessor uses .drop)
                need_df = True

            if need_df:
                names = _feature_names_from(preprocessor, self)
                if names is None:
                    # best-effort fallback: create generic column names
                    names = [f"f{i}" for i in range(X.shape[1])]
                try:
                    X_df = pd.DataFrame(X, columns=names)
                    X_in = preprocessor.transform_new(X_df) if hasattr(preprocessor, "transform_new") else preprocessor.transform(X_df)
                except Exception as exc:
                    # fallback: try calling transform on raw array
                    try:
                        X_in = preprocessor.transform(X)
                    except Exception:
                        raise RuntimeError(f"Preprocessor failed to transform input for plotting: {exc}") from exc
            else:
                # preprocessor exists but either X already DataFrame or transform_new not available
                if hasattr(preprocessor, "transform_new"):
                    X_in = preprocessor.transform_new(X)
                elif hasattr(preprocessor, "transform"):
                    X_in = preprocessor.transform(X)
                else:
                    # no transform method; assume input is already numeric
                    X_in = X

            # compute projection
            if method == "pca":
                proj = PCA(n_components=n_components, random_state=random_state).fit_transform(X_in)
            elif method == "tsne":
                proj = TSNE(n_components=n_components, random_state=random_state, init="pca").fit_transform(X_in)
            elif method == "umap":
                try:
                    import umap
                    proj = umap.UMAP(n_components=n_components, random_state=random_state).fit_transform(X_in)
                except Exception as exc:
                    raise RuntimeError("UMAP not available; install umap-learn or choose 'pca'/'tsne'") from exc
            else:
                raise ValueError("method must be one of 'pca','tsne','umap'")

        # prepare plotting
        # single subplot if only one of y_use/preds_use exists, two subplots if both
        plots = 1 + (1 if (y_use is not None and preds_use is not None) else 0)
        fig, axes = plt.subplots(1, plots, figsize=(figsize[0] * plots, figsize[1]), squeeze=False)
        axes = axes[0]

        def _scatter(ax, labels, subtitle):
            labels = np.asarray(labels)
            unique = np.unique(labels)
            cmap_use = cmap if len(unique) <= 10 else "tab20"
            for lab in unique:
                mask = labels == lab
                ax.scatter(proj[mask, 0], proj[mask, 1], s=20, alpha=0.8, label=str(lab))
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.set_title(subtitle)
            ax.legend(markerscale=2, fontsize="small", bbox_to_anchor=(1.05, 1), loc="upper left")

        if y_use is not None and preds_use is not None:
            _scatter(axes[0], y_use, "True labels")
            _scatter(axes[1], preds_use, "Predicted clusters")
        else:
            if preds_use is not None:
                _scatter(axes[0], preds_use, "Predicted clusters")
            elif y_use is not None:
                _scatter(axes[0], y_use, "True labels")
            else:
                ax = axes[0]
                ax.scatter(proj[:, 0], proj[:, 1], s=20, alpha=0.8)
                ax.set_xlabel("Dim 1")
                ax.set_ylabel("Dim 2")
                ax.set_title("Data projection")

        if title:
            fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 0.85, 0.95])
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        plt.show()
        return fig