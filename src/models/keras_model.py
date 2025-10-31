import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from typing import Any, Dict, Optional, List
import numpy as np

# tensorflow import
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# assume Model base class available in your codebase
from .model import Model   # adjust import to your project's Model base


class KerasClassifierModel(Model):
    """
    Keras-based classifier that conforms to your Model interface.

    Config options (examples):
    - name: "KerasClassifier"
    - scaler: bool (default True)
    - build_mode: "default" or "config"
    - epochs: int (default 20)
    - batch_size: int (default 32)
    - learning_rate: float (default 1e-3)
    - layers: optional list of layer specs when build_mode == "config". Example:
        [
          {"type": "Dense", "units": 128, "activation": "relu"},
          {"type": "Dropout", "rate": 0.3},
          {"type": "Dense", "units": 64, "activation": "relu"}
        ]
    - output_activation: "softmax" or "sigmoid" (default chosen based on n_classes)
    - loss: optional, inferred by task if not provided
    - metrics: optional list of metrics names
    - X_train/X_val/X_test and y_train/y_val/y_test expected in config at runtime
    """

    def __init__(self):
        super().__init__()
        self.is_built = False
        self._saved_split_labels = {"train": None, "val": None, "test": None}
        self._saved_split_predictions = {"train": None, "val": None, "test": None}
        self._label_to_index = None
        self._index_to_label = None
        self.model = None
        self.history = None

    def build(self, model: Any) -> bool:
        """
        Build from a config dict or accept a prebuilt tf.keras.Model.
        """
        if isinstance(model, dict):
            self.config.update(model)
            # default hyperparams
            self.config.setdefault("scaler", True)
            self.config.setdefault("epochs", 20)
            self.config.setdefault("batch_size", 32)
            self.config.setdefault("learning_rate", 1e-3)
            self.config.setdefault("build_mode", "default")  # or "config"
            # don't actually build the Keras graph until we know input shape (train time)
            self.is_built = True
            return True

        if isinstance(model, tf.keras.Model):
            self.model = model
            self.is_built = True
            return True

        raise ValueError("build expects a config dict or a tf.keras.Model instance")

    def _get_data(self, key: str) -> Optional[np.ndarray]:
        arr = self.config.get(key, None)
        return None if arr is None else np.asarray(arr)

    def _ensure_scaler(self, X: np.ndarray) -> np.ndarray:
        if not self.config.get("scaler", True):
            return X
        scaler = self.config.get("_internal_scaler", None)
        if scaler is None:
            scaler = StandardScaler()
            self.config["_internal_scaler"] = scaler
            return scaler.fit_transform(X)
        return scaler.transform(X)
    
    def _build_callbacks(self) -> list:
        """
        Build a list of Keras callbacks based on self.config.get("callbacks", {}).
        Accepts either a dict of named callback-configs or a list of callback-config objects.
        Each callback config is a dict with "type" and params.
        Example forms:
        "callbacks": {
            "reduce_lr": {"type":"ReduceLROnPlateau", "monitor":"val_loss", "factor":0.5, ...},
            "earlystop": {"type":"EarlyStopping", "monitor":"val_loss", "patience":200, ...}
        }
        or
        "callbacks": [
            {"type":"ReduceLROnPlateau", ...},
            {"type":"EarlyStopping", ...}
        ]
        """
        cfg = self.config.get("callbacks")
        if not cfg:
            return []

        # normalize to list
        if isinstance(cfg, dict):
            items = list(cfg.values())
        elif isinstance(cfg, list):
            items = cfg
        else:
            return []

        cb_list = []
        for item in items:
            typ = item.get("type")
            params = {k: v for k, v in item.items() if k != "type"}
            if typ == "ReduceLROnPlateau":
                cb = keras.callbacks.ReduceLROnPlateau(**params)
            elif typ == "EarlyStopping":
                cb = keras.callbacks.EarlyStopping(**params)
            elif typ == "ModelCheckpoint":
                cb = keras.callbacks.ModelCheckpoint(**params)
            elif typ == "CSVLogger":
                cb = keras.callbacks.CSVLogger(**params)
            else:
                # unknown type: skip or log; for safety, skip
                continue
            cb_list.append(cb)
        return cb_list

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

    def _build_model_from_config(self, input_dim: int, n_classes: int) -> tf.keras.Model:
        cfg_layers: List[Dict] = self.config.get("layers", None)
        lr = float(self.config.get("learning_rate", 1e-3))
        output_activation = self.config.get("output_activation", None)
        if output_activation is None:
            output_activation = "softmax" if n_classes > 2 else "sigmoid"

        inputs = layers.Input(shape=(input_dim,1))
        x = inputs
        if self.config.get("build_mode", "default") == "config" and cfg_layers:
            # construct from layers list
            for spec in cfg_layers:
                typ = spec.get("type", "Dense")
                #print(typ)
                if typ == "Dense":
                    units = int(spec.get("units"))
                    act = spec.get("activation", "relu")
                    x = layers.Dense(units, activation=act)(x)
                elif typ == "Dropout":
                    rate = float(spec.get("rate", 0.5))
                    x = layers.Dropout(rate)(x)
                elif typ == "BatchNormalization":
                    x = layers.BatchNormalization()(x)
                elif typ == "Activation":
                    x = layers.Activation(spec.get("activation", "relu"))(x)
                elif typ == "Conv1D":
                    x = layers.Conv1D(filters=int(spec.get("filters", 32)),
                                  kernel_size=int(spec.get("kernel_size", 3)),
                                  activation=spec.get("activation", "relu"),
                                  padding=spec.get("padding", "valid"))(x)
                elif typ == "Conv2D":
                    k = spec.get("kernel_size", [3, 3])
                    x = layers.Conv2D(filters=int(spec.get("filters", 32)),
                                  kernel_size=tuple(k),
                                  activation=spec.get("activation", "relu"),
                                  padding=spec.get("padding", "valid"))(x)
                elif typ == "MaxPooling1D":
                    x = layers.MaxPooling1D(pool_size=int(spec.get("pool_size", 2)))(x)
                elif typ == "MaxPooling2D":
                    x = layers.MaxPooling2D(pool_size=tuple(spec.get("pool_size", [2, 2])))
                elif typ == "Reshape":
                    target_shape = tuple(spec.get("target_shape"))
                    x = layers.Reshape(target_shape)(x)
                elif typ == "Flatten":
                    x = layers.Flatten()(x)
                else:
                    raise ValueError(f"Unsupported layer type in config: {typ}")
        else:
            # default small MLP
            x = layers.Dense(128, activation="relu")(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(64, activation="relu")(x)

        # output
        if n_classes == 2:
            out_units = 1
            out_act = "sigmoid" if output_activation is None else output_activation
            outputs = layers.Dense(out_units, activation=out_act)(x)
            loss = self.config.get("loss") or "binary_crossentropy"
            compile_metrics = self.config.get("metrics", ["accuracy"])
            model = models.Model(inputs=inputs, outputs=outputs)
            optimizer = optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss=loss, metrics=compile_metrics)
            return model
        else:
            out_units = n_classes
            out_act = "softmax" if output_activation is None else output_activation
            outputs = layers.Dense(out_units, activation=out_act)(x)
            loss = self.config.get("loss") or "sparse_categorical_crossentropy"
            compile_metrics = self.config.get("metrics", ["accuracy"])
            model = models.Model(inputs=inputs, outputs=outputs)
            optimizer = optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss=loss, metrics=compile_metrics)
            return model

    def train(self) -> Dict:
        if not self.is_built:
            raise RuntimeError("Model not built. Call build(...) first.")

        X = self._get_data("X_train")
        y = self.config.get("y_train", None)
        if X is None or y is None:
            raise ValueError("X_train and y_train must be provided in config before train()")

        # scaling
        Xs = self._ensure_scaler(X)

        # label encoding
        y_idx = self._prepare_labels(y)
        n_classes = len(np.unique(y_idx))

        # build keras model if not already provided
        if self.model is None:
            input_dim = int(Xs.shape[1])
            self.model = self._build_model_from_config(input_dim=input_dim, n_classes=n_classes)

        # fit
        epochs = int(self.config.get("epochs", 20))
        batch_size = int(self.config.get("batch_size", 32))
        validation_data = None
        X_val = self._get_data("X_val"); y_val = self.config.get("y_val", None)
        if X_val is not None and y_val is not None:
            Xv = self._ensure_scaler(X_val)
            yv = self._prepare_labels(y_val) if (self._label_to_index is None) else np.array([self._label_to_index.get(str(v), -1) for v in y_val])
            validation_data = (Xv, yv)

        # fit model (suppress verbose default to 1)
        cbs = self._build_callbacks()

        use_cbs = bool(self.config.get("callbacks", False)) and bool(cbs)

        self.history = self.model.fit(Xs, y_idx,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=validation_data,
                              verbose=1,
                              callbacks=cbs if use_cbs else None)

        # saved predictions on train
        if self.model.output_shape[-1] == 1:
            raw_preds = (self.model.predict(Xs).ravel() > 0.5).astype(int)
        else:
            raw_preds = np.argmax(self.model.predict(Xs), axis=1)

        # map back to original label dtype if mapping exists
        if self._index_to_label is not None:
            preds_labels = np.array([self._index_to_label[int(i)] for i in raw_preds])
        else:
            preds_labels = raw_preds

        self._saved_split_labels["train"] = np.asarray(y)
        self._saved_split_predictions["train"] = preds_labels

        res = {"n_samples": int(Xs.shape[0])}
        try:
            res["train_accuracy"] = float(accuracy_score(np.asarray(y), preds_labels))
            res["train_f1"] = float(f1_score(np.asarray(y), preds_labels, average="weighted"))
        except Exception:
            pass

        return res

    def validate(self) -> Dict:
        if not self.is_built:
            raise RuntimeError("Model not built. Call build(...) first.")
        X_val = self._get_data("X_val")
        y_val = self.config.get("y_val", None)
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val must be provided in config before validate()")
        Xv = self._ensure_scaler(X_val)
        # ensure label mapping exists
        if self._label_to_index is not None:
            yv = np.array([self._label_to_index.get(str(v), -1) for v in y_val])
        else:
            yv = self._prepare_labels(y_val)

        try:
            if self.model.output_shape[-1] == 1:
                raw_preds = (self.model.predict(Xv).ravel() > 0.5).astype(int)
            else:
                raw_preds = np.argmax(self.model.predict(Xv), axis=1)
        except Exception as exc:
            raise RuntimeError("Model must be trained (train()) before validate()") from exc

        if self._index_to_label is not None:
            preds_labels = np.array([self._index_to_label[int(i)] for i in raw_preds])
        else:
            preds_labels = raw_preds

        self._saved_split_labels["val"] = np.asarray(y_val)
        self._saved_split_predictions["val"] = preds_labels

        res = {"n_samples": int(Xv.shape[0])}
        try:
            res["val_accuracy"] = float(accuracy_score(np.asarray(y_val), preds_labels))
            res["val_f1"] = float(f1_score(np.asarray(y_val), preds_labels, average="weighted"))
            print("*"*40)
            print("\nConfusion matrix: \n")
            print(confusion_matrix(y_true=y_val,y_pred=preds_labels))
        except Exception:
            pass
        return res

    def predict(self) -> Dict:
        if not self.is_built or self.model is None:
            raise RuntimeError("Model not built and trained. Call build() then train() first.")
        target = self.config.get("fit_predict_on", "val")
        key_map = {"train": "X_train", "val": "X_val", "test": "X_test"}
        if target not in key_map:
            raise ValueError("fit_predict_on must be one of 'train','val','test'")
        X = self._get_data(key_map[target])
        if X is None:
            raise ValueError(f"{key_map[target]} must be provided in config before predict()")
        Xs = self._ensure_scaler(X)

        # get raw predictions
        if self.model.output_shape[-1] == 1:
            raw = (self.model.predict(Xs).ravel() > 0.5).astype(int)
        else:
            raw = np.argmax(self.model.predict(Xs), axis=1)

        if self._index_to_label is not None:
            preds_labels = np.array([self._index_to_label[int(i)] for i in raw])
        else:
            preds_labels = raw

        unique, counts = np.unique(preds_labels, return_counts=True)
        counts_by_label = {str(k): int(v) for k, v in zip(unique, counts)}

        self._saved_split_predictions[target] = preds_labels
        return {"labels": preds_labels, "counts": counts_by_label, "n_samples": int(Xs.shape[0])}

    def test(self) -> Dict:
        if not self.is_built or self.model is None:
            raise RuntimeError("Model not built and trained. Call build() then train() first.")
        X_test = self._get_data("X_test")
        if X_test is None:
            raise ValueError("X_test must be provided in config before test()")
        Xs = self._ensure_scaler(X_test)

        if self.model.output_shape[-1] == 1:
            raw = (self.model.predict(Xs).ravel() > 0.5).astype(int)
        else:
            raw = np.argmax(self.model.predict(Xs), axis=1)

        if self._index_to_label is not None:
            preds_labels = np.array([self._index_to_label[int(i)] for i in raw])
        else:
            preds_labels = raw

        res = {"n_samples": int(Xs.shape[0]), "labels": preds_labels}

        y_test = self.config.get("y_test", None)
        if y_test is not None:
            y_test = np.asarray(y_test)
            if y_test.shape[0] == preds_labels.shape[0]:
                try:
                    res["test_accuracy"] = float(accuracy_score(y_test, preds_labels))
                    res["test_f1"] = float(f1_score(y_test, preds_labels, average="weighted"))
                except Exception:
                    pass

        self._saved_split_predictions["test"] = preds_labels
        self._saved_split_labels["test"] = np.asarray(y_test) if y_test is not None else None
        return res
