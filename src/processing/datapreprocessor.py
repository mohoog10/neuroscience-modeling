from typing import Optional, Dict, Any, Tuple, Union, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessor:
    """
    Load CSV or accept DataFrame, optionally one-hot encode and scale,
    then split into train/val/test. Returns X/y as numpy arrays or pandas objects.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default = dict(
            filepath=None,
            df=None,
            target_column=None,
            drop_columns=None,
            categorical_columns=None,
            dummy=True,
            scaler="standard",   # "standard", "minmax", or None
            test_size=0.2,
            val_size=0.1,        # proportion of the remaining after test split
            random_state=0,
            return_type="numpy"  # "numpy" or "pandas"
        )
        self.config = {**default, **(config or {})}
        self._scaler = None
        self._feature_columns: Optional[List[str]] = None
        self._numeric_columns: Optional[List[str]] = None
        self._categorical_columns: Optional[List[str]] = None

    def load(self) -> pd.DataFrame:
        if self.config["df"] is not None:
            df = self.config["df"].copy()
        elif self.config["filepath"]:
            df = pd.read_csv(self.config["filepath"])
        else:
            raise ValueError("Either 'df' or 'filepath' must be provided in config")
        return df

    def _select_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        drop_cols = self.config.get("drop_columns") or []
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        target = None
        if self.config["target_column"] is not None:
            if self.config["target_column"] not in df.columns:
                raise KeyError(f"target_column '{self.config['target_column']}' not in dataframe")
            target = df[self.config["target_column"]].copy()
            df = df.drop(columns=[self.config["target_column"]])

        return df, target

    def _ensure_list_or_none(self, v):
        """Accept None, a single string, or an iterable; return list or None."""
        if v is None:
            return None
        # if user passed a single string, convert to single-item list
        if isinstance(v, str):
            return [v]
        try:
            # if it's already iterable (list/tuple/set), make a list
            return list(v)
        except TypeError:
            # fallback: wrap in list
            return [v]

    def _infer_categoricals(self, df: pd.DataFrame) -> List[str]:
        cats = self._ensure_list_or_none(self.config.get("categorical_columns"))
        if cats is not None:
            return [c for c in cats if c in df.columns]
        # infer object or category dtypes
        return [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c])]

    def _fit_scaler(self, X: pd.DataFrame):
        s = self.config.get("scaler")
        if s == "standard":
            self._scaler = StandardScaler()
        elif s == "minmax":
            self._scaler = MinMaxScaler()
        else:
            self._scaler = None

        if self._scaler is not None:
            numeric = X.select_dtypes(include=[np.number])
            if numeric.shape[1] > 0:
                self._numeric_columns = numeric.columns.tolist()
                self._scaler.fit(numeric.values)
            else:
                self._numeric_columns = []

    def _apply_scaler(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None or not self._numeric_columns:
            return X
        arr = X[self._numeric_columns].values
        scaled = self._scaler.transform(arr)
        X_loc = X.copy()
        X_loc.loc[:, self._numeric_columns] = scaled
        return X_loc

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # drop/target separation handled earlier
        df_proc = df.copy()
        cats = self._infer_categoricals(df_proc)
        self._categorical_columns = cats
 
        if self.config.get("dummy", True) and cats:
            df_proc = pd.get_dummies(df_proc, columns=cats, drop_first=False)
        # record final feature columns
        self._feature_columns = df_proc.columns.tolist()

        # fit scaler on full df (you may choose fit only on train later; this fits on full data by default)
        self._fit_scaler(df_proc)
        df_scaled = self._apply_scaler(df_proc)
        return df_scaled

    def split(self, df: Optional[pd.DataFrame] = None):
        """
        Returns: X_train, y_train, X_val, y_val, X_test, y_test
        """

        df = df if df is not None else self.load()
        X_df, y = self._select_columns(df)
        X_df = self.prepare_features(X_df)
        test_size = float(self.config.get("test_size", 0.2))
        val_size = float(self.config.get("val_size", 0.1))
        rs = self.config.get("random_state", None)

        # first split off test
        if test_size and test_size > 0:
            if y is None:
                X_temp, X_test = train_test_split(
                    X_df, test_size=test_size, random_state=rs, shuffle=True
                )
                y_temp, y_test = None, None
            else:
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X_df, y, test_size=test_size, random_state=rs, shuffle=True
                )
        else:
            X_temp, y_temp = X_df, y
            X_test, y_test = pd.DataFrame(columns=X_df.columns), None

        # then split temp into train and val (val_size is fraction of the original after removing test)
        if val_size and val_size > 0:
            rel_val = val_size / (1.0 - test_size) if test_size < 1.0 else 0.0
            if rel_val <= 0:
                X_train, X_val, y_train, y_val = X_temp, pd.DataFrame(columns=X_df.columns), y_temp, None
            else:
                if y_temp is None:
                    X_train, X_val = train_test_split(
                        X_temp, test_size=rel_val, random_state=rs, shuffle=True
                    )
                    y_train, y_val = None, None
                else:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp, test_size=rel_val, random_state=rs, shuffle=True
                    )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = pd.DataFrame(columns=X_df.columns), None

        # convert return types (numpy or pandas)
        if self.config.get("return_type") == "numpy":
            def maybe_to_numpy(obj):
                if obj is None:
                    return None
                if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
                    return obj.values
                return obj
            return (
                maybe_to_numpy(X_train), maybe_to_numpy(y_train),
                maybe_to_numpy(X_val), maybe_to_numpy(y_val),
                maybe_to_numpy(X_test), maybe_to_numpy(y_test)
            )
        else:
            return X_train, y_train, X_val, y_val, X_test, y_test


    def transform_new(self, df_new: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        """
        Apply stored preprocessing (dummies + scaler) to new data.
        It will align columns to the fitted feature set; missing columns will be filled with zeros.
        """
        if self._feature_columns is None:
            raise RuntimeError("Preprocessor not fitted yet. Call split() or prepare_features() first.")

        df = df_new.copy()
        # drop specified columns if any
        drop_cols = self.config.get("drop_columns") or []
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # one-hot encode same categorical columns
        if self.config.get("dummy", True) and self._categorical_columns:
            df = pd.get_dummies(df, columns=self._categorical_columns, drop_first=False)

        # ensure same columns as training preprocessing
        for col in self._feature_columns:
            if col not in df.columns:
                df[col] = 0
        # drop extra columns that were not seen before
        extra = [c for c in df.columns if c not in self._feature_columns]
        if extra:
            df = df.drop(columns=extra)

        df = df[self._feature_columns]  # order columns
        df = self._apply_scaler(df)

        return df.values if self.config.get("return_type") == "numpy" else df

    # convenience shorthand
    def fit_split(self) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
        return self.split()

