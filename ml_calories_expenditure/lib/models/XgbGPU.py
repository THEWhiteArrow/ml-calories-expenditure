import numpy as np
import pandas as pd
import scipy.sparse as sp
from xgboost import XGBClassifier, XGBRegressor

from ml_calories_expenditure.lib.logger import setup_logger

try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = None

try:
    import cudf  # type: ignore
except ImportError:
    cudf = None

logger = setup_logger(__name__)


class XGBClassifierGPU(XGBClassifier):
    """SHEESH! This is a GPU compatible version of XGBClassifier. It uses the GPU for training and prediction if possible."""

    def __init__(self, **kwargs):

        kwargs.setdefault("device", "cuda")
        kwargs.setdefault("tree_method", "hist")
        super().__init__(**kwargs)

    def _use_gpu(self):
        # Decide based on the actual XGBClassifier parameters
        return (
            getattr(self, "tree_method", None) == "hist"
            and getattr(self, "device", None) == "cuda"
        )

    def _toggle_gpu(self, use_gpu: bool) -> "XGBClassifierGPU":
        """Toggle GPU usage for XGBoost."""
        if use_gpu:
            logger.info("Using GPU for XGBoost.")
            self.set_params(tree_method="hist", device="cuda")
        else:
            logger.warning("Using CPU for XGBoost. This may be slower than using GPU.")
            self.set_params(tree_method=None, device=None)

        return self

    def _to_gpu_array(self, X):

        if not self._use_gpu():
            return X
        # Pandas DataFrame -> cuDF if available, else CuPy, else NumPy
        elif isinstance(X, pd.DataFrame):
            if cudf is not None:
                return cudf.DataFrame.from_pandas(X)
            elif cp is not None:
                return cp.array(X.values)
            else:
                return X.values
        elif isinstance(X, np.ndarray):
            if cp is not None:
                return cp.array(X)
            else:
                return X
        elif sp.issparse(X):
            X_dense = X.toarray()
            if cp is not None:
                return cp.array(X_dense)
            else:
                return X_dense
        elif cudf is not None and isinstance(X, cudf.DataFrame):
            return X
        elif cp is not None and isinstance(X, cp.ndarray):
            return X
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

    def _to_gpu_label(self, y):
        if not self._use_gpu():
            return y
        elif isinstance(y, pd.Series):
            if cudf is not None:
                return cudf.Series(y)
            elif cp is not None:
                return cp.array(y.values)
            else:
                return y.values
        elif isinstance(y, np.ndarray):
            if cp is not None:
                return cp.array(y)
            else:
                return y
        elif cudf is not None and isinstance(y, cudf.Series):
            return y
        elif cp is not None and isinstance(y, cp.ndarray):
            return y
        else:
            return y

    def fit(self, X, y, **kwargs):
        X_fin = self._to_gpu_array(X)
        y_fin = self._to_gpu_label(y)
        return super().fit(X_fin, y_fin, **kwargs)

    def predict(self, X, **kwargs):
        X_fin = self._to_gpu_array(X)
        return super().predict(X_fin, **kwargs)

    def predict_proba(self, X, **kwargs):
        X_fin = self._to_gpu_array(X)
        return super().predict_proba(X_fin, **kwargs)


class XGBRegressorGPU(XGBRegressor):
    """SHEESH! This is a GPU compatible version of XGBRegressor. It uses the GPU for training and prediction if possible."""

    def __init__(self, **kwargs):

        kwargs.setdefault("device", "cuda")
        kwargs.setdefault("tree_method", "hist")
        super().__init__(**kwargs)

    def _use_gpu(self):
        # Decide based on the actual XGBClassifier parameters
        return (
            getattr(self, "tree_method", None) == "hist"
            and getattr(self, "device", None) == "cuda"
        )

    def _toggle_gpu(self, use_gpu: bool) -> "XGBClassifierGPU":
        """Toggle GPU usage for XGBoost."""
        if use_gpu:
            logger.info("Using GPU for XGBoost.")
            self.set_params(tree_method="hist", device="cuda")
        else:
            logger.warning("Using CPU for XGBoost. This may be slower than using GPU.")
            self.set_params(tree_method=None, device=None)

        return self

    def _to_gpu_array(self, X):

        if not self._use_gpu():
            return X
        # Pandas DataFrame -> cuDF if available, else CuPy, else NumPy
        elif isinstance(X, pd.DataFrame):
            if cudf is not None:
                return cudf.DataFrame.from_pandas(X)
            elif cp is not None:
                return cp.array(X.values)
            else:
                return X.values
        elif isinstance(X, np.ndarray):
            if cp is not None:
                return cp.array(X)
            else:
                return X
        elif sp.issparse(X):
            X_dense = X.toarray()
            if cp is not None:
                return cp.array(X_dense)
            else:
                return X_dense
        elif cudf is not None and isinstance(X, cudf.DataFrame):
            return X
        elif cp is not None and isinstance(X, cp.ndarray):
            return X
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")

    def _to_gpu_label(self, y):
        if not self._use_gpu():
            return y
        elif isinstance(y, pd.Series):
            if cudf is not None:
                return cudf.Series(y)
            elif cp is not None:
                return cp.array(y.values)
            else:
                return y.values
        elif isinstance(y, np.ndarray):
            if cp is not None:
                return cp.array(y)
            else:
                return y
        elif cudf is not None and isinstance(y, cudf.Series):
            return y
        elif cp is not None and isinstance(y, cp.ndarray):
            return y
        else:
            return y

    def fit(self, X, y, **kwargs):
        X_fin = self._to_gpu_array(X)
        y_fin = self._to_gpu_label(y)
        return super().fit(X_fin, y_fin, **kwargs)

    def predict(self, X, **kwargs):
        X_fin = self._to_gpu_array(X)
        return super().predict(X_fin, **kwargs)
