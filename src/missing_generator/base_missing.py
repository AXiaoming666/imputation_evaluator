import numpy as np
from scipy import optimize
from typing import List

from src.data_manager import DataManager


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def MCAR(
    dm: DataManager,
    target_column_names: List[str],
    missing_rate: float,
    random_state: int
) -> None:
    
    rng = np.random.RandomState(random_state)
    
    dm.missing_mask[target_column_names] = rng.rand(dm.meta_info["num_samples"], len(target_column_names)) < missing_rate


def MAR(
    dm: DataManager,
    target_column_names: List[str],
    missing_rate: float,
    random_state: int
) -> None:
    
    rng = np.random.RandomState(random_state)

    missing_related_features = dm.loaded_data[dm.meta_info["feature_names"]].drop(columns=target_column_names)

    coeffs: np.ndarray = rng.randn(missing_related_features.shape[1], len(target_column_names))
    Wx: np.ndarray = np.dot(missing_related_features, coeffs)
    coeffs /= np.std(Wx, axis=0, keepdims=True)
    
    def f(x: np.ndarray) -> float:
        return sigmoid(np.dot(missing_related_features, coeffs) + x).mean() - missing_rate

    intercepts = optimize.bisect(f, -10, 10)
    ps = sigmoid(np.dot(missing_related_features, coeffs) + intercepts)
    ber = rng.rand(dm.meta_info["num_samples"], len(target_column_names))
    dm.missing_mask[target_column_names] = ber < ps


def MNAR(
    dm: DataManager,
    target_column_names: List[str],
    missing_rate: float,
    random_state: int
) -> None:
    
    rng = np.random.RandomState(random_state)
    
    for target_column_name in target_column_names:
        missing_related_feature = dm.loaded_data[target_column_name].values.reshape(-1, 1) # type: ignore
        
        coeffs: np.ndarray = rng.randn(1, 1)
        Wx: np.ndarray = np.dot(missing_related_feature, coeffs)
        coeffs /= np.std(Wx, axis=0, keepdims=True)
        
        def f(x: np.ndarray) -> float:
            return sigmoid(np.dot(missing_related_feature, coeffs) + x).mean() - missing_rate

        intercepts = optimize.bisect(f, -10, 10)
        ps = sigmoid(np.dot(missing_related_feature, coeffs) + intercepts)
        ber = rng.rand(dm.meta_info["num_samples"], 1)
        dm.missing_mask[target_column_name] = ber < ps