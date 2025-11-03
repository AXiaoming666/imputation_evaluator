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
    gap_len: int,
    random_state: int
) -> None:
    
    rng = np.random.RandomState(random_state)
    
    df = dm.loaded_data[target_column_names]
    compressed_features = df.groupby(df.index // gap_len).mean()
    
    ps = np.full((compressed_features.shape[0], len(target_column_names)), missing_rate)
    ber = rng.rand(dm.meta_info["num_samples"], len(target_column_names))
    
    dm.missing_mask[target_column_names] = [[ber[i // gap_len, j] < ps[i // gap_len, j] for j in range(len(target_column_names))] for i in range(dm.meta_info["num_samples"])]


def MAR(
    dm: DataManager,
    target_column_names: List[str],
    missing_rate: float,
    gap_len: int,
    random_state: int
) -> None:
    
    rng = np.random.RandomState(random_state)
    
    df = dm.loaded_data[dm.meta_info["feature_names"]]
    compressed_features = df.groupby(df.index // gap_len).mean()

    missing_related_features = compressed_features.drop(columns=target_column_names)

    coeffs: np.ndarray = rng.randn(missing_related_features.shape[1], len(target_column_names))
    Wx: np.ndarray = np.dot(missing_related_features, coeffs)
    coeffs /= np.std(Wx, axis=0, keepdims=True)
    
    def f(x: np.ndarray) -> float:
        return sigmoid(np.dot(missing_related_features, coeffs) + x).mean() - missing_rate

    intercepts = optimize.bisect(f, -10, 10)
    ps = sigmoid(np.dot(missing_related_features, coeffs) + intercepts)
    ber = rng.rand(dm.meta_info["num_samples"] // gap_len, len(target_column_names))
    dm.missing_mask[target_column_names] = [[ber[i // gap_len, j] < ps[i // gap_len, j] for j in range(len(target_column_names))] for i in range(dm.meta_info["num_samples"])]


def MNAR(
    dm: DataManager,
    target_column_names: List[str],
    missing_rate: float,
    gap_len: int,
    random_state: int
) -> None:
    
    rng = np.random.RandomState(random_state)

    df = dm.loaded_data[target_column_names]
    compressed_features = df.groupby(df.index // gap_len).mean()

    for target_column_name in target_column_names:
        missing_related_feature = compressed_features.values.reshape(-1, 1) # type: ignore
        
        coeffs: np.ndarray = rng.randn(1, 1)
        Wx: np.ndarray = np.dot(missing_related_feature, coeffs)
        coeffs /= np.std(Wx, axis=0, keepdims=True)
        
        def f(x: np.ndarray) -> float:
            return sigmoid(np.dot(missing_related_feature, coeffs) + x).mean() - missing_rate

        intercepts = optimize.bisect(f, -10, 10)
        ps = sigmoid(np.dot(missing_related_feature, coeffs) + intercepts)
        ber = rng.rand(dm.meta_info["num_samples"], 1)
        dm.missing_mask[target_column_name] = [ber[i // gap_len, 0] < ps[i // gap_len, 0] for i in range(dm.meta_info["num_samples"])]