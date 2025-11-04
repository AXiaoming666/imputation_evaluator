import numpy as np
from scipy import optimize
from typing import List

from src.data_manager import DataManager


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def MCAR(
    dm: DataManager,
    target_column_names: List[str],
    missing_rate: float,
    random_state: int
) -> None:
    
    rng = np.random.RandomState(random_state)
    
    P = np.full((dm.meta_info["num_samples"], len(target_column_names)), missing_rate)
    R = rng.rand(dm.meta_info["num_samples"], len(target_column_names))
    
    dm.missing_mask[target_column_names] = R < P


def MAR(
    dm: DataManager,
    target_column_names: List[str],
    missing_rate: float,
    random_state: int
) -> None:
    
    rng = np.random.RandomState(random_state)

    X_prime = np.array(dm.loaded_data[dm.meta_info["feature_names"]].drop(columns=target_column_names).values)
    C = rng.rand(X_prime.shape[1], len(target_column_names))
    C /= np.std(np.dot(X_prime, C), axis=0, keepdims=True)
    I = np.dot(X_prime, C)
    B = np.zeros((1, len(target_column_names)))

    for d in range(len(target_column_names)):
        def f(x: np.ndarray) -> float:
            return sigmoid(I[:, d] + x).mean() - missing_rate

        B[0, d] = optimize.bisect(f, -10, 10)  # type: ignore

    P = sigmoid(I + np.dot(np.ones((dm.meta_info["num_samples"], 1)), B))
    R = rng.rand(dm.meta_info["num_samples"], len(target_column_names))
    dm.missing_mask[target_column_names] = R < P

def MNAR(
    dm: DataManager,
    target_column_names: List[str],
    missing_rate: float,
    random_state: int
) -> None:
    
    rng = np.random.RandomState(random_state)

    for d in range(len(target_column_names)):
        X_prime = np.array(dm.loaded_data[target_column_names[d]].values).reshape(-1, 1)
        C = rng.rand(1, 1)
        C /= np.std(np.dot(X_prime, C), axis=0, keepdims=True)
        I = np.dot(X_prime, C)
        B = np.zeros((1, 1))
        
        def f(x: np.ndarray) -> float:
            return sigmoid(I[:, 0] + x).mean() - missing_rate

        B[0, 0] = optimize.bisect(f, -10, 10)  # type: ignore
        P = sigmoid(I + np.dot(np.ones((dm.meta_info["num_samples"], 1)), B))
        R = rng.rand(dm.meta_info["num_samples"], 1)
        dm.missing_mask[target_column_names[d]] = R < P