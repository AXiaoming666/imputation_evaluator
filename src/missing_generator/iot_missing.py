import numpy as np

from src.data_manager import DataManager


def line_missing(
    dm: DataManager,
    missing_rate: float,
    random_state: int
) -> None:
    
    rng = np.random.RandomState(random_state)

    P = np.full((dm.meta_info["num_samples"], 1), missing_rate)
    R = rng.rand(dm.meta_info["num_samples"], 1)
    M = (R < P)
    M = np.repeat(M, dm.meta_info["num_features"], axis=1)

    dm.missing_mask[dm.meta_info["feature_names"]] = M


def block_missing(
    dm: DataManager,
    missing_rate: float,
    gap_len: int,
    random_state: int
) -> None:
    
    rng = np.random.RandomState(random_state)
    
    num_blocks = int(np.ceil(dm.meta_info["num_samples"] / gap_len))

    P = np.full((num_blocks, 1), missing_rate)
    R = rng.rand(num_blocks, 1)
    M = (R < P)
    M = np.repeat(M, gap_len, axis=0)
    M = M[:dm.meta_info["num_samples"]]
    M = np.tile(M, (1, dm.meta_info["num_features"]))

    dm.missing_mask[dm.meta_info["feature_names"]] = M