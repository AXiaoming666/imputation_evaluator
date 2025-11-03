import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from typing import List

from src.data_manager import DataManager


def validate(
    dm: DataManager,
    missing_rate: float,
    missing_type: str,
    target_column_names: List[str]
) -> None:
    check_missing_rate(dm, missing_rate, target_column_names)
    check_missing_type(dm, missing_type, target_column_names)


def check_missing_rate(dm: DataManager, missing_rate: float, target_column_names: List[str]) -> None:
    actual_missing_rate = dm.missing_mask[target_column_names].mean(axis=0)
    for col, rate in actual_missing_rate.items():
        if not np.isclose(rate, missing_rate, atol=0.01):
            raise ValueError(f"Expected missing rate {missing_rate}, but got {actual_missing_rate} in column {col}")


def check_missing_type(dm: DataManager, missing_type: str, target_column_names: List[str], alpha: float = 1e-5) -> None:
    feature_columns = dm.loaded_data[dm.meta_info["feature_names"]]
    target_mask = dm.missing_mask[target_column_names]
    
    significance_matrix = pd.DataFrame(
        index=target_column_names,
        columns=dm.meta_info["feature_names"]
    )
    
    for target_col in target_mask.columns:
        for feature_col in feature_columns.columns:
            x = target_mask[target_col]
            y = feature_columns[feature_col]
            
            p_value: float = pearsonr(x, y)[1] # type: ignore
            
            significance_matrix.loc[target_col, feature_col] = p_value < alpha
            
    
    if missing_type == "MCAR":
        for target_col in target_column_names:
            for feature_col in dm.meta_info["feature_names"]:
                if significance_matrix.loc[target_col, feature_col] == True:
                    raise ValueError(f"MCAR validation failed: Feature {feature_col} is significantly associated with missingness in target column {target_col}.")
    elif missing_type == "MAR":
        for target_col in target_column_names:
            associated = False
            for feature_col in dm.meta_info["feature_names"]:
                if feature_col in target_column_names:
                    continue
                if significance_matrix.loc[target_col, feature_col] == True:
                    associated = True
                    break
            if not associated:
                raise ValueError(f"MAR validation failed: No features are significantly associated with missingness in target column {target_col}.")
    elif missing_type == "MNAR":
        for target_col in target_column_names:
            if significance_matrix.loc[target_col, target_col] == False:
                raise ValueError(f"MNAR validation failed: Missingness in target column {target_col} is not significantly associated with itself.")
    else:
        raise ValueError(f"Unknown missing type: {missing_type}")
