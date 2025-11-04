import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from typing import List

from src.data_manager import DataManager


def validate(
    dm: DataManager,
    missing_rate: float,
    missing_type: str
) -> float:
    missing_feature_names: List[str] = dm.missing_mask.any(axis=0).index[dm.missing_mask.any(axis=0)].tolist()
    
    actual_missing_rate = check_missing_rate(dm, missing_rate, missing_feature_names)
    
    check_missing_type(dm, missing_type, missing_feature_names)
    
    return actual_missing_rate


def check_missing_rate(dm: DataManager, missing_rate: float, missing_feature_names: List[str]) -> float:
    actual_missing_rate = dm.missing_mask[missing_feature_names].mean(axis=0)
    for col, rate in actual_missing_rate.items():
        if not np.isclose(rate, missing_rate, atol=0.01):
            raise ValueError(f"Expected missing rate {missing_rate}, but got {rate} in column {col}")
    return actual_missing_rate.mean()


def check_missing_type(dm: DataManager, missing_type: str, missing_feature_names: List[str], alpha: float = 1e-5) -> None:
    feature_columns = dm.loaded_data[dm.meta_info["feature_names"]]
    target_mask = dm.missing_mask[missing_feature_names]
    
    if missing_type in ["LINE", "BLOCK"]:
        any_missing = target_mask.any(axis=1)
        all_missing = target_mask.all(axis=1)
        partly_missing = any_missing & ~all_missing
        if partly_missing.any():
            first_invalid_row = partly_missing[partly_missing].index[0]
            raise ValueError(f"{missing_type} missing validation failed: Row {first_invalid_row} is missing partly.")
    elif missing_type in ["MCAR", "MAR", "MNAR"]:
        significance_matrix = pd.DataFrame(
            index=missing_feature_names,
            columns=dm.meta_info["feature_names"]
        )
        
        for target_col in target_mask.columns:
            for feature_col in feature_columns.columns:
                binary = target_mask[target_col].values
                numerical = feature_columns[feature_col].values

                p_value: float = pointbiserialr(binary, numerical).pvalue  # type: ignore

                significance_matrix.loc[target_col, feature_col] = p_value < alpha
        
        if missing_type == "MCAR":
            for target_col in missing_feature_names:
                for feature_col in dm.meta_info["feature_names"]:
                    if significance_matrix.loc[target_col, feature_col] == True:
                        raise ValueError(f"MCAR validation failed: Feature {feature_col} is significantly associated with missingness in target column {target_col}.")
        elif missing_type == "MAR":
            for target_col in missing_feature_names:
                associated = False
                for feature_col in dm.meta_info["feature_names"]:
                    if feature_col in missing_feature_names:
                        continue
                    if significance_matrix.loc[target_col, feature_col] == True:
                        associated = True
                        break
                if not associated:
                    raise ValueError(f"MAR validation failed: No features are significantly associated with missingness in target column {target_col}.")
        elif missing_type == "MNAR":
            for target_col in missing_feature_names:
                if significance_matrix.loc[target_col, target_col] == False:
                    raise ValueError(f"MNAR validation failed: Missingness in target column {target_col} is not significantly associated with itself.")
    else:
        raise ValueError(f"Unknown missing type for validation: {missing_type}")
