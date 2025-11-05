from typing import List

from src.data_manager import DataManager
from src.missing_generator.base_missing import MCAR, MAR, MNAR
from src.missing_generator.iot_missing import line_missing, block_missing
from src.missing_generator.validator import validate


def generate_missing_data(
    dm: DataManager,
    missing_rate: float,
    missing_type: str,
    random_state: int = 42,
    gap_len: int | None = None,
    target_column: List[str] | None = None
) -> float:
    
    if missing_type == "MCAR":
        if target_column is None:
            raise ValueError("target_column must be specified for MCAR missing type.")
        MCAR(dm, target_column, missing_rate, random_state)
    elif missing_type == "MAR":
        if target_column is None:
            raise ValueError("target_column must be specified for MAR missing type.")
        MAR(dm, target_column, missing_rate, random_state)
    elif missing_type == "MNAR":
        if target_column is None:
            raise ValueError("target_column must be specified for MNAR missing type.")
        MNAR(dm, target_column, missing_rate, random_state)
    elif missing_type == "LINE":
        line_missing(dm, missing_rate, random_state)
    elif missing_type == "BLOCK":
        if gap_len is None:
            raise ValueError("gap_len must be specified for BLOCK missing type.")
        block_missing(dm, missing_rate, gap_len, random_state)
    else:
        raise ValueError(f"Unknown missing type: {missing_type}")
    
    return validate(dm, missing_rate, missing_type)
