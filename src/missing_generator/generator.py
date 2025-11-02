from typing import List

from src.data_manager import DataManager
from src.missing_generator.base_missing import MCAR, MAR, MNAR
from src.missing_generator.validator import validate


def generate_missing_data(
    dm: DataManager,
    target_column: List[str],
    missing_rate: float,
    missing_type: str,
    random_state: int
) -> None:
    if missing_type == "MCAR":
        MCAR(dm, target_column, missing_rate, random_state)
    elif missing_type == "MAR":
        MAR(dm, target_column, missing_rate, random_state)
    elif missing_type == "MNAR":
        MNAR(dm, target_column, missing_rate, random_state)
    else:
        raise ValueError(f"Unknown missing type: {missing_type}")

    validate(dm, missing_rate, missing_type, target_column)


if __name__ == "__main__":
    dm = DataManager()
    datasets = dm.scan_datasets()
    if datasets:
        dm.load_dataset(datasets[0])
        for missing_type in ["MCAR", "MAR", "MNAR"]:
            generate_missing_data(dm, target_column=["OT"], missing_rate=0.2, missing_type=missing_type, random_state=42)