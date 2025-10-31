import pandas as pd
import os
from typing import TypedDict, Dict, List
from datetime import datetime


class TimeInfo(TypedDict):
    start: datetime
    end: datetime
    frequency: str | None

class MetaInfo(TypedDict):
    num_samples: int
    num_features: int
    feature_names: List[str]
    time: TimeInfo
    mean: Dict[str, float]
    std: Dict[str, float]


class DataManager:
    def __init__(self, datasets_dir: str = "datasets/") -> None:
        self.datasets_dir: str = datasets_dir

        self.if_loaded: bool = False
        self.if_missing: bool = False
        self.if_imputed: bool = False


    def scan_datasets(self) -> List[str]:
        if not os.path.exists(self.datasets_dir):
            raise FileNotFoundError(f"The directory {self.datasets_dir} does not exist.")

        datasets: List[str] = [
            name.split(".")[0] for name in os.listdir(self.datasets_dir)
        ]
        return datasets


    def load_dataset(self, dataset_name: str) -> None:
        if dataset_name not in self.scan_datasets():
            raise ValueError(f"Dataset {dataset_name} not found in {self.datasets_dir}.")

        dataset_path: str = os.path.join(self.datasets_dir, f"{dataset_name}.csv")
        self.loaded_data = pd.read_csv(dataset_path)
        self.if_loaded = True

        if not "date" in self.loaded_data.columns:
            raise ValueError("The dataset must contain a 'date' column.")

        self.extract_meta_info()
        

    def extract_meta_info(self) -> None:
        if not self.if_loaded or self.loaded_data is None:
            raise RuntimeError("No dataset loaded. Please load a dataset first.")
        if pd.infer_freq(self.loaded_data["date"]) is None:
            raise ValueError("The 'date' column must have a consistent frequency.")

        self.meta_info: MetaInfo = {
            "num_samples": self.loaded_data.shape[0],
            "num_features": self.loaded_data.shape[1] - 1,
            "feature_names": [col for col in self.loaded_data.columns if col != "date"],
            "time": {
                "start": self.loaded_data["date"].min(),
                "end": self.loaded_data["date"].max(),
                "frequency": pd.infer_freq(self.loaded_data["date"])
            },
            "mean": self.loaded_data[self.loaded_data.columns[1:]].mean().to_dict(),
            "std": self.loaded_data[self.loaded_data.columns[1:]].std().to_dict()
        }
    


if __name__ == "__main__":
    manager = DataManager()
    datasets = manager.scan_datasets()
    print("Available datasets:", datasets)
    if datasets:
        manager.load_dataset(datasets[0])
        manager.extract_meta_info()
        print("Meta information:", manager.meta_info)