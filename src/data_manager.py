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
        self.if_normalized: bool = False


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
        self.z_score_normalize()
        
        self.missing_mask = pd.DataFrame(False, index=self.loaded_data.index, columns=self.loaded_data.columns)
        

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
            "mean": self.loaded_data[self.loaded_data.drop("date", axis=1).columns].mean().to_dict(),
            "std": self.loaded_data[self.loaded_data.drop("date", axis=1).columns].std().to_dict()
        }
    
    
    def z_score_normalize(self) -> None:
        if not self.if_loaded or self.loaded_data is None:
            raise RuntimeError("No dataset loaded. Please load a dataset first.")

        feature_cols: pd.Index = self.loaded_data.drop("date", axis=1).columns
        for col in feature_cols:
            mean = self.meta_info["mean"][col]
            std = self.meta_info["std"][col]
            self.loaded_data[col] = (self.loaded_data[col] - mean) / std

        self.if_normalized = True

if __name__ == "__main__":
    manager = DataManager()
    datasets = manager.scan_datasets()
    print("Available datasets:", datasets)
    if datasets:
        manager.load_dataset(datasets[0])
        print("Meta information:", manager.meta_info)
        
        print("First 5 rows of normalized data:")
        print(manager.loaded_data.head())