import pandas as pd
import numpy as np
import os
from typing import TypedDict, Dict, List, Optional
from datetime import datetime


class TimeInfo(TypedDict):
    start: datetime
    end: datetime
    frequency: Optional[str]

class MetaInfo(TypedDict):
    num_samples: int
    num_features: int
    feature_names: List[str]
    time: TimeInfo
    mean: Dict[str, float]
    std: Dict[str, float]


class DataManager:
    def __init__(
        self,
        datasets_dir: str = "datasets/"
    ) -> None:
        self.datasets_dir: str = datasets_dir
        self.missing_mask: pd.DataFrame

        self.if_loaded: bool = False
        self.if_normalized: bool = False
        self.if_imputed: bool = False   
    
    @property
    def loaded_data(
        self
    ) -> pd.DataFrame:
        if not self.if_loaded:
            raise RuntimeError("No dataset loaded. Please load a dataset first.")
        return self._loaded_data
    
    @loaded_data.setter
    def loaded_data(
        self,
        data: pd.DataFrame
    ) -> None:
        self._loaded_data = data
        
    @property
    def meta_info(
        self
    ) -> MetaInfo:
        if not self.if_loaded:
            raise RuntimeError("No dataset loaded. Please load a dataset first.")
        return self._meta_info
    
    @meta_info.setter
    def meta_info(
        self,
        info: MetaInfo
    ) -> None:
        self._meta_info = info
        
    @property
    def missing_data(
        self
    ) -> pd.DataFrame:
        if not self.if_loaded:
            raise RuntimeError("No dataset loaded. Please load a dataset first.")
        missing_data = self.loaded_data.copy()
        missing_data[self.missing_mask] = pd.NA
        return missing_data
        
    @property
    def imputed_data(
        self
    ) -> pd.DataFrame:
        if not self.if_imputed:
            raise RuntimeError("Data has not been imputed yet.")
        return self._imputed_data
    
    @imputed_data.setter
    def imputed_data(
        self,
        data: pd.DataFrame
    ) -> None:
        self.check_imputation(data)
        self._imputed_data = data
        self.if_imputed = True

    def scan_datasets(
        self
    ) -> List[str]:
        if not os.path.exists(self.datasets_dir):
            raise FileNotFoundError(f"The directory {self.datasets_dir} does not exist.")

        datasets: List[str] = [
            name.split(".")[0] for name in os.listdir(self.datasets_dir)
        ]
        return datasets

    def load_dataset(
        self,
        dataset_name: str
    ) -> None:
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
        
    def extract_meta_info(
        self
    ) -> None:
        if not self.if_loaded or self.loaded_data is None:
            raise RuntimeError("No dataset loaded. Please load a dataset first.")
        if pd.infer_freq(self.loaded_data["date"]) is None:
            raise ValueError("The 'date' column must have a consistent frequency.")

        self.meta_info = {
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
    
    def z_score_normalize(
        self
    ) -> None:
        if not self.if_loaded or self.loaded_data is None:
            raise RuntimeError("No dataset loaded. Please load a dataset first.")

        feature_cols: pd.Index = self.loaded_data.drop("date", axis=1).columns
        for col in feature_cols:
            mean = self.meta_info["mean"][col]
            std = self.meta_info["std"][col]
            self.loaded_data[col] = (self.loaded_data[col] - mean) / std

        self.if_normalized = True
    
    def check_imputation(
        self,
        imputed_data: pd.DataFrame
    ) -> None:
        if imputed_data.isna().any().any():
            raise ValueError("The imputed data contains NaN values.")
        
        original_values = self.loaded_data[self.meta_info["feature_names"]]
        imputed_values = imputed_data[self.meta_info["feature_names"]]
        
        if_equal = np.isclose(original_values, imputed_values, rtol=1e-05, atol=1e-08)
        if_equal_masked = if_equal | self.missing_mask[self.meta_info["feature_names"]]
        
        if not if_equal_masked.all().all():
            raise ValueError("The imputed data does not match the original data at non-missing positions.")
        
if __name__ == "__main__":
    manager = DataManager()
    datasets = manager.scan_datasets()
    print("Available datasets:", datasets)
    if datasets:
        manager.load_dataset(datasets[0])
        print("Meta information:", manager.meta_info)
        
        print("First 5 rows of normalized data:")
        print(manager.loaded_data.head())