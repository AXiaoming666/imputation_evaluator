import numpy as np
import pandas as pd
from typing import Optional

from src.models.base_model import BaseImputationModel


class MeanImputeModel(BaseImputationModel):
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.means: np.ndarray

    def fit(
        self,
        X: pd.DataFrame,
        time_stamp: pd.Series
    ) -> "MeanImputeModel":
        X_numpy = self._convert_to_numpy(X)
        self.means = np.nanmean(X_numpy, axis=0)
        return self

    def impute(
        self,
        X: pd.DataFrame,
        time_stamp: pd.Series
    ) -> pd.DataFrame:
        X_numpy = self._convert_to_numpy(X)
        mask = np.isnan(X_numpy)
        X_numpy[mask] = np.take(self.means, np.where(mask)[1])
        return self._convert_to_dataframe(X_numpy)
    

if __name__ == "__main__":
    from src.data_manager import DataManager
    from src.missing_generator.generator import generate_missing_data
    dm = DataManager()
    datasets = dm.scan_datasets()
    if datasets:
        dm.load_dataset(datasets[0])
        generate_missing_data(dm, target_column=["OT"], missing_rate=0.3, missing_type="MCAR", random_state=42)
        model = MeanImputeModel()
        imputed_data = model(dm.missing_data)
        dm.imputed_data = imputed_data
        print(dm.imputed_data.head())