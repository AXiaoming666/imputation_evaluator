import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseImputationModel(ABC):
    def __init__(
        self,
        **kwargs
    ) -> None:
        pass
        
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        time_stamp: pd.Series
    ) -> "BaseImputationModel":
        raise NotImplementedError("must implement fit method in subclass")
    
    @abstractmethod
    def impute(
        self,
        X: pd.DataFrame,
        time_stamp: pd.Series
    ) -> pd.DataFrame:
        raise NotImplementedError("must implement impute method in subclass")
    
    def __call__(
        self,
        time_series: pd.DataFrame
    ) -> pd.DataFrame:
        self.time_series = time_series
        time_stamp = time_series["date"]
        X = time_series.drop(columns=["date"])
        self._check_if_missing(X)
        self.fit(X, time_stamp)
        return self.impute(X, time_stamp)

    def _convert_to_numpy(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        return X.values
    
    def _convert_to_dataframe(
        self,
        X: np.ndarray
    ) -> pd.DataFrame:
        df = self.time_series.copy()
        imputed_values = pd.DataFrame(X, columns=df.columns.drop("date"), index=df.index)
        for col in imputed_values.columns:
            df[col] = imputed_values[col]
        return df

    def _check_if_missing(
        self,
        X: pd.DataFrame
    ) -> None:
        if not bool(X.isnull().values.any()):
            raise ValueError("No missing values found in the input data.")
