import pandas as pd


def RMSE(original_data: pd.DataFrame, imputed_data: pd.DataFrame) -> float:
    squared_diffs = (original_data - imputed_data) ** 2

    mean_squared_error = squared_diffs.mean().mean()

    return mean_squared_error ** 0.5


def MAE(original_data: pd.DataFrame, imputed_data: pd.DataFrame) -> float:
    abs_diffs = (original_data - imputed_data).abs()
    
    return abs_diffs.mean().mean()


def R2(original_data: pd.DataFrame, imputed_data: pd.DataFrame) -> float:
    ss_res = ((original_data - imputed_data) ** 2).sum().sum()
    ss_tot = ((original_data - original_data.mean()) ** 2).sum().sum()
    
    return 1 - (ss_res / ss_tot)