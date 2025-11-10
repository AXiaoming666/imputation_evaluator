import pandas as pd


def RMSE(original_data: pd.DataFrame | pd.Series, imputed_data: pd.DataFrame | pd.Series) -> float:
    squared_diffs = (original_data - imputed_data) ** 2

    if isinstance(squared_diffs, pd.Series):
        mean_squared_error = squared_diffs.mean()
    else:
        mean_squared_error = squared_diffs.mean().mean()

    return mean_squared_error ** 0.5


def MAE(original_data: pd.DataFrame | pd.Series, imputed_data: pd.DataFrame | pd.Series) -> float:
    abs_diffs = (original_data - imputed_data).abs()
    if isinstance(abs_diffs, pd.Series):
        abs_diffs = abs_diffs.mean()
    else:
        abs_diffs = abs_diffs.mean().mean()
    
    return abs_diffs


def R2(original_data: pd.DataFrame | pd.Series, imputed_data: pd.DataFrame | pd.Series) -> float:
    ss_res = ((original_data - imputed_data) ** 2).sum().sum()
    ss_tot = ((original_data - original_data.mean()) ** 2).sum().sum()
    
    return 1 - (ss_res / ss_tot)