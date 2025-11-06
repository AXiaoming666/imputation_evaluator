import pandas as pd
import numpy as np
from scipy.stats import entropy, ks_2samp
import ot


def KL_divergence(original_col: pd.Series, imputed_col: pd.Series) -> float:
    p = imputed_col / imputed_col.sum()
    q = original_col / original_col.sum()

    kl_div = float(entropy(p, q))
    return kl_div


def KS_statistic(original_col: pd.Series, imputed_col: pd.Series) -> float:
    ks_stat: float = ks_2samp(original_col, imputed_col)[0]  # type: ignore
    return ks_stat


def w2_distance(original_col: pd.Series, imputed_col: pd.Series) -> float:
    x1 = original_col.values.reshape(-1, 1)  # type: ignore
    x2 = imputed_col.values.reshape(-1, 1)  # type: ignore
    a = np.ones(len(x1)) / len(x1)
    b = np.ones(len(x2)) / len(x2)
    
    M = ot.dist(x1, x2, metric='euclidean')
    M_squared = M ** 2
    
    w2_squared = ot.emd2(a, b, M_squared)
    w2_distance = float(np.sqrt(w2_squared))

    return w2_distance