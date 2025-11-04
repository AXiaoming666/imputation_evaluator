from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd


from src.data_manager import DataManager
from src.missing_generator.generator import generate_missing_data

def plot_missing_mask(dm: DataManager) -> None:

    missing_feature_names: List[str] = dm.missing_mask.any(axis=0).index[dm.missing_mask.any(axis=0)].tolist()
    missing_feature_mask = dm.missing_mask[missing_feature_names]
    
    plt.figure(figsize=(12, 6))
    plt.imshow(missing_feature_mask.T, aspect='auto', cmap='gray_r')
    plt.colorbar(label='缺失情况 (1=缺失, 0=存在)')
    plt.yticks(ticks=np.arange(len(missing_feature_names)), labels=missing_feature_names)
    plt.xlabel('时间轴')
    plt.title('缺失数据掩码可视化')
    plt.show()


def plot_missing_relationship(dm: DataManager) -> None:

    missing_feature_names: List[str] = dm.missing_mask.any(axis=0).index[dm.missing_mask.any(axis=0)].tolist()
    missing_feature_mask = dm.missing_mask[missing_feature_names]
    feature_values = dm.loaded_data[dm.meta_info["feature_names"]]

    fig, axes = plt.subplots(len(missing_feature_names), feature_values.shape[1], 
                            figsize=(2 * len(feature_values.columns), 2 * len(missing_feature_names)))
    
    if len(missing_feature_names) == 1 and feature_values.shape[1] == 1:
        axes = np.array([[axes]])
    elif len(missing_feature_names) == 1:
        axes = axes.reshape(1, -1)
    elif feature_values.shape[1] == 1:
        axes = axes.reshape(-1, 1)

    for i, missing_col in enumerate(missing_feature_names):
        for j, feature_col in enumerate(feature_values.columns):
            axes[i, j].scatter(feature_values[feature_col], missing_feature_mask[missing_col], alpha=0.01, s=5)
            axes[i, j].set_xlabel(feature_col)
            axes[i, j].set_ylabel(f'{missing_col}缺失情况')
            axes[i, j].set_title(f'与{feature_col}的关系')

    plt.tight_layout()
    plt.show()
    

def plot_pearson_heatmap(dm: DataManager) -> None:
    feature_columns = dm.loaded_data[dm.meta_info["feature_names"]]
    missing_feature_names: List[str] = dm.missing_mask.any(axis=0).index[dm.missing_mask.any(axis=0)].tolist()
    missing_feature_mask = dm.missing_mask[missing_feature_names]

    corr_matrix = pd.DataFrame(
        index=missing_feature_names,
        columns=dm.meta_info["feature_names"]
    )

    for target_col in missing_feature_mask.columns:
        for feature_col in feature_columns.columns:
            x = missing_feature_mask[target_col]
            y = feature_columns[feature_col]

            corr: float = pearsonr(x, y)[0]  # type: ignore

            corr_matrix.loc[target_col, feature_col] = corr
    
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix.astype(float), cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Pearson相关系数')
    plt.xticks(ticks=np.arange(len(corr_matrix.columns)), labels=corr_matrix.columns.tolist(), rotation=45)
    plt.yticks(ticks=np.arange(len(corr_matrix.index)), labels=corr_matrix.index.tolist())
    plt.title('缺失特征与其他特征的Pearson相关系数热图')
    plt.show()


if __name__ == "__main__":
    dm = DataManager()
    datasets = dm.scan_datasets()
    if datasets:
        dm.load_dataset(datasets[0])
        generate_missing_data(dm, target_column=["OT"], missing_rate=0.4, missing_type="LINE", gap_len=1, random_state=42)
        plot_missing_mask(dm)