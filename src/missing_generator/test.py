from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import pointbiserialr
import pandas as pd
import seaborn as sns

from src.data_manager import DataManager
from src.missing_generator.generator import generate_missing_data


############################################################################
# functions for testing missing_generator
############################################################################


def test_all_conditions() -> None:
    dm = DataManager()
    datasets = dm.scan_datasets()
    if datasets:
        dm.load_dataset(datasets[0])
        for missing_type in ["MCAR", "MAR", "MNAR", "LINE"]:
            for missing_rate in [0.2, 0.4, 0.6, 0.8]:
                print(f"Generating missing data: type={missing_type}, rate={missing_rate}")
                generate_missing_data(dm, target_column=["OT"], missing_rate=missing_rate, missing_type=missing_type, random_state=42)
        for missing_rate in [0.2, 0.4, 0.6, 0.8]:
            for gap_len in [5, 10, 20]:
                print(f"Generating missing data: type=BLOCK, rate={missing_rate}, gap_len={gap_len}")
                generate_missing_data(dm, missing_rate=missing_rate, missing_type="BLOCK", gap_len=gap_len, random_state=42)


def compare_missing_types() -> None:
    dm1 = DataManager()
    dm2 = DataManager()
    dm3 = DataManager()
    
    datasets = dm1.scan_datasets()
    if not datasets:
        print("没有可用的数据集进行测试。")
        return

    dm1.load_dataset(datasets[0])
    dm2.load_dataset(datasets[0])
    dm3.load_dataset(datasets[0])
    
    generate_missing_data(dm1, target_column=["OT"], missing_rate=0.5, missing_type="MCAR", random_state=42)
    generate_missing_data(dm2, target_column=["OT"], missing_rate=0.5, missing_type="MAR", random_state=42)
    generate_missing_data(dm3, target_column=["OT"], missing_rate=0.5, missing_type="MNAR", random_state=42)
    
    plot_pointbiserialr_comparison(dm1, dm2, dm3)
    plot_relationships_comparison(dm1, dm2, dm3)


def plot_line_block_missing_mask() -> None:
    dm1 = DataManager()
    dm2 = DataManager()
    dm3 = DataManager()
    dm4 = DataManager()
    
    datasets = dm1.scan_datasets()
    if not datasets:
        print("没有可用的数据集进行测试。")
        return
    
    dm1.load_dataset(datasets[0])
    dm2.load_dataset(datasets[0])
    dm3.load_dataset(datasets[0])
    dm4.load_dataset(datasets[0])
    
    generate_missing_data(dm1, missing_rate=0.5, missing_type="LINE", random_state=42)
    generate_missing_data(dm2, missing_rate=0.5, missing_type="BLOCK", gap_len=10, random_state=42)
    generate_missing_data(dm3, missing_rate=0.5, missing_type="BLOCK", gap_len=20, random_state=42)
    generate_missing_data(dm4, missing_rate=0.5, missing_type="BLOCK", gap_len=30, random_state=42)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_missing_mask(dm1, axes[0][0])
    plot_missing_mask(dm2, axes[0][1])
    plot_missing_mask(dm3, axes[1][0])
    plot_missing_mask(dm4, axes[1][1])
    
    axes[0][0].set_title('线缺失')
    axes[0][1].set_title('块缺失 (缺失长度=10)')
    axes[1][0].set_title('块缺失 (缺失长度=20)')
    axes[1][1].set_title('块缺失 (缺失长度=30)')

    plt.tight_layout()
    plt.show()


############################################################################
# functions for plotting missing data characteristics
############################################################################


def plot_relationships_comparison(dm1: DataManager, dm2: DataManager, dm3: DataManager) -> None:
    fig, axes = plt.subplots(3, dm1.meta_info["num_features"], figsize=(12, 5))
    
    plot_missing_relationship(dm1, axes[0])
    plot_missing_relationship(dm2, axes[1])
    plot_missing_relationship(dm3, axes[2])
    
    axes[0][0].set_ylabel('MCAR')
    axes[1][0].set_ylabel('MAR')
    axes[2][0].set_ylabel('MNAR')

    for i, feature_name in enumerate(dm1.meta_info["feature_names"]):
        if feature_name == "OT":
            axes[0][i].set_title(f"{feature_name} (缺失特征)")
        else:
            axes[0][i].set_title(feature_name)

    plt.tight_layout()
    plt.show()
    
    
def plot_missing_relationship(dm: DataManager, ax: List[Axes]) -> None:
    missing_feature_name: str = dm.missing_mask.any(axis=0).index[dm.missing_mask.any(axis=0)][0]
    missing_feature_mask = dm.missing_mask[missing_feature_name]
    feature_values = dm.loaded_data[dm.meta_info["feature_names"]]

    for i, feature_col in enumerate(feature_values.columns):
        jitter = np.random.normal(0, 0.1, size=len(missing_feature_mask))
        
        ax[i].scatter(feature_values[feature_col][~missing_feature_mask], 
                     jitter[~missing_feature_mask], 
                     alpha=0.1, s=1, c='blue')
        ax[i].scatter(feature_values[feature_col][missing_feature_mask], 
                     jitter[missing_feature_mask] + 1, 
                     alpha=0.1, s=1, c='red')
        
        
        ax[i].set_ylim(-0.5, 1.5)
        if i == 0:
            ax[i].set_yticks([0, 1])
            ax[i].set_yticklabels(['存在', '缺失'])
        else:
            ax[i].set_yticks([])


def plot_pointbiserialr_comparison(dm1: DataManager, dm2: DataManager, dm3: DataManager) -> None:
    corr_matrix1 = calculate_pointbiserialr_corr(dm1)
    corr_matrix2 = calculate_pointbiserialr_corr(dm2)
    corr_matrix3 = calculate_pointbiserialr_corr(dm3)

    corr_matrix = pd.concat([corr_matrix1, corr_matrix2, corr_matrix3], keys=['MCAR', 'MAR', 'MNAR'])
    
    plt.figure(figsize=(10, 4))
    g = sns.heatmap(
        corr_matrix.values.astype(float),
        annot=True,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        fmt=".2f",
        cbar_kws={
            "label": "点二列相关系数"
        }
    )
    g.set_xticklabels(dm1.meta_info["feature_names"])
    g.set_yticklabels(["MCAR", "MAR", "MNAR"], rotation=0)
    g.set_xlabel("数值")
    g.set_ylabel("缺失掩码")
    
    plt.tight_layout()
    plt.show()


def calculate_pointbiserialr_corr(dm: DataManager) -> pd.DataFrame:
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

            corr: float = pointbiserialr(x, y)[0]  # type: ignore

            corr_matrix.loc[target_col, feature_col] = corr
    
    return corr_matrix


def plot_missing_mask(dm: DataManager, ax: Axes) -> None:

    missing_feature_names: List[str] = dm.missing_mask.any(axis=0).index[dm.missing_mask.any(axis=0)].tolist()
    missing_feature_mask = dm.missing_mask[missing_feature_names]

    ax.imshow(missing_feature_mask.T, aspect='auto', cmap='gray_r')
    ax.set_yticks(ticks=np.arange(len(missing_feature_names)), labels=missing_feature_names)


if __name__ == "__main__":
    plot_line_block_missing_mask()
    test_all_conditions()
    compare_missing_types()