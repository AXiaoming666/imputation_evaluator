import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Optional, List

class BaseImputationModel(ABC):
    def __init__(self, 
                 feature_names: Optional[List[str]] = None,
                 time_col: Optional[str] = None,
                 **kwargs):
        """
        初始化插补模型
        
        :param feature_names: 特征列名称列表
        :param time_col: 时间戳列名称（如需利用时间信息）
        :param **kwargs: 模型特有参数
        """
        self.feature_names = feature_names
        self.time_col = time_col
        self.model = None  # 存储实际模型对象
        self.is_fitted = False  # 标记模型是否已训练
        
    @abstractmethod
    def fit(self, 
            X: Union[pd.DataFrame, np.ndarray], 
            y: Optional[Union[pd.DataFrame, np.ndarray]] = None) -> "BaseImputationModel":
        """
        训练模型（适用于需要训练的模型）
        
        :param X: 输入数据（可能包含缺失值），形状为(n_samples, n_features)
        :param y: 目标值（通常无需传入，因插补任务中输入即目标）
        :return: 模型自身（便于链式调用）
        """
        # 实现说明：
        # 1. 需将输入数据转换为模型可处理的格式
        # 2. 训练完成后需将self.is_fitted设为True
        # 3. 对于无监督模型或无需训练的模型，可空实现但需标记is_fitted=True
        raise NotImplementedError("子类必须实现fit方法")
    
    @abstractmethod
    def impute(self, 
               X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        对含缺失值的数据进行插补
        
        :param X: 含缺失值的输入数据，形状为(n_samples, n_features)
        :return: 插补后的完整数据（与输入格式一致）
        """
        # 实现说明：
        # 1. 需检查模型是否已训练（is_fitted），未训练时应抛出异常
        # 2. 保持输出格式与输入一致（DataFrame/ndarray）
        # 3. 仅填充缺失值位置，不改变已有观测值
        raise NotImplementedError("子类必须实现impute方法")
    
    def __call__(self, 
                 X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        使模型实例可直接调用进行插补（简化接口）
        
        :param X: 含缺失值的输入数据
        :return: 插补后的完整数据
        """
        return self.impute(X)
    
    def _check_fitted(self) -> None:
        """检查模型是否已训练，未训练则抛出异常"""
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用fit方法")
    
    def _convert_to_numpy(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """将输入数据转换为numpy数组（内部处理用）"""
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, np.ndarray):
            return X.copy()
        else:
            raise TypeError(f"不支持的数据类型: {type(X)}，需为DataFrame或ndarray")
    
    def _convert_to_dataframe(self, 
                             X: np.ndarray, 
                             original: pd.DataFrame) -> pd.DataFrame:
        """将numpy数组转换回DataFrame（保持原始索引和列名）"""
        return pd.DataFrame(
            X,
            index=original.index,
            columns=original.columns
        )
    
    def get_params(self) -> dict:
        """获取模型参数（用于日志和报告）"""
        return {
            "model_type": self.__class__.__name__,
            "feature_names": self.feature_names,
            "time_col": self.time_col,
            "is_fitted": self.is_fitted
        }
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """保存模型到文件"""
        raise NotImplementedError("子类必须实现save_model方法")
    
    @abstractmethod
    def load_model(self, path: str) -> "BaseImputationModel":
        """从文件加载模型"""
        raise NotImplementedError("子类必须实现load_model方法")


class DummyImputer(BaseImputationModel):
    """
    基准插补器（用于测试接口和作为性能基线）
    使用均值填充数值型特征的缺失值
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None, **kwargs):
        super().__init__(feature_names=feature_names,** kwargs)
        self.means = None  # 存储各特征的均值
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> "DummyImputer":
        """计算各特征的均值作为插补值"""
        X_np = self._convert_to_numpy(X)
        # 计算每个特征的均值（忽略NaN）
        self.means = np.nanmean(X_np, axis=0)
        self.is_fitted = True
        return self
    
    def impute(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """用训练阶段计算的均值填充缺失值"""
        self._check_fitted()
        is_dataframe = isinstance(X, pd.DataFrame)
        original = X if is_dataframe else None
        X_np = self._convert_to_numpy(X)
        
        # 复制输入数据避免修改原数组
        X_imputed = X_np.copy()
        
        # 填充缺失值
        for i in range(X_np.shape[1]):
            mask = np.isnan(X_np[:, i])
            X_imputed[mask, i] = self.means[i]
        
        # 转换回原始格式
        return self._convert_to_dataframe(X_imputed, original) if is_dataframe else X_imputed
    
    def save_model(self, path: str) -> None:
        """保存均值参数"""
        np.savez(path, means=self.means, feature_names=self.feature_names)
    
    def load_model(self, path: str) -> "DummyImputer":
        """加载均值参数"""
        data = np.load(path)
        self.means = data["means"]
        self.feature_names = data["feature_names"].tolist()
        self.is_fitted = True
        return self
    