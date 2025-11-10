import numpy as np
from typing import Optional, Dict, List, Tuple
from scipy import stats
import warnings


class sampler:
    def __init__(
        self,
        k_min: int = 30,
        k_max: int = 10000,
        random_state: int = 42,
        alpha: float = 0.05,
        delta: float = 0.05,
        additional_samples: int = 0
    ) -> None:
        self.rng = np.random.RandomState(random_state)
        self.k_min: int = k_min
        self.k_max: int = k_max
        self.alpha: float = alpha
        self.delta: float = delta
        self.additional_samples: int = additional_samples
        
        self.k: int = 0
        self.metrics_dict: Dict[str, List[int | float]] = {}
        self.if_converged: bool = False
        self.converged_k: int
        
    def __call__(
        self
    ) -> Optional[int]:
        if self.if_converged and self.k >= self.converged_k + self.additional_samples:
            return None
        
        if self.k >= self.k_max:
            warnings.warn("Maximum number of iterations reached without convergence.")
            return None
        
        self.k += 1
        return self.rng.randint(0, 2**32)
    
    def update_metrics(
        self,
        metrics: Dict[str, int | float]
    ) -> None:
        if self.k == 1:
            for key in metrics.keys():
                self.metrics_dict[key] = []
                
        if not self.check_dict_keys(metrics):
            raise ValueError("The keys of the input metrics dictionary do not match the existing keys.")
        
        for key, value in metrics.items():
            self.metrics_dict[key].append(value)
        
        if not self.if_converged:
            self.check_convergence()
            if self.if_converged:
                print(f"Convergence achieved at iteration {self.k}.")
        
    def check_dict_keys(
        self,
        metrics: Dict[str, int | float]
    ) -> bool:
        if set(metrics.keys()) != set(self.metrics_dict.keys()):
            return False
        return True
    
    def check_convergence(
        self
    ) -> bool:
        if self.k < self.k_min:
            return False
        
        for key, values in self.metrics_dict.items():
            mean = np.mean(values)
            if mean == 0:
                warnings.warn("Mean of metric values is zero, cannot compute relative width.")
                return False
            std = np.std(values, ddof=1)
            z_score = stats.norm.ppf(1 - self.alpha/2)
            confidence_width = 2 * z_score * (std / np.sqrt(self.k))
            relative_width = confidence_width / abs(mean)
            
            if relative_width > self.delta:
                return False
        
        self.if_converged = True
        self.converged_k = self.k
        return True
    
    def get_metrics(
        self
    ) -> Dict[str, Tuple[float, float]]:
        if not self.if_converged and self.k < self.k_max:
            raise RuntimeError("Metrics are not available until convergence is achieved or maximum iterations reached.")
        
        final_metrics: Dict[str, Tuple[float, float]] = {}
        for key, values in self.metrics_dict.items():
            final_metrics[key] = (float(np.mean(values)), float(np.std(values, ddof=1)))
        
        return final_metrics
    
    def plot_convergence(
        self
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        
        if self.k < self.k_min:
            raise RuntimeError("Not enough iterations to plot convergence.")
        
        for key, values in self.metrics_dict.items():
            values_array = np.array(values)
            iterations = np.arange(1, self.k + 1)
            
            cumulative_mean = np.array([np.mean(values_array[:i]) for i in range(1, len(values_array) + 1)])
            cumulative_std = np.array([np.std(values_array[:i], ddof=1) if i > 1 else 0 for i in range(1, len(values_array) + 1)])
            z_score = stats.norm.ppf(1 - self.alpha/2)
            confidence_interval = z_score * cumulative_std / np.sqrt(iterations)
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()
            
            mean_line = ax1.plot(iterations, cumulative_mean, 'b-', 
                               label='累计均值', linewidth=2, zorder=3)
            ci_fill = ax1.fill_between(iterations, 
                                     cumulative_mean - confidence_interval,
                                     cumulative_mean + confidence_interval,
                                     alpha=0.2, color='b', label=f'{int((1-self.alpha)*100)}%置信区间', zorder=1)
            scatter = ax1.scatter(iterations, values_array, c='gray', 
                                alpha=0.3, s=20, label='采样数据', zorder=2)
            
            if self.if_converged:
                ax1.axvline(x=self.converged_k, color='g', linestyle='--', label=f'收敛点')
            
            ax1.set_xlabel('采样次数')
            ax1.set_ylabel('均值', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # 在右侧y轴绘制标准差
            std_line = ax2.plot(iterations, cumulative_std, 'r--', 
                              label='累计标准差', linewidth=1.5)
            ax2.set_ylabel('标准差', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # 添加标题和网格
            plt.title(f'{key} 收敛过程')
            ax1.grid(True, alpha=0.3)
            
            # 合并图例
            lines = mean_line + std_line
            if self.if_converged:
                convergence_line = Line2D([0], [0], color='g', linestyle='--', label=f'收敛点')
                lines.append(convergence_line)
            labels = [str(l.get_label()) for l in lines]
            ax1.legend(lines + [Rectangle((0,0),1,1, fc='b', alpha=0.2)], 
                      labels + [f'{int((1-self.alpha)*100)}%置信区间'],
                      loc='lower right')
            
            plt.tight_layout()
            plt.show()
            

if __name__ == "__main__":
    from src.data_manager import DataManager
    from src.missing_generator.generator import generate_missing_data
    from src.models.classical_models.mean_impute import MeanImputeModel
    from src.evaluator.sample_error import RMSE, MAE
    
    dm = DataManager()
    
    datasets = dm.scan_datasets()
    if not datasets:
        raise RuntimeError("No datasets available for testing.")
    
    dm.load_dataset(datasets[0])
    for random_state in [42]:
        sampler_instance = sampler(k_min=50, k_max=10000, random_state=random_state, alpha=0.05, delta=0.001, additional_samples=100)
        random_seed = sampler_instance()
        while random_seed is not None:
            generate_missing_data(dm, target_column=["OT"], missing_rate=0.5, missing_type="MCAR", random_state=random_seed)
            dm.imputed_data = MeanImputeModel()(dm.missing_data)
            metrics = {"RMSE": RMSE(dm.loaded_data["OT"], dm.imputed_data["OT"]), "MAE": MAE(dm.loaded_data["OT"], dm.imputed_data["OT"])}
            sampler_instance.update_metrics(metrics)
            random_seed = sampler_instance()
        
        sampler_instance.plot_convergence()
        print(sampler_instance.get_metrics())