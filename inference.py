import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from PaiNN import PainnModel
import warnings

warnings.filterwarnings('ignore')

# 设置绘图样式 - 使用标准matplotlib样式
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def load_painn_model(checkpoint_path: str, device: torch.device = None) -> PainnModel:
    """
    加载PaiNN模型
    
    Args:
        checkpoint_path: 检查点文件路径
        device: 设备 (CPU/GPU)
    
    Returns:
        加载好的PaiNN模型
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = PainnModel(
        num_interactions=6,
        hidden_state_size=128,
        cutoff=6.0,
        pdb=True
    )
    
    # 加载状态字典
    state_dict = torch.load(checkpoint_path, map_location=device)["state_dict"]
    
    # 移除'potential.'前缀（如果存在）
    new_state_dict = {k.replace("potential.", ""): v for k, v in state_dict.items()}
    
    # 加载权重
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model


def calculate_forces_mae(model: PainnModel, dataloader: DataLoader, 
                        device: torch.device = None) -> list:
    """
    计算模型在数据集上的力MAE
    
    Args:
        model: PaiNN模型
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        每个样本的MAE列表
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mae_list = []
    model.to(device)
    

    for data in tqdm(dataloader, desc="Calculating forces MAE"):
        # 移动到设备
        data = data.to(device)
        data.pbc = torch.as_tensor([True, True, True], device=device)
        
        # 获取真实力
        true_forces = data.force
        
        # 模型预测
        _, pred_forces = model(data)
        
        # 计算MAE (L1 norm)
        mae = torch.mean((pred_forces - true_forces).norm(dim=-1))
        mae_list.append(mae.item())
    
    return mae_list


def plot_mae_comparison(baseline_mae: list, selftraining_mae: list, 
                        save_dir: str = './') -> None:
    """
    绘制MAE对比图
    
    Args:
        baseline_mae: 基线模型的MAE列表
        selftraining_mae: 自训练模型的MAE列表
        save_dir: 保存图像的目录
    """
    # 计算平均MAE
    baseline_mean_mae = np.mean(baseline_mae)
    selftraining_mean_mae = np.mean(selftraining_mae)
    improvement = baseline_mean_mae - selftraining_mean_mae
    improvement_percent = (improvement / baseline_mean_mae * 100) if baseline_mean_mae > 0 else 0
    
    print("\n" + "="*60)
    print("MAE COMPARISON RESULTS")
    print("="*60)
    print(f"Baseline Mean MAE:     {baseline_mean_mae:.4f} eV/Å")
    print(f"Self-training Mean MAE: {selftraining_mean_mae:.4f} eV/Å")
    print(f"Improvement:           {improvement:.4f} eV/Å ({improvement_percent:.1f}%)")
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 散点图：基线 vs 自训练
    scatter = axes[0].scatter(baseline_mae, selftraining_mae, alpha=0.7, s=30, 
                             c=np.arange(len(baseline_mae)), cmap='viridis')
    max_val = max(np.max(baseline_mae), np.max(selftraining_mae)) * 1.05
    axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y = x (equal performance)')
    axes[0].fill_between([0, max_val], [0, max_val * 0.9], [0, max_val * 1.1], 
                        alpha=0.1, color='gray', label='±10% region')
    axes[0].set_xlabel('Baseline Force MAE (eV/Å)', fontweight='bold')
    axes[0].set_ylabel('Self-training Force MAE (eV/Å)', fontweight='bold')
    axes[0].set_title('Scatter Comparison of Force MAEs', fontweight='bold', pad=15)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label('Sample Index', rotation=270, labelpad=15)
    
    # 2. 直方图对比
    bins = np.linspace(0, max_val, 25)
    axes[1].hist(baseline_mae, bins=bins, alpha=0.6, label=f'Baseline\nMean: {baseline_mean_mae:.3f}',
                color='blue', edgecolor='black', linewidth=0.5)
    axes[1].hist(selftraining_mae, bins=bins, alpha=0.6, label=f'Self-training\nMean: {selftraining_mean_mae:.3f}',
                color='orange', edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Force MAE (eV/Å)', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].set_title('Distribution of Force MAEs', fontweight='bold', pad=15)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 3. 箱形图
    mae_data = [baseline_mae, selftraining_mae]
    bp = axes[2].boxplot(mae_data, labels=['Baseline', 'Self-training'], 
                        patch_artist=True, widths=0.6)
    
    # 设置箱形图颜色和样式
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
        patch.set_alpha(0.7)
    
    # 设置中位数线样式
    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2)
    
    axes[2].set_ylabel('Force MAE (eV/Å)', fontweight='bold')
    axes[2].set_title('Statistical Comparison', fontweight='bold', pad=15)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # 在箱形图上添加均值标记
    for i, (data, mean_val) in enumerate(zip(mae_data, [baseline_mean_mae, selftraining_mean_mae]), 1):
        axes[2].plot(i, mean_val, 'k*', markersize=10, label='Mean' if i == 1 else "")
    
    axes[2].legend(['Mean'], loc='upper right')
    
    # 调整布局
    plt.suptitle(f'Force Prediction Performance Comparison\n'
                f'Improvement: {improvement:.3f} eV/Å ({improvement_percent:.1f}%)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图像
    save_path = f"{save_dir}force_mae_comparison_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✓ Comparison plot saved to: {save_path}")
    
    plt.show()
    
    # 额外保存单独的散点图（高质量版本）
    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(baseline_mae, selftraining_mae, alpha=0.8, s=40, 
                         c=np.arange(len(baseline_mae)), cmap='plasma')
    max_val = max(np.max(baseline_mae), np.max(selftraining_mae)) * 1.05
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2.5, label='y = x')
    
    # 添加文本注释
    plt.text(max_val*0.05, max_val*0.95, 
             f'Baseline mean: {baseline_mean_mae:.3f} eV/Å\n'
             f'Self-training mean: {selftraining_mean_mae:.3f} eV/Å\n'
             f'Improvement: {improvement:.3f} eV/Å\n'
             f'({improvement_percent:.1f}%)',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, verticalalignment='top')
    
    plt.xlabel('Baseline Force MAE (eV/Å)', fontweight='bold', fontsize=13)
    plt.ylabel('Self-training Force MAE (eV/Å)', fontweight='bold', fontsize=13)
    plt.title('Force MAE Scatter Plot', fontweight='bold', fontsize=15, pad=15)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sample Index', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}force_mae_scatter_detailed.png", dpi=300)
    print(f"✓ Detailed scatter plot saved to: {save_dir}force_mae_scatter_detailed.png")
    
    # 保存性能改进图
    plt.figure(figsize=(10, 6))
    indices = np.arange(len(baseline_mae))
    width = 0.35
    
    plt.bar(indices - width/2, baseline_mae, width, label='Baseline', alpha=0.7, color='skyblue')
    plt.bar(indices + width/2, selftraining_mae, width, label='Self-training', alpha=0.7, color='lightcoral')
    
    plt.xlabel('Sample Index', fontweight='bold')
    plt.ylabel('Force MAE (eV/Å)', fontweight='bold')
    plt.title('Force MAE Comparison for Each Sample', fontweight='bold', pad=15)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{save_dir}force_mae_bar_comparison.png", dpi=300)
    print(f"✓ Bar comparison plot saved to: {save_dir}force_mae_bar_comparison.png")
    
    # 打印更多统计信息
    print("\n" + "="*60)
    print("DETAILED STATISTICS")
    print("="*60)
    
    for name, data in [("Baseline", baseline_mae), ("Self-training", selftraining_mae)]:
        print(f"\n{name}:")
        print(f"  Mean:    {np.mean(data):.4f} eV/Å")
        print(f"  Std:     {np.std(data):.4f} eV/Å")
        print(f"  Min:     {np.min(data):.4f} eV/Å")
        print(f"  25%:     {np.percentile(data, 25):.4f} eV/Å")
        print(f"  Median:  {np.median(data):.4f} eV/Å")
        print(f"  75%:     {np.percentile(data, 75):.4f} eV/Å")
        print(f"  Max:     {np.max(data):.4f} eV/Å")


def main():
    """主函数"""
    print("="*60)
    print("FORCE MAE COMPARISON: Baseline vs Self-training Models")
    print("="*60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载模型
    print("\n" + "-"*40)
    print("Loading models...")
    print("-"*40)
    
    try:
        baseline_model = load_painn_model('Baseline.ckpt', device)
        print("✓ Baseline model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading Baseline model: {e}")
        return
    
    try:
        selftraining_model = load_painn_model('Selftraining2.ckpt', device)
        print("✓ Self-training model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading Self-training model: {e}")
        return
    
    # 加载数据
    print("\n" + "-"*40)
    print("Loading dataset...")
    print("-"*40)
    
    try:
        finetune_dataset = torch.load('test.pt', weights_only=False)
        finetune_loader = DataLoader(finetune_dataset, batch_size=1, shuffle=False)
        print(f"✓ Dataset loaded successfully")
        print(f"  Number of samples: {len(finetune_dataset)}")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # 计算MAE
    print("\n" + "-"*40)
    print("Calculating MAE values...")
    print("-"*40)
    
    try:
        print("\nCalculating MAE for Baseline model...")
        baseline_mae_list = calculate_forces_mae(baseline_model, finetune_loader, device)
        
        print("\nCalculating MAE for Self-training model...")
        selftraining_mae_list = calculate_forces_mae(selftraining_model, finetune_loader, device)
    except Exception as e:
        print(f"✗ Error during MAE calculation: {e}")
        return
    
    # 绘制对比图
    print("\n" + "-"*40)
    print("Generating visualizations...")
    print("-"*40)
    
    plot_mae_comparison(baseline_mae_list, selftraining_mae_list, save_dir='./')
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()