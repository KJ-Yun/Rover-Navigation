import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import random
from pathlib import Path
from tqdm import tqdm

from TCPredictionNet import TCPredictionNet
from dataset import PCImageDataset, collate_fn

# 设置随机种子确保可复现性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_heatmap(data, title, save_path, cmap='viridis'):
    """
    创建纯热力图并保存（无坐标轴、标题等装饰）
    
    Args:
        data: 2D numpy array 或 torch tensor
        title: 图片标题（仅用于函数参数一致性，实际不显示）
        save_path: 保存路径
        cmap: 颜色映射
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    # 如果是3D数据（1, H, W），取第一个通道
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    
    # 处理NaN值用于可视化，将NaN设为0用于显示
    # data_vis = np.where(np.isnan(data), 0, data)
    data_vis = data.copy()
    
    # 获取数据的高度和宽度
    height, width = data_vis.shape
    
    # 创建图形，设置尺寸与数据尺寸成比例
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    
    # 使用指定的colormap，并设置NaN颜色为白色
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='white')  # NaN 单独显示成白色
    
    # 显示图像，关闭所有装饰
    ax.imshow(data_vis, cmap=cmap_obj, aspect='equal', interpolation='nearest')
    
    # 移除所有装饰元素
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')
    
    # 移除边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 保存纯图片，无边距
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()


def calculate_metrics_ignore_nan(y_true, y_pred):
    """
    计算忽略NaN值的MSE和MAE
    
    Args:
        y_true: 真实值 (torch.Tensor)
        y_pred: 预测值 (torch.Tensor)
    
    Returns:
        mse, mae
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 展平数组
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # 找到非NaN的mask
    valid_mask = ~np.isnan(y_true_flat)
    
    if not np.any(valid_mask):
        return float('nan'), float('nan')
    
    # 只计算非NaN位置的指标
    y_true_valid = y_true_flat[valid_mask]
    y_pred_valid = y_pred_flat[valid_mask]
    
    # 手动计算MSE和MAE
    mse = np.mean((y_true_valid - y_pred_valid) ** 2)
    mae = np.mean(np.abs(y_true_valid - y_pred_valid))
    
    return mse, mae

def inference(model, dataset, device, output_dir, sample_ratio=0.2):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_size = len(dataset)
    sample_size = int(dataset_size * sample_ratio)
    indices = random.sample(range(dataset_size), sample_size)
    sampled_dataset = Subset(dataset, indices)

    dataloader = DataLoader(sampled_dataset, batch_size=1, shuffle=False, 
                           num_workers=0, collate_fn=collate_fn)

    model.eval()
    model.to(device)

    all_results = []  # 保存每个样本的结果 (idx, mse, mae, label, pred)

    print(f"开始推理，共处理 {len(sampled_dataset)} 个样本...")
    with torch.no_grad():
        for idx, (pcs, imgs, costs) in enumerate(tqdm(dataloader, desc="推理中")):
            pcs = [pc.to(device) for pc in pcs]
            imgs = imgs.to(device)
            costs = costs.to(device)

            predictions = model(pcs, imgs)
            prediction = predictions[0]
            label = costs[0]

            mse, mae = calculate_metrics_ignore_nan(label, prediction)

            if not np.isnan(mse):
                all_results.append((idx, mse, mae, label, prediction))

            label_path = output_dir / f"{idx}_label.png"
            pred_path = output_dir / f"{idx}_pred.png"
            create_heatmap(label, f"Label - Sample {idx}", label_path, cmap='viridis')
            create_heatmap(prediction, f"Prediction - Sample {idx}", pred_path, cmap='viridis')

    # ===========================
    # 选出 MSE/MAE 最小的前 5 个
    # ===========================
    top_mse = sorted(all_results, key=lambda x: x[1])[:5]
    top_mae = sorted(all_results, key=lambda x: x[2])[:5]

    # 创建保存目录
    top_mse_dir = output_dir / "top_mse"
    top_mae_dir = output_dir / "top_mae"
    top_mse_dir.mkdir(exist_ok=True)
    top_mae_dir.mkdir(exist_ok=True)

    for rank, (idx, mse, mae, label, pred) in enumerate(top_mse, 1):
        create_heatmap(label, f"Label-MSE{mse:.4f}", top_mse_dir / f"rank{rank}_{idx}_label.png")
        create_heatmap(pred, f"Pred-MSE{mse:.4f}", top_mse_dir / f"rank{rank}_{idx}_pred.png")

    for rank, (idx, mse, mae, label, pred) in enumerate(top_mae, 1):
        create_heatmap(label, f"Label-MAE{mae:.4f}", top_mae_dir / f"rank{rank}_{idx}_label.png")
        create_heatmap(pred, f"Pred-MAE{mae:.4f}", top_mae_dir / f"rank{rank}_{idx}_pred.png")

    # 平均指标
    if all_results:
        avg_mse = np.mean([r[1] for r in all_results])
        avg_mae = np.mean([r[2] for r in all_results])
    else:
        avg_mse = float("nan")
        avg_mae = float("nan")

    print(f"推理完成！平均 MSE={avg_mse:.6f}, 平均 MAE={avg_mae:.6f}")
    print(f"Top-5 已保存到 {top_mse_dir} 和 {top_mae_dir}")

    return avg_mse, avg_mae


def load_model(checkpoint_path, device):
    """
    加载模型
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
    
    Returns:
        model: 加载的模型
    """
    # 创建模型实例（参数需要与训练时一致）
    model = TCPredictionNet(
        pc_range=(0, 0, -5, 1, 1, 5),
        bev_H=100, bev_W=100, bev_channels=8,
        max_points_per_pillar=100,
        use_relative_xyz=True, use_rgb=True,
        fpn_out_channels=128,
        use_modulation=True, modulation_dim=384,
        dinov3_repo="model/dinov3",
        dinov3_weight="model/dinov3/weight/weight.pth"
    )
    
    # 加载检查点
    if os.path.exists(checkpoint_path):
        print(f"加载模型检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 处理不同的检查点格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("模型加载成功!")
    else:
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    return model

def main():
    # 设置参数
    set_seed(42)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 路径设置
    checkpoint_path = "checkpoints/best_model.pth"  # 检查点路径
    output_dir = "output"  # 输出目录
    data_dir = "data"  # 数据目录
    
    # 加载模型
    model = load_model(checkpoint_path, device)
    model.to(device)
    
    # 创建数据集（参数需要与训练时一致）
    dataset = PCImageDataset(
        data_dir,
        patch_size=1.0, 
        offset=-1.5,
        img_size=1024, 
        min_points=100,
        img_subdir="Colmap/images",
        sample_step=0.1,
        prefilter_samples=True
    )
    
    print(f"数据集创建成功，共 {len(dataset)} 个样本")
    
    # 执行推理
    avg_mse, avg_mae = inference(model, dataset, device, output_dir, sample_ratio=0.2)
    
    # 保存结果到文件
    results_file = Path(output_dir) / "inference_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"推理结果\n")
        f.write(f"=" * 30 + "\n")
        f.write(f"模型检查点: {checkpoint_path}\n")
        f.write(f"数据目录: {data_dir}\n")
        f.write(f"采样比例: 20%\n")
        f.write(f"平均 MSE: {avg_mse:.6f}\n")
        f.write(f"平均 MAE: {avg_mae:.6f}\n")
    
    print(f"结果已保存到: {results_file}")

if __name__ == "__main__":
    main()