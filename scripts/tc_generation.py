from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import json
import argparse
import os


# === 定义通行性计算函数 ===
def remove_gravity(acc_data):
    gravity = np.median(acc_data, axis=0)
    return acc_data - gravity

def compute_spatial_features(acc_data, gyro_data, position_data):
    """
    计算基于空间的特征

    参数:
    acc_data: 加速度数据 (N, 3)
    gyro_data: 陀螺仪数据 (N, 3)
    position_data: 位置数据 (N, 3)

    返回:
    rms_acc_spatial: 空间RMS加速度
    cumulative_angle_spatial: 空间累积角度
    rms_jerk_spatial: 空间RMS急动度
    """

    # 移除重力
    acc_corr = remove_gravity(acc_data)
    acc_mag = np.linalg.norm(acc_corr, axis=1)

    # 计算相邻点之间的空间距离
    position_diff = np.diff(position_data, axis=0)
    ds_array = np.linalg.norm(position_diff, axis=1)
    ds_array = np.where(ds_array == 0, 1e-6, ds_array)  # 避免除零

    # 计算空间导数 (加速度对空间的变化率)
    acc_spatial_deriv = np.diff(acc_corr, axis=0) / ds_array.reshape(-1, 1)

    # 计算角速度对空间的变化率
    gyro_spatial_deriv = np.diff(gyro_data, axis=0) / ds_array.reshape(-1, 1)
    gyro_mag = np.linalg.norm(gyro_data, axis=1)

    # 空间累积角度 (角度变化量/距离)
    cumulative_angle_spatial = np.sum(gyro_mag[:-1] * ds_array) / np.sum(ds_array)

    # 计算空间域特征
    # 1. 空间RMS加速度 (加速度在空间上的均方根)
    rms_acc_spatial = np.sqrt(np.mean(acc_mag ** 2))

    # 2. 空间RMS急动度 (加速度对空间的变化率的均方根)
    jerk_spatial_mag = np.linalg.norm(acc_spatial_deriv, axis=1)
    rms_jerk_spatial = np.sqrt(np.mean(jerk_spatial_mag ** 2))

    return rms_acc_spatial, cumulative_angle_spatial, rms_jerk_spatial


def compute_traversability_cost_spatial(acc_data, gyro_data, position_data,
                                        w1=1.0, w2=1.0, w3=1.0):
    """
    计算基于空间的通行性代价

    参数:
    acc_data: 加速度数据
    gyro_data: 陀螺仪数据
    position_data: 位置数据
    w1, w2, w3: 权重系数
    """
    rms_acc, cumulative_angle, rms_jerk = compute_spatial_features(
        acc_data, gyro_data, position_data)
    return w1 * rms_acc + w2 * cumulative_angle + w3 * rms_jerk


# === 归一化函数 ===
def normalize_values(values, method='minmax'):
    """
    对数值进行归一化，确保结果在[0,1]范围内

    参数:
    values: 需要归一化的数值数组
    method: 归一化方法
        - 'minmax': Min-Max归一化到[0,1]
        - 'zscore': Z-score标准化后映射到[0,1]
        - 'robust': 鲁棒归一化后映射到[0,1]
        - 'log_minmax': 对数变换后Min-Max归一化
    """
    values = np.array(values)
    
    # 检查是否有无效值
    if len(values) == 0 or np.all(np.isnan(values)):
        return values

    if method == 'minmax':
        # Min-Max归一化到[0,1]
        v_min, v_max = np.nanmin(values), np.nanmax(values)
        if v_max == v_min:
            return np.full_like(values, 0.5)  # 如果所有值相同，返回0.5
        normalized = (values - v_min) / (v_max - v_min)
        
    elif method == 'zscore':
        # Z-score标准化后映射到[0,1]
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        if std_val == 0:
            normalized = np.full_like(values, 0.5)
        else:
            z_scores = (values - mean_val) / std_val
            # 使用sigmoid函数将z-score映射到[0,1]
            normalized = 1 / (1 + np.exp(-z_scores))
            
    elif method == 'robust':
        # 鲁棒归一化后映射到[0,1]
        median = np.nanmedian(values)
        q75, q25 = np.nanpercentile(values, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            normalized = np.full_like(values, 0.5)
        else:
            robust_scores = (values - median) / iqr
            # 使用tanh函数将robust scores映射到[0,1]
            normalized = 0.5 * (np.tanh(robust_scores) + 1)
            
    elif method == 'log_minmax':
        # 对数变换后Min-Max归一化
        # 确保所有值为正
        min_val = np.nanmin(values)
        if min_val <= 0:
            values = values - min_val + 1e-6  # 平移使所有值为正
        log_values = np.log(values)
        v_min, v_max = np.nanmin(log_values), np.nanmax(log_values)
        if v_max == v_min:
            normalized = np.full_like(values, 0.5)
        else:
            normalized = (log_values - v_min) / (v_max - v_min)
    else:
        raise ValueError(f"不支持的归一化方法: {method}")
    
    # 确保结果严格在[0,1]范围内
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized

def bayesian_kernel_interpolation(x_vals, y_vals, tc_vals, grid_xi, grid_yi, radius=1.0):
    """
    使用贝叶斯核插值替代griddata
    """
    def sparse_kernel(d, r):
        """论文中的稀疏核函数"""
        if d > r:
            return 0
        return (2 + np.cos(2*np.pi*d/r))/3 * (1 - d/r) + np.sin(2*np.pi*d/r)/(2*np.pi)
    
    H, W = grid_xi.shape
    result = np.full((H, W), np.nan)
    
    for i in range(H):
        for j in range(W):
            target_x, target_y = grid_xi[i, j], grid_yi[i, j]
            
            # 计算到所有已知点的距离
            distances = np.sqrt((x_vals - target_x)**2 + (y_vals - target_y)**2)
            
            # 计算核权重
            weights = np.array([sparse_kernel(d, radius) for d in distances])
            
            # 只考虑权重大于0的点
            valid_mask = weights > 0
            if np.sum(valid_mask) > 0:
                weighted_sum = np.sum(weights[valid_mask] * tc_vals[valid_mask])
                weight_sum = np.sum(weights[valid_mask])
                result[i, j] = weighted_sum / weight_sum
    
    return result

def compute_tc_from_imu_data(
    imu_data: np.ndarray,
    pc_range: tuple,
    grid_size: float = 0.2,  # 最终输出分辨率
    init_grid_size: float = 0.2,  # 初始计算TC的粗分辨率
    min_points_per_cell: int = 5,
    normalization_method: str = 'minmax'
) -> tuple:
    """
    imu_data: shape=(N,10) [[time,x,y,z,acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z]]
    pc_range: 点云范围 (x0, x1, y0, y1)
    grid_size: 最终输出的分辨率（米）
    init_grid_size: 初始计算TC的分辨率（米）
    
    返回:
    tc_grid_interp: 插值后的TC地图
    valid_grid_count: 有效栅格数量（初始计算阶段）
    """
    assert imu_data.shape[1] == 10, "imu_data shape must be (N, 10)"
    x0, x1, y0, y1 = pc_range
    assert x1 > x0 and y1 > y0, "x1 and y1 must be greater than x0 and y0"

    positions = imu_data[:, 1:4]
    acc_data = imu_data[:, 4:7]
    gyro_data = imu_data[:, 7:]

    # === 初始空间栅格化 ===
    grid_x = np.floor(positions[:, 0] / init_grid_size).astype(int)
    grid_y = np.floor(positions[:, 1] / init_grid_size).astype(int)
    grid_keys = list(zip(grid_x, grid_y))

    grid_index_dict = defaultdict(list)
    for i, key in enumerate(grid_keys):
        grid_index_dict[key].append(i)

    # === 计算每个格子的TC ===
    tc_map = {}
    for key, idxs in grid_index_dict.items():
        if len(idxs) < min_points_per_cell:
            continue
        acc_block = acc_data[idxs]
        gyro_block = gyro_data[idxs]
        position_block = positions[idxs]
        
        # 添加异常检查
        if np.any(np.isnan(acc_block)) or np.any(np.isnan(gyro_block)) or np.any(np.isnan(position_block)):
            continue
            
        cost = compute_traversability_cost_spatial(acc_block, gyro_block, position_block)
        
        # 检查cost是否有效
        if np.isnan(cost) or np.isinf(cost):
            continue
            
        tc_map[key] = cost

    # 记录有效栅格数量
    valid_grid_count = len(tc_map)

    if not tc_map:
        # 全部无数据
        H = int((y1 - y0) / grid_size)
        W = int((x1 - x0) / grid_size)
        return np.full((H, W), np.nan), 0

    # === 归一化TC值 ===
    tc_vals = np.array(list(tc_map.values()))
    tc_vals_normalized = normalize_values(tc_vals, method=normalization_method)
    tc_map_normalized = {k: v for k, v in zip(tc_map.keys(), tc_vals_normalized)}

    # === 准备插值 ===
    x_keys, y_keys = zip(*tc_map_normalized.keys())
    x_vals = np.array(x_keys) * init_grid_size
    y_vals = np.array(y_keys) * init_grid_size
    tc_vals = np.array(list(tc_map_normalized.values()))

    # 输出网格坐标
    x_range = np.arange(x0, x1, grid_size)
    y_range = np.arange(y0, y1, grid_size)
    grid_xi, grid_yi = np.meshgrid(x_range, y_range)

    # 插值处理
    try:
        tc_grid_interp = bayesian_kernel_interpolation(
            x_vals=x_vals,
            y_vals=y_vals,
            tc_vals=tc_vals,
            grid_xi=grid_xi,
            grid_yi=grid_yi,
            radius=1.0
        )
        # tc_grid_interp = griddata(
        #     points=(x_vals, y_vals),
        #     values=tc_vals,
        #     xi=(grid_xi, grid_yi),
        #     method='cubic',
        #     fill_value=np.nan
        # )
    except Exception as e:
        print(f"Cubic interpolation failed: {e}, trying linear interpolation")
        try:
            tc_grid_interp = griddata(
                points=(x_vals, y_vals),
                values=tc_vals,
                xi=(grid_xi, grid_yi),
                method='linear',
                fill_value=np.nan
            )
        except Exception as e:
            print(f"Linear interpolation failed: {e}, trying nearest interpolation")
            tc_grid_interp = griddata(
                points=(x_vals, y_vals),
                values=tc_vals,
                xi=(grid_xi, grid_yi),
                method='nearest',
                fill_value=np.nan
            )
    
    # 严格确保结果在[0,1]范围内
    tc_grid_interp = np.clip(tc_grid_interp, 0.0, 1.0)

    return tc_grid_interp, valid_grid_count

def save_tc_map_as_npy(tc_map: np.ndarray, output_path: str):
    """
    将TC Map保存为numpy文件
    
    参数:
    tc_map: H×W ndarray，值在[0,1]范围内
    output_path: 输出文件路径（包含.npy后缀）
    """
    # 确保TC map的值严格在[0,1]范围内
    tc_map_clipped = np.clip(tc_map, 0.0, 1.0)
    
    # 保存为numpy文件
    np.save(output_path, tc_map_clipped)
    
    # 输出统计信息
    H, W = tc_map_clipped.shape
    valid_cells = ~np.isnan(tc_map_clipped)
    
    print(f"TC Map已保存至: {output_path}")
    print(f"数组形状: {H} x {W}")
    if np.any(valid_cells):
        print(f"有效网格数: {np.sum(valid_cells)} / {H * W}")
        print(f"TC值范围: [{np.nanmin(tc_map_clipped):.4f}, {np.nanmax(tc_map_clipped):.4f}]")
        print(f"TC平均值: {np.nanmean(tc_map_clipped):.4f}")
    else:
        print("警告: 所有网格都是NaN值")

def save_meta_json(pc_range: tuple, grid_size: float, valid_grid: int, output_path: str):
    """
    保存元数据到JSON文件
    
    参数:
    pc_range: 裁剪范围 (x0, x1, y0, y1)
    grid_size: 最终网格分辨率
    valid_grid: 有效栅格数量
    output_path: 输出文件路径（包含.json后缀）
    """
    meta_data = {
        "pc_range": list(pc_range),
        "grid_size": grid_size,
        "valid_grid": valid_grid
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(meta_data, f, indent=2)
        print(f"元数据已保存至: {output_path}")
    except Exception as e:
        print(f"错误: 保存元数据失败: {e}")

def save_tc_map_as_png(tc_map: np.ndarray, pc_range: tuple, grid_size: float, output_path: str):
    """
    将TC Map保存为PNG图片
    
    参数:
    tc_map: H×W ndarray
    pc_range: (x0, x1, y0, y1) 对应实际坐标范围
    grid_size: 每格的实际尺寸
    output_path: 输出文件路径（包含.png后缀）
    """
    x0, x1, y0, y1 = pc_range
    H, W = tc_map.shape
    extent = [x0, x1, y0, y1]

    # 创建图形，设置合适的尺寸和DPI
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制TC Map
    im = ax.imshow(tc_map, origin='lower', cmap='inferno',
                   extent=extent, aspect='equal', vmin=0, vmax=1)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label='Traversability Cost')
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # 设置标题和标签
    ax.set_title(f"2D TC Map (Grid={grid_size}m)", fontsize=14)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存为PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()  # 关闭图形以释放内存
    
    print(f"TC Map已保存至: {output_path}")

def visualize_tc_map(tc_map: np.ndarray, pc_range: tuple, grid_size: float):
    """
    可视化 TC Map (用于调试显示)
    """
    x0, x1, y0, y1 = pc_range
    H, W = tc_map.shape
    extent = [x0, x1, y0, y1]

    plt.figure(figsize=(8, 8))
    im = plt.imshow(tc_map, origin='lower', cmap='inferno',
                    extent=extent, aspect='equal', vmin=0, vmax=1)
    plt.colorbar(im, label='Traversability Cost')
    plt.title(f"2D TC Map (Grid={grid_size}m)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='计算并保存Traversability Cost Map')
    parser.add_argument('--data_file', type=str, required=True, 
                       help='IMU数据文件路径')
    parser.add_argument('--pc_range', type=float, nargs=4, 
                       help='点云范围: x0 x1 y0 y1')
    parser.add_argument('--grid_size', type=float, default=0.01,
                       help='输出网格分辨率（米）')
    parser.add_argument('--init_grid_size', type=float, default=0.2,
                       help='初始计算TC的网格分辨率（米）')
    parser.add_argument('--min_points', type=int, default=10,
                       help='每个网格最少点数要求')
    parser.add_argument('--normalization', type=str, default='minmax',
                       choices=['minmax', 'zscore', 'robust', 'log_minmax'],
                       help='归一化方法')
    parser.add_argument('--show_plot', action='store_true',
                       help='是否显示可视化图像')
    
    args = parser.parse_args()
    
    # 验证输入文件
    if not os.path.exists(args.data_file):
        print(f"错误: 数据文件不存在: {args.data_file}")
        return
    
    # === 读取数据 ===
    try:
        with open(args.data_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"错误: 读取数据文件失败: {e}")
        return

    # === 提取字段 ===
    imu_data = []
    for frame_id, info in data.items():
        try:
            imu_data.append([
                info['time'],
                info['x'], info['y'], info['z'],
                info['acc_x'], info['acc_y'], info['acc_z'],
                info['gyr_x'], info['gyr_y'], info['gyr_z']
            ])
        except KeyError as e:
            print(f"警告: 帧 {frame_id} 缺少字段 {e}")
            continue
    
    if not imu_data:
        print("错误: 没有有效的IMU数据")
        return
        
    imu_data = np.array(imu_data)
    print(f"加载了 {len(imu_data)} 个数据点")

    # === 计算TC ===
    print("正在计算Traversability Cost Map...")
    pc_range = tuple(args.pc_range)
    tc_map, valid_grid_count = compute_tc_from_imu_data(
        imu_data, 
        pc_range,
        grid_size=args.grid_size,
        init_grid_size=args.init_grid_size,
        min_points_per_cell=args.min_points,
        normalization_method=args.normalization
    )
    
    print(f"TC Map计算完成，形状: {tc_map.shape}")
    print(f"有效栅格数（初始阶段）: {valid_grid_count}")

    # === 确定输出目录 ===
    input_dir = os.path.dirname(args.data_file)
    
    # === 保存元数据JSON文件（新增功能）===
    meta_output_path = os.path.join(input_dir, "meta.json")
    save_meta_json(pc_range, args.grid_size, valid_grid_count, meta_output_path)

    # === 保存numpy数组文件（主要输出）===
    npy_output_path = os.path.join(input_dir, "label.npy")
    
    try:
        save_tc_map_as_npy(tc_map, npy_output_path)
    except Exception as e:
        print(f"错误: 保存numpy文件失败: {e}")

    # === 保存PNG可视化图片 ===
    png_output_path = os.path.join(input_dir, "costmap.png")
    try:
        save_tc_map_as_png(tc_map, pc_range, args.grid_size, png_output_path)
    except Exception as e:
        print(f"错误: 保存PNG文件失败: {e}")

    # === 可选显示可视化图像 ===
    if args.show_plot:
        try:
            visualize_tc_map(tc_map, pc_range, args.grid_size)
        except Exception as e:
            print(f"警告: 显示图像失败: {e}")


if __name__ == '__main__':
    main()