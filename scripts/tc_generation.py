from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import json
import argparse


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
    对数值进行归一化

    参数:
    values: 需要归一化的数值数组
    method: 归一化方法
        - 'minmax': Min-Max归一化到[0,1]
        - 'zscore': Z-score标准化 (均值0，标准差1)
        - 'robust': 鲁棒归一化 (使用中位数和四分位距)
        - 'log_minmax': 对数变换后Min-Max归一化
    """
    values = np.array(values)

    if method == 'minmax':
        # Min-Max归一化到[0,1]
        v_min, v_max = values.min(), values.max()
        if v_max == v_min:
            return np.ones_like(values) * 0.5  # 如果所有值相同，返回0.5
        return (values - v_min) / (v_max - v_min)

    elif method == 'zscore':
        # Z-score标准化
        return (values - values.mean()) / values.std()

    elif method == 'robust':
        # 鲁棒归一化 (使用中位数和IQR)
        median = np.median(values)
        q75, q25 = np.percentile(values, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return np.zeros_like(values)
        return (values - median) / iqr

    elif method == 'log_minmax':
        # 对数变换后Min-Max归一化 (适用于有很大值的情况)
        log_values = np.log1p(values)  # log1p(x) = log(1+x)，避免log(0)
        v_min, v_max = log_values.min(), log_values.max()
        if v_max == v_min:
            return np.ones_like(values) * 0.5
        return (log_values - v_min) / (v_max - v_min)

    else:
        raise ValueError(f"不支持的归一化方法: {method}")

def compute_tc_from_imu_data(
    imu_data: np.ndarray,
    x0: float, x1: float, y0: float, y1: float,
    grid_size: float = 0.2,  # 最终输出分辨率
    init_grid_size: float = 0.2,  # 初始计算TC的粗分辨率
    min_points_per_cell: int = 5,
    normalization_method: str = 'minmax'
) -> np.ndarray:
    """
    imu_data: shape=(N,10) [[time,x,y,z,acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z]]
    (x0,x1,y0,y1): 输出范围（米）
    grid_size: 最终输出的分辨率（米）
    init_grid_size: 初始计算TC的分辨率（米）
    """
    assert imu_data.shape[1] == 10, "imu_data shape must be (N, 10)"
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
        cost = compute_traversability_cost_spatial(acc_block, gyro_block, position_block)
        tc_map[key] = cost

    if not tc_map:
        # 全部无数据
        H = int((y1 - y0) / grid_size)
        W = int((x1 - x0) / grid_size)
        return np.full((H, W), np.nan)

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
    tc_grid_interp = griddata(
        points=(x_vals, y_vals),
        values=tc_vals,
        xi=(grid_xi, grid_yi),
        method='cubic',
        fill_value=np.nan
    )

    return tc_grid_interp

def visualize_tc_map(tc_map: np.ndarray, x0: float, x1: float, y0: float, y1: float, grid_size: float):
    """
    可视化 TC Map
    tc_map: H×W ndarray（包含 NaN）
    (x0, x1, y0, y1): 对应实际坐标范围
    grid_size: 每格的实际尺寸
    """
    H, W = tc_map.shape
    extent = [x0, x1, y0, y1]  # 用于imshow的坐标映射

    plt.figure(figsize=(6, 6))
    im = plt.imshow(tc_map, origin='lower', cmap='inferno',
                    extent=extent, aspect='auto')
    plt.colorbar(im, label='Traversability Cost')
    plt.title(f"2D TC Map (Grid={grid_size}m)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(False)
    plt.tight_layout()
    # plt.savefig('./tc_map_interpolated.jpg', dpi=300)
    plt.show()



def main():
    pharse = argparse.ArgumentParser()
    pharse.add_argument('--data_file', type=str, default='./data/imu_data.json', help='IMU数据文件路径')
    args = pharse.parse_args()
    
    # === 读取数据 ===
    with open (args.data_file, 'r') as f:
        data = json.load(f)  # 路径根据实际情况调整

    # === 提取字段 ===
    imu_data = []
    for frame_id, info in data.items():
        imu_data.append([
            info['time'],
            info['x'], info['y'], info['z'],
            info['acc_x'], info['acc_y'], info['acc_z'],
            info['gyr_x'], info['gyr_y'], info['gyr_z']
        ])
    imu_data = np.array(imu_data)

    # === 计算TC ===
    tc_map_range = (0, 2, -4, -2)
    grid_size = 0.01
    tc_map = compute_tc_from_imu_data(imu_data, *tc_map_range, grid_size)
    print(tc_map.shape)

    # === 可视化 ===
    visualize_tc_map(tc_map, *tc_map_range, grid_size)


if __name__ == '__main__':
    main()
