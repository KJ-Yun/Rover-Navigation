import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import json


# === 加载点云（N,6） ===
def load_point_cloud_npy(file_path):
    data = np.load(file_path)
    print(data.shape)
    print(data[:5])
    points = data[:, :3]
    colors = data[:, 3:]
    return points, colors


# === 加载轨迹位姿 ===
def load_poses(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    positions = []
    for frame_id, info in data.items():
        roll, pitch, yaw = info['roll'], info['pitch'], info['yaw']
        x, y, z = info['x'], info['y'], info['z']
        positions.append([x, y, z])
    return np.array(positions)


# === 绘制俯视图（x-y 平面） ===
def draw_topdown(points, colors, trajectory, pc_range, save_path="topdown_view.png", use_pc_range=True):
    if use_pc_range:
        x_min, x_max, y_min, y_max = pc_range

        mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & \
               (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        points = points[mask]
        colors = colors[mask]
    else:
        x_min = points[:, 0].min()
        x_max = points[:, 0].max()
        y_min = points[:, 1].min()
        y_max = points[:, 1].max()

    fig, ax = plt.subplots(figsize=(10, 10))

    # 点云 x-y 显示
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=0.5, alpha=0.8)

    # 绘制轨迹线
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=1, label='Trajectory')

    # 绘制起点红点
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'ro', markersize=8, label='Start Point')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Top-Down View of Colored Point Cloud with Trajectory')
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 图像已保存：{save_path}")


# === 主程序 ===
if __name__ == '__main__':
    pc_range = [-0.5, 3, -7, 0.5]
    use_pc_range = True  # 是否使用预设的pc_range裁剪点云显示范围
    root_dir = "/home/jet/Desktop/proj/data/15/"
    points, colors = load_point_cloud_npy(root_dir + "pointcloud.npy")
    trajectory = load_poses(root_dir + "data.json")
    draw_topdown(points, colors, trajectory, pc_range, save_path=root_dir + "topdown_view.png", use_pc_range=use_pc_range)
