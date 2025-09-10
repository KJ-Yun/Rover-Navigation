import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import json
import re
import os
import open3d as o3d

# 坐标转换矩阵
T_lidar2cam = np.array([
        [-0.999946993346232,-0.00800835494170636,0.00647122468774145,-0.00498667711611016],
        [-0.00664188970645313,0.0214710177437833,-0.999747408447841,-0.0158810369834515],
        [0.00786738831880689,-0.999737396343747,-0.0215230702472327,-0.145366505605775],
        [0,0,0,1]
])
T_lidar2imu = np.array([
    [1, 0, 0, 0.011],
    [0, 1, 0, 0.02329],
    [0, 0, 1, -0.04412],
    [0, 0, 0, 1]
])
T_cam2ego = np.array([
    [-0.99972652, -0.0190869,  -0.01351239, -0.00110909],
    [0.00389408,  0.43386909, -0.90096751, -0.04466106],
    [0.02305929, -0.90077373, -0.4336761, 0.05583883],
    [0, 0, 0, 1]
])

# 构建变换矩阵
T_imu2lidar = np.linalg.inv(T_lidar2imu)
T_imu2ego = T_cam2ego @ T_lidar2cam @ T_imu2lidar
T_lidar2ego = T_cam2ego @ T_lidar2cam  # lidar到ego的直接变换

def load_images_txt(filename):
    """加载COLMAP图像文件（如果需要）"""
    images = []
    pattern = re.compile(
        r'^(\d+)\s+' +                     # IMAGE_ID
        r'([-+]?\d*\.\d+|\d+)\s+' +       # QW
        r'([-+]?\d*\.\d+|\d+)\s+' +       # QX
        r'([-+]?\d*\.\d+|\d+)\s+' +       # QY
        r'([-+]?\d*\.\d+|\d+)\s+' +       # QZ
        r'([-+]?\d*\.\d+|\d+)\s+' +       # TX
        r'([-+]?\d*\.\d+|\d+)\s+' +       # TY
        r'([-+]?\d*\.\d+|\d+)\s+' +       # TZ
        r'(\d+)\s+' +                     # CAMERA_ID
        r'(\S+)$'                        # NAME (非空白字符直到行尾)
    )
    with open(filename, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].strip()
        if pattern.match(line):
            parts = line.split()
            img_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            cam_id = int(parts[8])
            name = parts[9]

            # 构建相机在 IMU 坐标系下的 4x4 矩阵
            R_cam_in_imu = R.from_quat([qx, qy, qz, qw]).as_matrix()
            T_cam_in_imu = np.eye(4)
            T_cam_in_imu[:3, :3] = R_cam_in_imu
            T_cam_in_imu[:3, 3] = [tx, ty, tz]

            # 转换到 Ego 坐标系
            T_cam_in_ego = T_imu2ego @ T_cam_in_imu
            pos_in_ego = T_cam_in_ego[:3, 3]

            images.append({
                'id': img_id,
                'tx': pos_in_ego[0], 'ty': pos_in_ego[1], 'tz': pos_in_ego[2],
                'cam_id': cam_id, 'name': name
            })
    return pd.DataFrame(images)

def build_transform_matrix(rpy_deg, trans_xyz):
    """根据欧拉角和平移构建变换矩阵"""
    r = R.from_euler('xyz', rpy_deg, degrees=True)
    t = np.eye(4)
    t[:3, :3] = r.as_matrix()
    t[:3, 3] = trans_xyz
    return t

def decompose_transform_matrix(T):
    """分解变换矩阵为欧拉角和平移"""
    r = R.from_matrix(T[:3, :3])
    rpy_deg = r.as_euler('xyz', degrees=True)
    trans_xyz = T[:3, 3]
    return rpy_deg, trans_xyz

def process_pointcloud(pcd_path, output_path):
    """处理点云数据，转换到ego坐标系"""
    if not os.path.exists(pcd_path):
        print(f"警告: 点云文件 {pcd_path} 不存在，跳过点云处理")
        return False
        
    print("正在处理点云数据...")
    
    # 加载点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"加载点云，包含 {len(pcd.points)} 个点")
    
    # 提取点的坐标
    points = np.asarray(pcd.points)
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack([points, ones])  # N×4 齐次坐标
    
    # 应用变换：lidar → ego
    points_ego_hom = (T_lidar2ego @ points_hom.T).T
    points_ego = points_ego_hom[:, :3]
    
    # 保存点云数据
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        # 合并坐标和颜色信息
        colored_pointcloud = np.hstack([points_ego, colors])
        print(f"保存带颜色的点云，形状: {colored_pointcloud.shape}")
    else:
        # 如果没有颜色信息，创建默认白色
        colors = np.ones((points_ego.shape[0], 3))  # 白色 [1, 1, 1]
        colored_pointcloud = np.hstack([points_ego, colors])
        print(f"保存点云（添加默认颜色），形状: {colored_pointcloud.shape}")
    
    # 保存为npy文件
    np.save(output_path, colored_pointcloud)
    print(f"✅ 点云数据已保存到: {output_path}")
    
    return True

def process_imu_data(mat_file, imu_file, output_path):
    """处理IMU数据，转换到ego坐标系并保存为JSON"""
    print("=== 处理IMU数据 ===")
    
    # 加载IMU相关数据
    mat_cols = ['time', 'roll', 'pitch', 'yaw', 'x', 'y', 'z',
                'vx', 'vy', 'vz', 'bgx', 'bgy', 'bgz',
                'bax', 'bay', 'baz', 'inv_t', 'dummy1', 'dummy2', 'n_points']
    mat = pd.read_csv(mat_file, sep=r'\s+', names=mat_cols, engine='python')
    print(f"加载mat数据: {len(mat)} 行")

    imu_cols = ['time', 'gyr_x', 'gyr_y', 'gyr_z', 'acc_x', 'acc_y', 'acc_z']
    imu = pd.read_csv(imu_file, sep=r'\s+', names=imu_cols, engine='python')
    print(f"加载imu数据: {len(imu)} 行")

    R_imu2ego = T_imu2ego[:3, :3]

    # 准备输出字典
    meta_info = {}
    
    for i, row in mat.iterrows():
        time = row['time']
        frame_id = str(i + 1)  # 使用字符串作为key

        # 1. 位姿变换：IMU → EGO
        rpy_imu = [row['roll'], row['pitch'], row['yaw']]
        trans_imu = [row['x'], row['y'], row['z']]
        T_imu = build_transform_matrix(rpy_imu, trans_imu)
        T_ego = T_imu2ego @ T_imu
        rpy_ego, xyz_ego = decompose_transform_matrix(T_ego)

        # 2. 匹配最近的IMU数据
        imu_idx = np.argmin(np.abs(imu['time'] - time))
        imu_row = imu.iloc[imu_idx]
        acc_imu = np.array([imu_row['acc_x'], imu_row['acc_y'], imu_row['acc_z']])
        gyr_imu = np.array([imu_row['gyr_x'], imu_row['gyr_y'], imu_row['gyr_z']])
        
        # 3. 加速度和陀螺仪数据转换到Ego坐标系
        acc_ego = R_imu2ego @ acc_imu
        gyr_ego = R_imu2ego @ gyr_imu

        # 4. 计算图像序号（按照原代码逻辑）
        img_idx = (i // 2) + 1
        img_name = f"{img_idx:05d}.png"  # 5位数字，前面补0

        # 5. 构建当前帧的信息
        frame_info = {
            "time": float(time),
            "roll": float(rpy_ego[0]),
            "pitch": float(rpy_ego[1]),
            "yaw": float(rpy_ego[2]),
            "x": float(xyz_ego[0]),
            "y": float(xyz_ego[1]),
            "z": float(xyz_ego[2]),
            "acc_x": float(acc_ego[0]),
            "acc_y": float(acc_ego[1]),
            "acc_z": float(acc_ego[2]),
            "gyr_x": float(gyr_ego[0]),
            "gyr_y": float(gyr_ego[1]),
            "gyr_z": float(gyr_ego[2]),
            "image_name": img_name
        }
        
        meta_info[frame_id] = frame_info

    # 保存IMU数据为JSON文件
    json_output_path = os.path.join(output_path, 'data.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ IMU数据已保存到: {json_output_path}")
    print(f"包含 {len(meta_info)} 帧数据")
    
    return json_output_path, len(meta_info)

if __name__ == "__main__":
    # 添加命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='处理IMU数据和点云数据，转换到ego坐标系')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径（包含mat_out.txt, imu.txt, all_raw_points.pcd）')
    args = parser.parse_args()
    
    mat_file = os.path.join(args.data_dir, 'mat_out.txt')
    imu_file = os.path.join(args.data_dir, 'imu.txt')
    pcd_file = os.path.join(args.data_dir, 'PCD', 'all_raw_points.pcd')
    output_path = args.data_dir  # 输出路径与数据目录相同

    print("开始处理IMU和点云数据...")
    
    # 处理IMU数据
    json_path, frame_count = process_imu_data(mat_file, imu_file, output_path)
    
    # 处理点云数据
    print("\n=== 处理点云数据 ===")
    pcd_output_path = os.path.join(output_path, 'pointcloud.npy')
    pcd_success = process_pointcloud(pcd_file, pcd_output_path)
    
    # 输出处理结果
    print("\n" + "="*50)
    print("✅ 所有数据处理完成!")
    print(f"- IMU数据: {json_path} ({frame_count} 帧)")
    if pcd_success:
        print(f"- 点云数据: {pcd_output_path}")
    print("="*50)