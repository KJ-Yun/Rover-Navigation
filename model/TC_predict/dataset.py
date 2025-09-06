import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def collate_fn(batch):
    """
    自定义 collate，用于 DataLoader
    batch: list of (pc_list, img, cost)
    """
    pcs, imgs, costs = [], [], []
    for sample in batch:
        if sample is None:  # 可能有跳过的情况
            continue
        pc_list, img, cost = sample
        pcs.append(pc_list)     # list of N_i x 6
        imgs.append(img)        # Tensor
        costs.append(cost)      # Tensor

    # imgs, costs 可以堆叠成 tensor
    imgs = torch.stack(imgs, dim=0)
    costs = torch.stack(costs, dim=0)

    return pcs, imgs, costs


class PCImageDataset(Dataset):
    def __init__(self, root_dir, patch_size=1.0, offset=-1.5,
                 img_size=256, img_subdir="images",
                 sample_step=1.0, min_points=100):
        """
        Args:
            root_dir (str): 根目录，里面包含若干 sub_dir
            patch_size (float): patch 大小（米）
            offset (float): 预测 offset（米）
            img_size (int): 输出图像大小
            img_subdir (str): 图像所在子文件夹
            sample_step (float): 两个样本之间的最小间距（米）
            min_points (int): patch 内点云最少点数，低于则跳过
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.offset = offset
        self.img_size = img_size
        self.img_subdir = img_subdir
        self.sample_step = sample_step
        self.min_points = min_points

        # 图像预处理
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

        self.samples = []
        self.subdirs = []

        # 遍历 root_dir 下的所有 sub_dir
        for sub in sorted(os.listdir(root_dir)):
            sub_path = os.path.join(root_dir, sub)
            if not os.path.isdir(sub_path):
                continue

            data_path = os.path.join(sub_path, "data.json")
            meta_path = os.path.join(sub_path, "meta.json")
            pc_path = os.path.join(sub_path, "pointcloud.npy")
            label_path = os.path.join(sub_path, "label.npy")
            img_path = os.path.join(sub_path, img_subdir)

            if not (os.path.exists(data_path) and os.path.exists(meta_path) and
                    os.path.exists(pc_path) and os.path.exists(label_path)):
                print(f"[WARN] Skip {sub_path}, missing required files.")
                continue

            with open(data_path, "r") as f:
                traj = json.load(f)
            with open(meta_path, "r") as f:
                meta = json.load(f)

            traj_sorted = [traj[k] for k in sorted(traj.keys(), key=lambda x: int(x))]

            # 按 sample_step 采样
            filtered_indices = []
            last_x, last_y = None, None
            for i, pose in enumerate(traj_sorted):
                x, y = pose["x"], pose["y"]
                if last_x is None:
                    filtered_indices.append(i)
                    last_x, last_y = x, y
                else:
                    dist = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
                    # print(f"[DEBUG] Subdir {sub}, frame {i} and frame {filtered_indices[-1]} distance: {dist:.3f}m")
                    if dist >= self.sample_step:
                        filtered_indices.append(i)
                        last_x, last_y = x, y

            # 存储子目录信息
            sub_info = {
                "path": sub_path,
                "traj": traj_sorted,
                "meta": meta,
                "pc": np.load(pc_path),
                "tc": np.load(label_path),
                "img_root": img_path,
                "indices": filtered_indices
            }
            self.subdirs.append(sub_info)

            # 建立 samples
            for i in filtered_indices:
                self.samples.append((len(self.subdirs) - 1, i))

        print(f"[INFO] Loaded {len(self.subdirs)} sequences, {len(self.samples)} raw samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        debug_mode = False  # 设置为True启用调试
        
        sub_idx, frame_idx = self.samples[idx]
        sub_info = self.subdirs[sub_idx]
        if debug_mode:
            print(f"\n[DEBUG] Fetching sample index: {sub_idx}-{frame_idx}")
        traj = sub_info["traj"]
        meta = sub_info["meta"]
        pc_all = sub_info["pc"]
        tc = sub_info["tc"]
        img_root = sub_info["img_root"]

        cur_pose = traj[frame_idx]
        img_path = os.path.join(img_root, cur_pose["image_name"])

        # === 1. 加载图像 ===
        if not os.path.exists(img_path):
            print(f"[ERROR] Image not found: {img_path}")
            return None
            
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        # === 2. 获取当前位姿 ===
        x, y, yaw = cur_pose["x"], cur_pose["y"], cur_pose["yaw"]

        # === 3. 计算裁剪中心（局部坐标系偏移转全局坐标系）===
        # 局部坐标系偏移：x=0, y=-1.5（前方1.5米）
        local_offset_x = 0.0
        local_offset_y = self.offset
        
        # 坐标系变换：局部坐标 -> 全局坐标
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # 全局坐标系中的裁剪中心
        crop_center_x = x + cos_yaw * local_offset_x - sin_yaw * local_offset_y
        crop_center_y = y + sin_yaw * local_offset_x + cos_yaw * local_offset_y

        # === 4. 定义裁剪范围 ===
        half_patch = self.patch_size / 2
        crop_x_min = crop_center_x - half_patch
        crop_x_max = crop_center_x + half_patch
        crop_y_min = crop_center_y - half_patch
        crop_y_max = crop_center_y + half_patch

        # === 5. 裁剪点云（物理范围）===
        pc_mask = (
            (pc_all[:, 0] >= crop_x_min) & (pc_all[:, 0] <= crop_x_max) &
            (pc_all[:, 1] >= crop_y_min) & (pc_all[:, 1] <= crop_y_max)
        )
        pc_cropped = pc_all[pc_mask].copy()

        if pc_cropped.shape[0] < self.min_points:
            print(f"[WARN] Skip sample {sub_idx}-{frame_idx}, only {pc_cropped.shape[0]} points in patch.")
            return None

        # === 6. 将裁剪后的点云移动到[0,1,0,1]范围内 ===
        # X方向：从[crop_x_min, crop_x_max] -> [0, 1]
        pc_cropped[:, 0] = pc_cropped[:, 0] - crop_x_min
        # Y方向：从[crop_y_min, crop_y_max] -> [0, 1] 
        pc_cropped[:, 1] = pc_cropped[:, 1] - crop_y_min
        
        # === 7. 裁剪TC地图（栅格范围）===
        grid_size = meta["grid_size"]
        tc_x0, tc_x1, tc_y0, tc_y1 = meta["pc_range"]
        
        # 检查裁剪范围是否在TC地图范围内
        if (crop_x_min < tc_x0 or crop_x_max > tc_x1 or 
            crop_y_min < tc_y0 or crop_y_max > tc_y1):
            print(f"[WARN] Skip sample {sub_idx}-{frame_idx}, crop area out of TC map bounds.")
            print(f"  Crop range: x[{crop_x_min:.2f}, {crop_x_max:.2f}], y[{crop_y_min:.2f}, {crop_y_max:.2f}]")
            print(f"  TC range: x[{tc_x0:.2f}, {tc_x1:.2f}], y[{tc_y0:.2f}, {tc_y1:.2f}]")
            return None

        # 计算TC地图中的栅格索引范围
        tc_x_min_idx = int((crop_x_min - tc_x0) / grid_size)
        tc_x_max_idx = int((crop_x_max - tc_x0) / grid_size)
        tc_y_min_idx = int((crop_y_min - tc_y0) / grid_size)
        tc_y_max_idx = int((crop_y_max - tc_y0) / grid_size)
        
        # 确保索引在有效范围内
        tc_H, tc_W = tc.shape
        tc_x_min_idx = max(0, tc_x_min_idx)
        tc_x_max_idx = min(tc_W, tc_x_max_idx)
        tc_y_min_idx = max(0, tc_y_min_idx)
        tc_y_max_idx = min(tc_H, tc_y_max_idx)
        
        # 检查是否有有效的TC区域
        if tc_x_max_idx <= tc_x_min_idx or tc_y_max_idx <= tc_y_min_idx:
            print(f"[WARN] Skip sample {sub_idx}-{frame_idx}, invalid TC crop indices.")
            return None

        # 裁剪TC patch
        tc_patch = tc[tc_y_min_idx:tc_y_max_idx, tc_x_min_idx:tc_x_max_idx]
        
        # === 8. 验证TC patch大小 ===
        expected_tc_size = int(self.patch_size / grid_size)
        actual_tc_h, actual_tc_w = tc_patch.shape
        
        # 如果尺寸不完全匹配，进行适当调整（通常是边界效应导致的1个像素差异）
        if abs(actual_tc_h - expected_tc_size) > 1 or abs(actual_tc_w - expected_tc_size) > 1:
            print(f"[WARN] Skip sample {sub_idx}-{frame_idx}, TC patch size mismatch.")
            print(f"  Expected: {expected_tc_size}x{expected_tc_size}, got: {actual_tc_h}x{actual_tc_w}")
            return None
        
        # 如果有轻微尺寸差异，进行裁剪或填充到期望尺寸
        if actual_tc_h != expected_tc_size or actual_tc_w != expected_tc_size:
            # 创建期望尺寸的数组，用NaN填充
            tc_patch_resized = np.full((expected_tc_size, expected_tc_size), np.nan, dtype=np.float32)
            
            # 计算复制的范围
            copy_h = min(actual_tc_h, expected_tc_size)
            copy_w = min(actual_tc_w, expected_tc_size)
            
            # 居中复制
            start_h = (expected_tc_size - copy_h) // 2
            start_w = (expected_tc_size - copy_w) // 2
            
            tc_patch_resized[start_h:start_h+copy_h, start_w:start_w+copy_w] = \
                tc_patch[:copy_h, :copy_w]
            
            tc_patch = tc_patch_resized

        tc_patch = torch.tensor(tc_patch, dtype=torch.float32).unsqueeze(0)

        # === 9. 最终验证 ===
        # 验证点云范围是否在[0,1,0,1]内
        if pc_cropped.shape[0] > 0:
            pc_x_min, pc_x_max = pc_cropped[:, 0].min(), pc_cropped[:, 0].max()
            pc_y_min, pc_y_max = pc_cropped[:, 1].min(), pc_cropped[:, 1].max()
            
            if not (0 <= pc_x_min and pc_x_max <= self.patch_size and 0 <= pc_y_min and pc_y_max <= self.patch_size):
                print(f"[ERROR] Point cloud not in [0,1,0,1] range!")
                print(f"  X range: [{pc_x_min:.4f}, {pc_x_max:.4f}]")
                print(f"  Y range: [{pc_y_min:.4f}, {pc_y_max:.4f}]")
                raise IndexError("Point cloud range validation failed")

        # 验证TC patch尺寸
        if tc_patch.shape[1] != expected_tc_size or tc_patch.shape[2] != expected_tc_size:
            print(f"[ERROR] TC patch final size mismatch!")
            print(f"  Expected: 1x{expected_tc_size}x{expected_tc_size}")
            print(f"  Got: {tc_patch.shape}")
            raise IndexError("TC patch size validation failed")

        # === 10. 调试信息（可选开启）===
        if debug_mode:
            print(f"\n=== Sample {sub_idx}-{frame_idx} ===")
            print(f"Pose: ({x:.3f}, {y:.3f}, {np.degrees(yaw):.1f}°)")
            print(f"Crop center: ({crop_center_x:.3f}, {crop_center_y:.3f})")
            print(f"Crop range: x[{crop_x_min:.3f}, {crop_x_max:.3f}], y[{crop_y_min:.3f}, {crop_y_max:.3f}]")
            print(f"Point cloud: {pc_cropped.shape[0]} points")
            if pc_cropped.shape[0] > 0:
                print(f"  X range: [{pc_x_min:.4f}, {pc_x_max:.4f}]")
                print(f"  Y range: [{pc_y_min:.4f}, {pc_y_max:.4f}]")
            print(f"TC patch: {tc_patch.shape}")
            print(f"Image: {img.shape}")

        # 转换点云为列表格式
        pc_list = pc_cropped.tolist()

        return pc_list, img, tc_patch



if __name__ == "__main__":
    dataset = PCImageDataset("D:/rover_pro/data/",
                             patch_size=1.0, offset=-1.5,
                             img_size=1024, min_points=100,
                             img_subdir="Colmap/images",
                             sample_step=0.2)
    print(f"Dataset size: {len(dataset)}")

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

    for i, (pc, img, cost) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Point Cloud: {len(pc[0])}")  # (B, npoints, 6)
        print(f"  Image: {img.shape}")       # (B, 3, img_size, img_size)
        print(f"  Cost Patch: {cost.shape}") # (B, 1, H, W)
        if i == 2:
            break