import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
import random


def collate_fn(batch):
    """
    自定义 collate，用于 DataLoader
    batch: list of (pc_list, img, cost)
    """
    # 过滤掉None样本
    valid_samples = [sample for sample in batch if sample is not None]
    
    # 如果整个batch都是None，返回空结果
    if len(valid_samples) == 0:
        return [], torch.empty(0, 3, 256, 256), torch.empty(0, 1, 100, 100)
    
    pcs, imgs, costs = [], [], []
    for sample in valid_samples:
        pc_list, img, cost = sample
        pcs.append(pc_list)     # list of N_i x 6
        imgs.append(img)        # Tensor
        costs.append(cost)      # Tensor

    # imgs, costs 可以堆叠成 tensor
    imgs = torch.stack(imgs, dim=0)
    costs = torch.stack(costs, dim=0)

    return pcs, imgs, costs


class RobustnessAugmentation:
    """鲁棒性测试的数据增强类"""
    
    def __init__(self, 
                 image_occlusion=False,
                 occlusion_ratio=0.15,
                 occlusion_num_patches=5,
                 pointcloud_sparsify=False,
                 sparsify_ratio=0.3,
                 pointcloud_noise=False,
                 noise_std=0.02):
        """
        Args:
            image_occlusion (bool): 是否启用图像遮挡
            occlusion_ratio (float): 遮挡区域占图像的比例 [0.1-0.3]
            occlusion_num_patches (int): 遮挡块的数量
            pointcloud_sparsify (bool): 是否启用点云稀疏化
            sparsify_ratio (float): 保留点的比例 [0.1-0.8]
            pointcloud_noise (bool): 是否添加点云噪声
            noise_std (float): 高斯噪声标准差（米）
        """
        self.image_occlusion = image_occlusion
        self.occlusion_ratio = occlusion_ratio
        self.occlusion_num_patches = occlusion_num_patches
        
        self.pointcloud_sparsify = pointcloud_sparsify
        self.sparsify_ratio = sparsify_ratio
        
        self.pointcloud_noise = pointcloud_noise
        self.noise_std = noise_std
    
    def occlude_image(self, img_tensor):
        """
        随机遮挡图像的部分区域
        
        Args:
            img_tensor: torch.Tensor, shape (C, H, W)
        Returns:
            torch.Tensor: 遮挡后的图像
        """
        if not self.image_occlusion:
            return img_tensor
            
        C, H, W = img_tensor.shape
        img_copy = img_tensor.clone()
        
        # 计算总遮挡面积
        total_area = H * W
        target_occluded_area = int(total_area * self.occlusion_ratio)
        
        # 随机生成多个遮挡块
        for _ in range(self.occlusion_num_patches):
            # 随机选择遮挡块大小
            max_size = int(np.sqrt(target_occluded_area / self.occlusion_num_patches))
            patch_h = random.randint(max_size // 3, max_size)
            patch_w = random.randint(max_size // 3, max_size)
            
            # 随机选择位置
            start_h = random.randint(0, max(1, H - patch_h))
            start_w = random.randint(0, max(1, W - patch_w))
            
            # 选择遮挡方式
            occlusion_type = random.choice(['black', 'white', 'noise', 'blur'])
            
            if occlusion_type == 'black':
                img_copy[:, start_h:start_h+patch_h, start_w:start_w+patch_w] = 0.0
            elif occlusion_type == 'white':
                img_copy[:, start_h:start_h+patch_h, start_w:start_w+patch_w] = 1.0
            elif occlusion_type == 'noise':
                noise = torch.randn(C, patch_h, patch_w) * 0.5
                img_copy[:, start_h:start_h+patch_h, start_w:start_w+patch_w] = noise
            elif occlusion_type == 'blur':
                # 简单的均值滤波模拟模糊
                patch = img_copy[:, start_h:start_h+patch_h, start_w:start_w+patch_w]
                blurred = torch.ones_like(patch) * patch.mean()
                img_copy[:, start_h:start_h+patch_h, start_w:start_w+patch_w] = blurred
        
        return img_copy
    
    def sparsify_pointcloud(self, pc_tensor):
        """
        随机稀疏化点云
        
        Args:
            pc_tensor: torch.Tensor, shape (N, 6) - xyz+rgb
        Returns:
            torch.Tensor: 稀疏化后的点云
        """
        if not self.pointcloud_sparsify or pc_tensor.shape[0] == 0:
            return pc_tensor
        
        N = pc_tensor.shape[0]
        keep_num = max(1, int(N * self.sparsify_ratio))
        
        # 随机选择要保留的点的索引
        indices = torch.randperm(N)[:keep_num]
        indices = torch.sort(indices)[0]  # 保持顺序
        
        return pc_tensor[indices]
    
    def add_pointcloud_noise(self, pc_tensor):
        """
        给点云添加高斯噪声
        
        Args:
            pc_tensor: torch.Tensor, shape (N, 6) - xyz+rgb
        Returns:
            torch.Tensor: 添加噪声后的点云
        """
        if not self.pointcloud_noise or pc_tensor.shape[0] == 0:
            return pc_tensor
        
        pc_copy = pc_tensor.clone()
                
        # 10%点云添加噪声
        noise_idx = torch.rand(pc_tensor.shape[0]) < 0.1
        xyz_noise = torch.normal(0, self.noise_std, size=(noise_idx.sum(), 3))
        pc_copy[noise_idx, :3] += xyz_noise
        
        # RGB噪声（可选）- 添加轻微的颜色噪声
        # if random.random() < 0.3:  # 30%概率添加RGB噪声
        #     rgb_noise = torch.normal(0, 0.01, size=(pc_tensor.shape[0], 3))
        #     pc_copy[:, 3:6] = torch.clamp(pc_copy[:, 3:6] + rgb_noise, 0, 1)
        
        return pc_copy
    
    def apply_augmentations(self, pc_tensor, img_tensor):
        """
        应用所有启用的增强
        
        Args:
            pc_tensor: torch.Tensor, 点云
            img_tensor: torch.Tensor, 图像
        Returns:
            tuple: (增强后的点云, 增强后的图像)
        """
        # 应用点云增强
        if self.pointcloud_sparsify:
            pc_tensor = self.sparsify_pointcloud(pc_tensor)
        
        if self.pointcloud_noise:
            pc_tensor = self.add_pointcloud_noise(pc_tensor)
        
        # 应用图像增强
        if self.image_occlusion:
            img_tensor = self.occlude_image(img_tensor)
        
        return pc_tensor, img_tensor


class PCImageDataset(Dataset):
    def __init__(self, root_dir, patch_size=1.0, offset=-1.5,
                 img_size=256, img_subdir="images",
                 sample_step=1.0, min_points=100, 
                 prefilter_samples=True,
                 # 新增的鲁棒性测试参数
                 robustness_test=False,
                 image_occlusion=False,
                 occlusion_ratio=0.15,
                 occlusion_num_patches=5,
                 pointcloud_sparsify=False,
                 sparsify_ratio=0.3,
                 pointcloud_noise=False,
                 noise_std=0.02):
        """
        Args:
            root_dir (str): 根目录，里面包含若干 sub_dir
            patch_size (float): patch 大小（米）
            offset (float): 预测 offset（米）
            img_size (int): 输出图像大小
            img_subdir (str): 图像所在子文件夹
            sample_step (float): 两个样本之间的最小间距（米）
            min_points (int): patch 内点云最少点数，低于则跳过
            prefilter_samples (bool): 是否在初始化时预过滤有效样本
            
            # 鲁棒性测试参数
            robustness_test (bool): 是否启用鲁棒性测试模式
            image_occlusion (bool): 是否启用图像遮挡
            occlusion_ratio (float): 遮挡区域占图像的比例
            occlusion_num_patches (int): 遮挡块的数量
            pointcloud_sparsify (bool): 是否启用点云稀疏化
            sparsify_ratio (float): 保留点的比例
            pointcloud_noise (bool): 是否添加点云噪声
            noise_std (float): 高斯噪声标准差（米）
        """
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.offset = offset
        self.img_size = img_size
        self.img_subdir = img_subdir
        self.sample_step = sample_step
        self.min_points = min_points
        self.prefilter_samples = prefilter_samples

        # 鲁棒性测试设置
        self.robustness_test = robustness_test
        if robustness_test:
            self.augmentation = RobustnessAugmentation(
                image_occlusion=image_occlusion,
                occlusion_ratio=occlusion_ratio,
                occlusion_num_patches=occlusion_num_patches,
                pointcloud_sparsify=pointcloud_sparsify,
                sparsify_ratio=sparsify_ratio,
                pointcloud_noise=pointcloud_noise,
                noise_std=noise_std
            )
            print(f"[INFO] Robustness testing enabled:")
            print(f"  - Image occlusion: {image_occlusion} (ratio={occlusion_ratio})")
            print(f"  - Pointcloud sparsify: {pointcloud_sparsify} (ratio={sparsify_ratio})")
            print(f"  - Pointcloud noise: {pointcloud_noise} (std={noise_std})")
        else:
            self.augmentation = None

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

            # 建立 samples - 如果启用预过滤，只添加有效样本
            if self.prefilter_samples:
                print(f"[INFO] Pre-filtering samples for sequence {sub}...")
                for i in filtered_indices:
                    if self._is_valid_sample(len(self.subdirs) - 1, i):
                        self.samples.append((len(self.subdirs) - 1, i))
            else:
                # 不预过滤，添加所有样本
                for i in filtered_indices:
                    self.samples.append((len(self.subdirs) - 1, i))

        if self.prefilter_samples:
            print(f"[INFO] Loaded {len(self.subdirs)} sequences, {len(self.samples)} valid samples (pre-filtered).")
        else:
            print(f"[INFO] Loaded {len(self.subdirs)} sequences, {len(self.samples)} raw samples (not pre-filtered).")

    def _is_valid_sample(self, sub_idx, frame_idx):
        """
        检查样本是否有效（不会被跳过）
        """
        try:
            sub_info = self.subdirs[sub_idx]
            traj = sub_info["traj"]
            meta = sub_info["meta"]
            pc_all = sub_info["pc"]
            tc = sub_info["tc"]
            img_root = sub_info["img_root"]

            cur_pose = traj[frame_idx]
            img_path = os.path.join(img_root, cur_pose["image_name"])

            # 检查图像是否存在
            if not os.path.exists(img_path):
                return False

            # 获取当前位姿
            x, y, yaw = cur_pose["x"], cur_pose["y"], cur_pose["yaw"]

            # 计算裁剪中心
            local_offset_x = 0.0
            local_offset_y = self.offset
            
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            
            crop_center_x = x + cos_yaw * local_offset_x - sin_yaw * local_offset_y
            crop_center_y = y + sin_yaw * local_offset_x + cos_yaw * local_offset_y

            # 定义裁剪范围
            half_patch = self.patch_size / 2
            crop_x_min = crop_center_x - half_patch
            crop_x_max = crop_center_x + half_patch
            crop_y_min = crop_center_y - half_patch
            crop_y_max = crop_center_y + half_patch

            # 检查裁剪后的点云是否有足够的点
            pc_mask = (
                (pc_all[:, 0] >= crop_x_min) & (pc_all[:, 0] <= crop_x_max) &
                (pc_all[:, 1] >= crop_y_min) & (pc_all[:, 1] <= crop_y_max)
            )
            pc_cropped = pc_all[pc_mask]

            # 考虑稀疏化后的点数
            min_points_threshold = self.min_points
            if self.robustness_test and self.augmentation.pointcloud_sparsify:
                # 如果启用稀疏化，需要确保稀疏化后还有足够的点
                min_points_threshold = int(self.min_points / self.augmentation.sparsify_ratio) + 10

            if pc_cropped.shape[0] < min_points_threshold:
                return False

            # 检查裁剪范围是否在TC地图范围内
            tc_x0, tc_x1, tc_y0, tc_y1 = meta["pc_range"]
            
            if (crop_x_min < tc_x0 or crop_x_max > tc_x1 or 
                crop_y_min < tc_y0 or crop_y_max > tc_y1):
                return False

            # 检查TC栅格索引是否有效
            grid_size = meta["grid_size"]
            tc_x_min_idx = int((crop_x_min - tc_x0) / grid_size)
            tc_x_max_idx = int((crop_x_max - tc_x0) / grid_size)
            tc_y_min_idx = int((crop_y_min - tc_y0) / grid_size)
            tc_y_max_idx = int((crop_y_max - tc_y0) / grid_size)
            
            tc_H, tc_W = tc.shape
            tc_x_min_idx = max(0, tc_x_min_idx)
            tc_x_max_idx = min(tc_W, tc_x_max_idx)
            tc_y_min_idx = max(0, tc_y_min_idx)
            tc_y_max_idx = min(tc_H, tc_y_max_idx)
            
            if tc_x_max_idx <= tc_x_min_idx or tc_y_max_idx <= tc_y_min_idx:
                return False

            # 检查TC patch尺寸是否合理
            expected_tc_size = int(self.patch_size / grid_size)
            actual_tc_h = tc_y_max_idx - tc_y_min_idx
            actual_tc_w = tc_x_max_idx - tc_x_min_idx
            
            if abs(actual_tc_h - expected_tc_size) > 1 or abs(actual_tc_w - expected_tc_size) > 1:
                return False

            return True

        except Exception:
            return False

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
            if not self.prefilter_samples:  # 只有在未预过滤时才打印警告
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
            if not self.prefilter_samples:  # 只有在未预过滤时才打印警告
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
            if not self.prefilter_samples:  # 只有在未预过滤时才打印警告
                print(f"[WARN] Skip sample {sub_idx}-{frame_idx}, invalid TC crop indices.")
            return None

        # 裁剪TC patch
        tc_patch = tc[tc_y_min_idx:tc_y_max_idx, tc_x_min_idx:tc_x_max_idx]
        
        # === 8. 验证TC patch大小 ===
        expected_tc_size = int(self.patch_size / grid_size)
        actual_tc_h, actual_tc_w = tc_patch.shape
        
        # 如果尺寸不完全匹配，进行适当调整（通常是边界效应导致的1个像素差异）
        if abs(actual_tc_h - expected_tc_size) > 1 or abs(actual_tc_w - expected_tc_size) > 1:
            if not self.prefilter_samples:  # 只有在未预过滤时才打印警告
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

        # 转换点云为tensor
        pc = torch.from_numpy(pc_cropped).float()

        # === 9. 应用鲁棒性测试增强 ===
        if self.robustness_test and self.augmentation is not None:
            pc, img = self.augmentation.apply_augmentations(pc, img)
            
            # 稀疏化后重新检查点数
            if self.augmentation.pointcloud_noise:
                aug_pc_mask = (
                    (pc[:, 0] >= 0) & (pc[:, 0] <= self.patch_size) &
                    (pc[:, 1] >= 0) & (pc[:, 1] <= self.patch_size)
                )
                pc = pc[aug_pc_mask]
            if pc.shape[0] < self.min_points:
                if debug_mode:
                    print(f"[WARN] Skip sample {sub_idx}-{frame_idx} after sparsification, only {pc.shape[0]} points left.")
                return None


        # === 10. 最终验证 ===
        # 验证点云范围是否在[0,patch_size,0,patch_size]内
        if pc.shape[0] > 0:
            pc_x_min, pc_x_max = pc[:, 0].min(), pc[:, 0].max()
            pc_y_min, pc_y_max = pc[:, 1].min(), pc[:, 1].max()
            
            if not (0 <= pc_x_min and pc_x_max <= self.patch_size and 0 <= pc_y_min and pc_y_max <= self.patch_size):
                print(f"[ERROR] Point cloud not in [0,{self.patch_size},0,{self.patch_size}] range!")
                print(f"  X range: [{pc_x_min:.4f}, {pc_x_max:.4f}]")
                print(f"  Y range: [{pc_y_min:.4f}, {pc_y_max:.4f}]")
                raise IndexError("Point cloud range validation failed")

        # 验证TC patch尺寸
        if tc_patch.shape[1] != expected_tc_size or tc_patch.shape[2] != expected_tc_size:
            print(f"[ERROR] TC patch final size mismatch!")
            print(f"  Expected: 1x{expected_tc_size}x{expected_tc_size}")
            print(f"  Got: {tc_patch.shape}")
            raise IndexError("TC patch size validation failed")

        # === 11. 调试信息（可选开启）===
        if debug_mode:
            print(f"\n=== Sample {sub_idx}-{frame_idx} ===")
            print(f"Pose: ({x:.3f}, {y:.3f}, {np.degrees(yaw):.1f}°)")
            print(f"Crop center: ({crop_center_x:.3f}, {crop_center_y:.3f})")
            print(f"Crop range: x[{crop_x_min:.3f}, {crop_x_max:.3f}], y[{crop_y_min:.3f}, {crop_y_max:.3f}]")
            print(f"Point cloud: {pc.shape[0]} points")
            if pc.shape[0] > 0:
                print(f"  X range: [{pc_x_min:.4f}, {pc_x_max:.4f}]")
                print(f"  Y range: [{pc_y_min:.4f}, {pc_y_max:.4f}]")
            print(f"TC patch: {tc_patch.shape}")
            print(f"Image: {img.shape}")
            if self.robustness_test:
                print(f"Robustness augmentations applied:")
                print(f"  - Image occlusion: {self.augmentation.image_occlusion}")
                print(f"  - PC sparsify: {self.augmentation.pointcloud_sparsify}")
                print(f"  - PC noise: {self.augmentation.pointcloud_noise}")

        return pc, img, tc_patch


# 便利函数用于创建不同的鲁棒性测试配置
def create_robustness_configs():
    """
    创建预定义的鲁棒性测试配置
    """
    configs = {}
    
    # 1. 图像遮挡测试
    configs['image_occlusion'] = {
        'robustness_test': True,
        'image_occlusion': True,
        'occlusion_ratio': 0.2,
        'occlusion_num_patches': 3,
        'pointcloud_sparsify': False,
        'pointcloud_noise': False
    }
    
    # 2. 点云稀疏化测试
    configs['pointcloud_sparse'] = {
        'robustness_test': True,
        'image_occlusion': False,
        'pointcloud_sparsify': True,
        'sparsify_ratio': 0.7,  # 保留70%的点
        'pointcloud_noise': False
    }
    
    # 3. 点云噪声测试
    configs['pointcloud_noise'] = {
        'robustness_test': True,
        'image_occlusion': False,
        'pointcloud_sparsify': False,
        'pointcloud_noise': True,
        'noise_std': 0.03  # 3cm标准差
    }
    
    # 4. 轻度综合测试
    configs['mild_combined'] = {
        'robustness_test': True,
        'image_occlusion': True,
        'occlusion_ratio': 0.1,
        'occlusion_num_patches': 2,
        'pointcloud_sparsify': True,
        'sparsify_ratio': 0.7,
        'pointcloud_noise': True,
        'noise_std': 0.01
    }
    
    # 5. 严重综合测试
    configs['severe_combined'] = {
        'robustness_test': True,
        'image_occlusion': True,
        'occlusion_ratio': 0.25,
        'occlusion_num_patches': 5,
        'pointcloud_sparsify': True,
        'sparsify_ratio': 0.3,
        'pointcloud_noise': True,
        'noise_std': 0.05
    }
    
    return configs


if __name__ == "__main__":
    # 获取预定义配置
    configs = create_robustness_configs()
    
    # 示例1: 正常数据集（无增强）
    # print("=== Normal Dataset ===")
    # normal_dataset = PCImageDataset("data",
    #                                patch_size=1.0, offset=-1.5,
    #                                img_size=256, min_points=100,
    #                                img_subdir="Colmap/images",
    #                                sample_step=0.1,
    #                                prefilter_samples=True,
    #                                robustness_test=False)  # 关闭鲁棒性测试
    # print(f"Normal dataset size: {len(normal_dataset)}")
    
    # 示例2: 图像遮挡测试
    # print("\n=== Image Occlusion Test ===")
    # occlusion_dataset = PCImageDataset("data",
    #                                   patch_size=1.0, offset=-1.5,
    #                                   img_size=256, min_points=100,
    #                                   img_subdir="Colmap/images",
    #                                   sample_step=0.1,
    #                                   prefilter_samples=True,
    #                                   **configs['image_occlusion'])
    # print(f"Occlusion dataset size: {len(occlusion_dataset)}")
    
    # 示例3: 点云稀疏化测试
    # print("\n=== Pointcloud Sparsification Test ===")
    # sparse_dataset = PCImageDataset("data",
    #                                patch_size=1.0, offset=-1.5,
    #                                img_size=256, min_points=70,  # 降低最小点数要求
    #                                img_subdir="Colmap/images",
    #                                sample_step=0.1,
    #                                prefilter_samples=True,
    #                                **configs['pointcloud_sparse'])
    # print(f"Sparse dataset size: {len(sparse_dataset)}")

    # 示例4: 点云噪声测试
    print("\n=== Pointcloud Noise Test ===")
    noise_dataset = PCImageDataset("data",
                                  patch_size=1.0, offset=-1.5,
                                  img_size=256, min_points=100,
                                  img_subdir="Colmap/images",
                                  sample_step=0.1,
                                  prefilter_samples=True,
                                  **configs['pointcloud_noise'])
    print(f"Noise dataset size: {len(noise_dataset)}")
    
    # 示例5: 综合测试
    # print("\n=== Severe Combined Test ===")
    # combined_dataset = PCImageDataset("data",
    #                                  patch_size=1.0, offset=-1.5,
    #                                  img_size=256, min_points=30,  # 更低的最小点数
    #                                  img_subdir="Colmap/images",
    #                                  sample_step=0.1,
    #                                  prefilter_samples=True,
    #                                  **configs['severe_combined'])
    # print(f"Combined dataset size: {len(combined_dataset)}")
    
    # 测试DataLoader
    from torch.utils.data import DataLoader
    
    print("\n=== Testing DataLoader ===")
    dataloader = DataLoader(noise_dataset, batch_size=2, shuffle=False, 
                           num_workers=0, collate_fn=collate_fn)
    
    for i, (pc_batch, img_batch, cost_batch) in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Point Cloud batch: {len(pc_batch)} samples")
        if len(pc_batch) > 0:
            print(f"    Sample 0 points: {pc_batch[0].shape}")
        print(f"  Image batch: {img_batch.shape}")
        print(f"  Cost batch: {cost_batch.shape}")
        break  # 只测试第一个batch