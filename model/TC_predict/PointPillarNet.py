import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class PFNLayer(nn.Module):
    """
    单层 PFN：逐点 MLP -> 对 pillar 内点做 max-pool 得到 pillar 特征
    in_channels: 输入逐点特征维度
    out_channels: 输出 pillar 特征维度（也是 scatter 后的 C）
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, pts_feats):
        """
        pts_feats: (P, T, Fin)  P为pillar数, T为pillar内采样点数(定长), Fin为逐点特征维度
        return: (P, Fout)
        """
        P, T, Fin = pts_feats.shape
        x = self.linear(pts_feats.reshape(P * T, Fin))              # (P*T, Fout)
        x = self.bn(x)                                               # BN 在点级别
        x = self.act(x)
        x = x.reshape(P, T, -1)                                      # (P, T, Fout)
        x = torch.amax(x, dim=1)                                     # MaxPool over points -> (P, Fout)
        return x


class PointPillarsEncoder(nn.Module):
    """
    仅包含：voxelization(按H/W栅格化到pillar)、Pillar FeatureNet(PFN)、Scatter到BEV
    - 输入：list[Tensor(Ni, Cin)]，每个 batch 元素一个点云，Cin=3 或 6
      * (x,y,z) 或 (r,g,b,x,y,z)
      * RGB 值建议在 [0,1]，若是 [0,255] 会自动归一化
    - 输出：Tensor (B, C, H, W)，可直接接入 TCPredictionNet
    """
    def __init__(
        self,
        x_range, y_range, z_range,
        H, W,
        max_points_per_pillar=100,
        pfn_out_channels=8,          # 你 TCPredictionNet 默认输入是8，这里默认对齐
        use_relative_xyz=True,
        use_rgb=True,                # 若输入为(N,3)时自动忽略
    ):
        super().__init__()
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        self.H, self.W = H, W
        self.voxel_x = (self.x_max - self.x_min) / float(W)
        self.voxel_y = (self.y_max - self.y_min) / float(H)
        self.z_center = 0.5 * (self.z_min + self.z_max)

        self.max_points_per_pillar = max_points_per_pillar
        self.use_relative_xyz = use_relative_xyz
        self.use_rgb = use_rgb

        # 逐点特征维度 = [x,y,z] + [r,g,b?] + [dx,dy,dz?] + [1(占位/存在性)]
        fin = 3  # x,y,z
        # r,g,b
        fin += 3 if self.use_rgb else 0
        # 相对位移
        fin += 3 if self.use_relative_xyz else 0
        # 常数1（可作为存在性/稳定项）
        fin += 1

        self.pfn = PFNLayer(fin, pfn_out_channels)
        self.out_channels = pfn_out_channels

    @torch.no_grad()
    def _preprocess_batch(self, all_points, batch_idx, device="cpu"):
        """
        把多个 batch 的点云拼接后进行 pillar 化
        输入:
            all_points: (∑Ni, Cin) 点云 (x,y,z[,r,g,b])
            batch_idx:  (∑Ni,) 每个点属于哪个 batch
        输出:
            pillar_pts_feats: (P, T, Fin) 每个 pillar 内点的特征 (零填充到 T=max_points_per_pillar)
            ji: (P, 2) 每个 pillar 的 (j,i) 网格索引
            pillar_batch: (P,) 每个 pillar 属于哪个 batch
        """
        Cin = all_points.shape[1]
        assert Cin in [3, 6], "点云必须是 (x,y,z) 或 (r,g,b,x,y,z)"

        # 解析坐标和RGB
        if Cin == 3:
            x, y, z = all_points[:, 0], all_points[:, 1], all_points[:, 2]
            rgb = None
        else:
            r, g, b, x, y, z = all_points[:, 3], all_points[:, 4], all_points[:, 5], all_points[:, 0], all_points[:, 1], all_points[:, 2]
            # RGB归一化到[0,1]
            rgb = torch.stack([r, g, b], dim=1)  # (N, 3)
            if rgb.max() > 1.1:  # 判断是否需要归一化
                rgb = rgb / 255.0

        i = ((x - self.x_min) / self.voxel_x).long()
        j = ((y - self.y_min) / self.voxel_y).long()

        # 过滤边界外的点
        mask = (i >= 0) & (i < self.W) & (j >= 0) & (j < self.H) & (z >= self.z_min) & (z < self.z_max)
        i, j, x, y, z, batch_idx = i[mask], j[mask], x[mask], y[mask], z[mask], batch_idx[mask]
        if rgb is not None:
            rgb = rgb[mask]

        if x.shape[0] == 0:
            # 计算正确的特征维度
            fin = 3  # x,y,z
            fin += 3 if self.use_rgb and Cin == 6 else 0
            fin += 3 if self.use_relative_xyz else 0
            fin += 1  # 常数项
            
            return (torch.zeros((0, self.max_points_per_pillar, fin), device=device),
                    torch.zeros((0, 2), dtype=torch.long, device=device),
                    torch.zeros((0,), dtype=torch.long, device=device))

        # 找出每个 (batch, j, i) 对应的 pillar
        pillar_keys = torch.stack([batch_idx, j, i], dim=1)  # 注意：这里改为(batch, j, i)
        uniq_keys, inverse = torch.unique(pillar_keys, dim=0, return_inverse=True) # point -> pillar 映射

        P = uniq_keys.shape[0]  # 总 pillar 数
        T = self.max_points_per_pillar

        # 计算pillar中心用于相对位移
        pillar_centers = {}
        for p in range(P):
            batch_p, j_p, i_p = uniq_keys[p]
            # 计算pillar中心坐标
            center_x = self.x_min + (i_p + 0.5) * self.voxel_x
            center_y = self.y_min + (j_p + 0.5) * self.voxel_y
            center_z = self.z_center
            pillar_centers[p] = (center_x, center_y, center_z)

        # 构建特征向量
        N = x.shape[0]
        fin = 3  # x,y,z base
        fin += 3 if self.use_rgb and rgb is not None else 0
        fin += 3 if self.use_relative_xyz else 0
        fin += 1  # constant

        point_features = torch.zeros((N, fin), device=device)
        
        # 基础坐标特征
        point_features[:, 0] = x
        point_features[:, 1] = y
        point_features[:, 2] = z
        feat_idx = 3

        # RGB特征
        if self.use_rgb and rgb is not None:
            point_features[:, feat_idx:feat_idx+3] = rgb
            feat_idx += 3

        # 相对位移特征
        if self.use_relative_xyz:
            for n in range(N):
                pid = inverse[n].item()
                center_x, center_y, center_z = pillar_centers[pid]
                point_features[n, feat_idx] = x[n] - center_x
                point_features[n, feat_idx+1] = y[n] - center_y
                point_features[n, feat_idx+2] = z[n] - center_z
            feat_idx += 3

        # 常数项
        point_features[:, feat_idx] = 1.0

        # 初始化pillar特征张量
        pillar_pts_feats = torch.zeros((P, T, fin), device=device)
        counts = torch.zeros((P,), dtype=torch.long, device=device)

        # 把点放进 pillar
        for n in range(N):
            pid = inverse[n].item()
            cnt = counts[pid]
            if cnt < T:  # 只保留前 T 个点
                pillar_pts_feats[pid, cnt] = point_features[n]
                counts[pid] += 1

        # 输出 ji 和 batch
        pillar_batch = uniq_keys[:, 0]  # (P,)
        pillar_ji = uniq_keys[:, 1:]    # (P,2), (j,i)

        return pillar_pts_feats, pillar_ji, pillar_batch


    def forward(self, batch_points):
        """
        batch_points: list[Tensor(Ni, Cin)], Cin=3或6; 形状分别为 (x,y,z) 或 (r,g,b,x,y,z)
        return: bev (B, C, H, W)
        """
        assert isinstance(batch_points, (list, tuple)), "请以 list[Tensor] 形式按 batch 传入点云"
        device = next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else (
            batch_points[0].device if torch.is_tensor(batch_points[0]) else "cpu"
        )
        B = len(batch_points)

        # 拼接所有点云
        all_points = []
        batch_idx = []
        for b, pts in enumerate(batch_points):
            if not torch.is_tensor(pts):
                pts = torch.tensor(pts, dtype=torch.float32, device=device)
            else:
                pts = pts.to(device).float()
            all_points.append(pts)
            batch_idx.append(torch.full((pts.shape[0],), b, device=device, dtype=torch.long))

        all_points = torch.cat(all_points, dim=0)   # (∑Ni, Cin)
        batch_idx = torch.cat(batch_idx, dim=0)     # (∑Ni,)

        # 批量预处理 -> (P, T, Fin), (P, 2), (P,)
        pillar_pts_feats, ji, pillar_batch = self._preprocess_batch(all_points, batch_idx, device=device)
        # pillar_pts_feats: (P, T, Fin)
        # ji: (P, 2) -> 每个pillar的 (j, i) 坐标
        # pillar_batch: (P,) -> 每个pillar属于哪个batch

        if pillar_pts_feats.shape[0] == 0:
            return torch.zeros((B, self.out_channels, self.H, self.W), device=device)

        # PFN 编码 -> (P, C)
        pillar_embeds = self.pfn(pillar_pts_feats)  # (P, C)

        # Scatter -> BEV
        bev = torch.zeros((B, self.out_channels, self.H, self.W), device=device)

        j = ji[:, 0].long()
        i = ji[:, 1].long()
        b = pillar_batch.long()

        # 使用更安全的scatter方式
        for p in range(pillar_embeds.shape[0]):
            bev[b[p], :, j[p], i[p]] = pillar_embeds[p]

        return bev
    
# === 训练脚本模板 ===
if __name__ == "__main__":

    class DummyPCDataset(Dataset):
        def __init__(self, n_samples=100, points=10000, x_range=(-50,50), y_range=(-50,50), z_range=(-5,3)):
            self.n = n_samples
            self.points = points

        def __getitem__(self, idx):
            pc = np.random.uniform(low=[-50,-50,-5], high=[50,50,3], size=(self.points, 3)).astype('float32')  # 模拟点云 (x,y,z)
            pc = torch.tensor(pc)
            return pc

        def __len__(self):
            return self.n


    # 模型、数据、优化器
    # model = TCPredictionNet(input_channels=8)
    model = PointPillarsEncoder(
        x_range=(-50, 50), y_range=(-50, 50), z_range=(-5, 3),
        H=200, W=200,
        max_points_per_pillar=100,
        pfn_out_channels=8,
        use_relative_xyz=True,
        use_rgb=False,
    )
    dataset = DummyPCDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    model.eval()
    for pcs in tqdm(dataloader):
        pcs = [pc.to(device) for pc in pcs]
        with torch.no_grad():
            bev = model(pcs)
        assert bev.shape == (len(pcs), 8, 200, 200)
