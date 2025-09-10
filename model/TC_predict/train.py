# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms

from TCPredictionNet import TCPredictionNet
from dataset import PCImageDataset, collate_fn

class MSELossIgnoreNaN(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input, target):
        mask = ~torch.isnan(target)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=input.device)
        
        valid_input = input[mask]
        valid_target = target[mask]
        
        loss = (valid_input - valid_target) ** 2
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ====== 数据集 ======
class RandomPCImageDataset(Dataset):
    def __init__(self, n_samples=100, points=5000, img_size=256,
                 pc_range=(-50, 50, -50, 50, -5, 3), bev_size=(200, 200)):
        self.n = n_samples
        self.points = points
        self.pc_range = pc_range
        self.img_size = img_size
        self.bev_H, self.bev_W = bev_size

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # === 1. 随机点云 (x,y,z) ===
        low = [self.pc_range[0], self.pc_range[1], self.pc_range[2]]
        high = [self.pc_range[3], self.pc_range[4], self.pc_range[5]]
        pc = np.random.uniform(low=low, high=high,
                               size=(self.points, 3)).astype("float32")
        pc = torch.tensor(pc)  # (N,3)

        # === 2. 随机图像 ===
        img = np.random.randint(0, 256, size=(self.img_size, self.img_size, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        img = self.img_transform(img)

        # === 3. 随机 ground truth cost map ===
        cost = torch.rand(1, self.bev_H, self.bev_W)

        return pc, img, cost


# ====== 训练脚本 ======
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 模型 ===
    model = TCPredictionNet(
        pc_range=(0, 0, -5, 1, 1, 5),
        bev_H=100, bev_W=100, bev_channels=8,
        max_points_per_pillar=100,
        use_relative_xyz=True, use_rgb=True,
        fpn_out_channels=128,
        use_modulation=True, modulation_dim=384,
        dinov3_repo="model/dinov3",
        dinov3_weight="model/dinov3/weight/weight.pth"
    ).to(device)

    # === 数据 ===
    dataset = PCImageDataset("data",
                             patch_size=1.0, offset=-1.5,
                             img_size=1024, min_points=100,
                             img_subdir="Colmap/images",
                             sample_step=0.1,
                             prefilter_samples=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # === 优化器 & 损失 ===
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = MSELossIgnoreNaN()

    # === 训练循环 ===
    for epoch in range(5):
        model.train()
        epoch_loss = 0
        for pcs, imgs, costs in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            pcs = [pc.to(device) for pc in pcs]     # list of (N,3)
            imgs = imgs.to(device)                 # (B,3,H,W)
            costs = costs.to(device)               # (B,1,H,W)

            # === 打印一次设备信息（只打印第一个batch即可） ===
            if epoch == 0 and epoch_loss == 0:
                print("模型所在 device:", next(model.parameters()).device)
                print("点云 device:", pcs[0].device)
                print("图像 device:", imgs.device)
                print("cost map device:", costs.device)

            optimizer.zero_grad()
            preds = model(pcs, imgs)               # (B,1,H,W)
            loss = criterion(preds, costs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {epoch_loss/len(dataloader):.4f}")


if __name__ == "__main__":
    main()
