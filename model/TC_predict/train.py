# tc_prediction_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# === FPN 模块 ===
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, features):
        # 自上而下融合
        last_inner = self.lateral_convs[-1](features[-1])
        results = [self.output_convs[-1](last_inner)]

        for i in range(len(features) - 2, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            inner_top_down = F.interpolate(last_inner, size=lateral.shape[-2:], mode="nearest")
            last_inner = lateral + inner_top_down
            results.insert(0, self.output_convs[i](last_inner))

        return results[0]  # 仅使用最高分辨率输出


# === TC预测网络主结构 ===
class TCPredictionNet(nn.Module):
    def __init__(self, input_channels=8, fpn_out_channels=128):
        super(TCPredictionNet, self).__init__()

        # 使用 torchvision 的 resnet34 作为 backbone
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.initial = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )
        self.layer1 = base_model.layer1  # 输出通道: 256
        self.layer2 = base_model.layer2  # 输出通道: 512
        self.layer3 = base_model.layer3  # 输出通道: 1024
        self.layer4 = base_model.layer4  # 输出通道: 2048

        # 修改输入通道（适配多通道输入）
        self.initial[0] = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # FPN: 多尺度融合
        self.fpn = FPN(in_channels_list=[256, 512, 1024, 2048], out_channels=fpn_out_channels)

        # Head: TC值回归
        self.head = nn.Sequential(
            nn.Conv2d(fpn_out_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),  # 输出单通道TC图
            nn.Sigmoid()  # TC值范围归一化 [0, 1]
        )

    def forward(self, x):
        input_size = x.shape[2:]  # 保存原始输入尺寸

        x = self.initial(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        fpn_out = self.fpn([f1, f2, f3, f4])
        out = self.head(fpn_out)

        # 将输出恢复至原始输入分辨率
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out


# === 训练脚本模板 ===
if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from tqdm import tqdm


    class DummyBEVDataset(Dataset):
        def __init__(self, n_samples=100):
            self.n = n_samples
            self.transform = transforms.ToTensor()

        def __getitem__(self, idx):
            x = torch.rand(8, 256, 256)  # 模拟BEV输入
            y = torch.rand(1, 256, 256)  # TC ground truth
            return x, y

        def __len__(self):
            return self.n


    # 模型、数据、优化器
    model = TCPredictionNet(input_channels=8)
    dataset = DummyBEVDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # === 训练循环 ===
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")
