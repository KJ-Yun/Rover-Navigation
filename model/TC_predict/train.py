# tc_prediction_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image


# === FPN 模块 ===
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, use_modulation=False, modulation_dim=384):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        # 是否启用线性特征调制
        self.use_modulation = use_modulation
        if use_modulation:
            # 对 p3、p4 两个层分别做 FiLM 调制 (通道数=out_channels)
            self.modulation_p3 = nn.Linear(modulation_dim, out_channels * 2)
            self.modulation_p4 = nn.Linear(modulation_dim, out_channels * 2)

    def forward(self, features, seq_code=None):
        # 自上而下融合
        last_inner = self.lateral_convs[-1](features[-1])
        results = [self.output_convs[-1](last_inner)]

        for i in range(len(features) - 2, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            inner_top_down = F.interpolate(last_inner, size=lateral.shape[-2:], mode="nearest")
            last_inner = lateral + inner_top_down
            out = self.output_convs[i](last_inner)

            if self.use_modulation and seq_code is not None:
                if i == 2:  # p4
                    gamma, beta = self.modulation_p4(seq_code).chunk(2, dim=-1)
                    gamma, beta = gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
                    print(gamma.shape, beta.shape, out.shape)
                    out = gamma * out + beta
                elif i == 1:  # p3
                    gamma, beta = self.modulation_p3(seq_code).chunk(2, dim=-1)
                    gamma, beta = gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
                    out = gamma * out + beta

            results.insert(0, out)

        return results[0]  # 仅使用最高分辨率输出


# === TC预测网络主结构 ===
class TCPredictionNet(nn.Module):
    def __init__(self, input_channels=8, fpn_out_channels=128, use_modulation=False, modulation_dim=384,
                 dinov3_repo="/mnt/d/Rover-Navigation/model/dinov3",
                 dinov3_weight="/mnt/d/Rover-Navigation/model/dinov3/weight/weight.pth"):
        super(TCPredictionNet, self).__init__()

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

        # 修改输入通道
        self.initial[0] = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.fpn = FPN(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=fpn_out_channels,
            use_modulation=use_modulation,
            modulation_dim=modulation_dim
        )

        # Head: TC值回归
        self.head = nn.Sequential(
            nn.Conv2d(fpn_out_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        if use_modulation:
            self.dino_size = 384
            self.dinov3 = torch.hub.load(dinov3_repo, 'dinov3_vits16',
                                         source='local', weights=dinov3_weight).eval()

    

    def forward(self, x, image=None):
        input_size = x.shape[2:]

        seq_code = None
        if hasattr(self, "dinov3") and image is not None:
            seq_code = self.dinov3(image)  # (N,384)

        x = self.initial(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        fpn_out = self.fpn([f1, f2, f3, f4], seq_code=seq_code)
        out = self.head(fpn_out)

        # 上采样回原图分辨率
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out


# === 训练脚本模板 ===
if __name__ == "__main__":

    class DummyBEVDataset(Dataset):
        def __init__(self, n_samples=100, image_size=1024):
            self.n = n_samples
            self.transform = self.make_transform(image_size)

        def make_transform(self, resize_size: int | list[int] = 768):
            to_tensor = transforms.ToTensor()
            resize = transforms.Resize((resize_size, resize_size), antialias=True)
            normalize = transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
            return transforms.Compose([to_tensor, resize, normalize])

        def __getitem__(self, idx):
            bev = torch.rand(8, 256, 256)  # 模拟BEV输入
            image = torch.rand(3, 1080, 1920)  # 模拟图像输入
            image = Image.fromarray((image.numpy() * 255).astype('uint8').transpose(1, 2, 0))  # 转为PIL图像
            image = self.transform(image)
            cost = torch.rand(1, 256, 256)  # TC ground truth
            return bev, image, cost

        def __len__(self):
            return self.n


    # 模型、数据、优化器
    # model = TCPredictionNet(input_channels=8)
    model = TCPredictionNet(input_channels=8, fpn_out_channels=128, use_modulation=True, modulation_dim=384)
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
        for bev, image, cost in tqdm(dataloader):
            bev, image, cost = bev.to(device), image.to(device), cost.to(device)
            optimizer.zero_grad()
            pred = model(bev, image)
            loss = criterion(pred, cost)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")
