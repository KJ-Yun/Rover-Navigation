from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
import os
import logging
import torch.multiprocessing as mp
import torch.distributed as dist
from TCPredictionNet import TCPredictionNet
from dataset import PCImageDataset, collate_fn
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm

class HuberLossIgnoreNaN(nn.Module):
    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__()
        self.base_loss = nn.HuberLoss(delta=delta, reduction='none')  # 原生 HuberLoss
        self.reduction = reduction

    def forward(self, input, target):
        # 保证 target 在相同设备和 dtype
        target = target.to(input.device, dtype=input.dtype)

        # mask: 只保留非 NaN 的位置
        mask = ~torch.isnan(target)

        # 如果全是 NaN，返回 0，避免 NaN 反传
        if mask.sum() == 0:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype, requires_grad=True)
        if torch.any(torch.isnan(input)):
            raise ValueError("Input contains NaN values.")

        # 计算原始 loss（逐元素）
        loss = self.base_loss(input, target)

        # 应用 mask
        loss = loss[mask]

        # reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

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

def ddp_setup():
    # Initialize DDP
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

def setup_logging(log_dir, rank):
    # Configure logging
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_log_'+datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+'.txt')
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARN,  # Only main process logs at INFO
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() if rank == 0 else logging.NullHandler()  # Only main process logs to console
        ]
    )

def train_net(epochs=40, batch_size=2, lr=0.00001,
              checkpoint_interval=5, log_dir='logs', checkpoint_dir='checkpoints'):
    
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f'cuda:{local_rank}')

    # Setup logging
    if local_rank == 0:
        setup_logging(log_dir, local_rank)
        logging.info(f"Starting training on {world_size} GPUs")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize dataset and sampler
    train_dataset = PCImageDataset("data",
                             patch_size=1.0, offset=-1.5,
                             img_size=1024, min_points=100,
                             img_subdir="Colmap/images",
                             sample_step=0.1,
                             prefilter_samples=True)
        
    train_indices, val_indices = train_test_split(list(range(len(train_dataset))), test_size=0.1, random_state=42)
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=local_rank, shuffle=False)
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, sampler=val_sampler, collate_fn=collate_fn)

    # Load model and wrap with DDP
    net = TCPredictionNet(
        pc_range=(0, 0, -5, 1, 1, 5),
        bev_H=100, bev_W=100, bev_channels=8,
        max_points_per_pillar=100,
        use_relative_xyz=True, use_rgb=True,
        fpn_out_channels=128,
        use_modulation=False, modulation_dim=384,
        dinov3_repo="model/dinov3",
        dinov3_weight="model/dinov3/weight/weight.pth"
    ).to(device)

    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DDP(net, device_ids=[local_rank], find_unused_parameters=True)

    # Define optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay=1e-8)
    criterion = MSELossIgnoreNaN()

    # Initialize metrics
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # Shuffle dataset every epoch

        net.train()
        train_loss = 0
        for pcs, imgs, costs in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            pcs = [pc.to(device) for pc in pcs]     # list of (N,3)
            imgs = imgs.to(device)                 # (B,3,H,W)
            costs = costs.to(device)               # (B,1,H,W)

            if epoch == 0 and train_loss == 0:
                print("模型所在 device:", next(net.parameters()).device)
                print("点云 device:", pcs[0].device)
                print("图像 device:", imgs.device)
                print("cost map device:", costs.device)

            optimizer.zero_grad()
            preds = net(pcs, imgs)               # (B,1,H,W)
            loss = criterion(preds, costs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
   
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for pcs, imgs, costs in tqdm(val_loader, desc=f"Validation {epoch+1}"):
                pcs = [pc.to(device) for pc in pcs]     # list of (N,3)
                imgs = imgs.to(device)                 # (B,3,H,W)
                costs = costs.to(device)               # (B,1,H,W)
                pred = net(pcs, imgs)
                loss = criterion(pred, costs)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        # 汇总训练和验证损失
        train_loss_tensor = torch.tensor(avg_train_loss).to(device)
        val_loss_tensor = torch.tensor(avg_val_loss).to(device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        global_avg_train_loss = train_loss_tensor.item() / dist.get_world_size()
        global_avg_val_loss = val_loss_tensor.item() / dist.get_world_size()

        # 仅在主进程中执行以下操作
        if local_rank == 0:
            # 保存最优模型
            if global_avg_val_loss < best_val_loss:
                best_val_loss = global_avg_val_loss
                torch.save(net.module.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))

            # 定期保存检查点
            if (epoch + 1) % checkpoint_interval == 0:
                torch.save(net.module.state_dict(), os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

            # 记录训练和验证损失
            logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {global_avg_train_loss:.4f}, Val Loss: {global_avg_val_loss:.4f}")

            # 更新损失记录列表
            train_losses.append(global_avg_train_loss)
            val_losses.append(global_avg_val_loss)

            # 绘制学习曲线
            plt.figure()
            epochs_range = range(1, len(train_losses) + 1)
            plt.plot(epochs_range, train_losses, label='Train Loss')
            plt.plot(epochs_range, val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.savefig(os.path.join(log_dir, 'learning_curve.png'))
            plt.close()

@record
def main():
    ddp_setup()
    train_net(epochs=30, batch_size=8,checkpoint_interval=1, log_dir='logs/no_modulation', checkpoint_dir='checkpoints/no_modulation')
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
