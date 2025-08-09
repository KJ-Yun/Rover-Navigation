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
from model.unet_model import UNet
from utils.dataset import AI4MarsDataset
from torch.distributed.elastic.multiprocessing.errors import record

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

def train_net(image_path, label_path, epochs=40, batch_size=2, lr=0.00001,
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
    train_dataset = AI4MarsDataset(image_dir=image_path, label_dir=label_path, train=True)
    train_indices, val_indices = train_test_split(list(range(len(train_dataset))), test_size=0.2, random_state=42)
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=local_rank, shuffle=False)
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_subset, batch_size=batch_size, sampler=val_sampler)

    # Load model and wrap with DDP
    net = UNet(n_channels=1, n_classes=5).to(device)
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DDP(net, device_ids=[local_rank])

    # Define optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay=1e-8)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Initialize metrics
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  # Shuffle dataset every epoch

        net.train()
        train_loss = 0
        for image,label in train_loader:
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            optimizer.zero_grad()
            pred = net(image)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()         
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for image,label in val_loader:
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)
                pred = net(image)
                loss = criterion(pred, label)
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
    train_net(image_path="ai4mars-dataset-merged-0.1/msl/images/edr",
              label_path="ai4mars-dataset-merged-0.1/msl/labels/train", epochs=100, batch_size=2,checkpoint_interval=10)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
