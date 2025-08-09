import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import logging
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from model.unet_model import UNet
from utils.dataset import AI4MarsDataset

def setup_logging(log_dir):
    # Configure logging
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This also outputs logs to console
        ]
    )

def train_net(net, device, image_path, label_path, epochs=40, batch_size=1, lr=0.00001,
               checkpoint_interval=5, log_dir='logs', checkpoint_dir='checkpoints'):
    # Setup logging
    setup_logging(log_dir)
    logging.info("Starting training")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize datasets
    train_dataset = AI4MarsDataset(image_dir=image_path, label_dir=label_path, train=True)
    
    # Split into training and validation sets
    train_indices, val_indices = train_test_split(list(range(len(train_dataset))), test_size=0.2, random_state=42)
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)

    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Define optimizer and loss function
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Initialize metrics
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(epochs):
        # Training phase
        net.train()
        train_loss = 0
        for image, label in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            pred = net(image)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)
                pred = net(image)
                loss = criterion(pred, label)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))

        # Save checkpoint at intervals
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # Log results
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Plot learning curves and save directly
        plt.figure()
        epochs_range = range(1, len(train_losses) + 1)  # 动态更新为train_losses和val_losses的长度
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(log_dir, 'learning_curve.png'))  # Save plot directly
        plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load network
    net = UNet(n_channels=1, n_classes=5)
    net.to(device=device)
    # Set data paths and train
    image_path = "ai4mars-dataset-merged-0.1/msl/images/edr"
    label_path = "ai4mars-dataset-merged-0.1/msl/labels/train"
    train_net(net, device, image_path, label_path, epochs=100, batch_size=4)
