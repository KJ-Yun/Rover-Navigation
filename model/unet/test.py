from datetime import datetime
import logging
import os
import torch
from model.unet_model import UNet
from utils.dataset import AI4MarsDataset
import utils.metric as metric
import numpy as np

def setup_logging(log_dir):
    # Configure logging
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'test_log_'+datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+'.txt')
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    setup_logging('logs')
    net = UNet(n_channels=1, n_classes=5).to(device)
    net.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device, weights_only=True))
    image_dir="ai4mars-dataset-merged-0.1/msl/images/edr"
    test_label_dir = 'ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min3-100agree'
    test_dataset = AI4MarsDataset(image_dir=image_dir, label_dir=test_label_dir, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    # 测试模式
    net.eval()
    # 遍历素有图片
    test_metric = torch.zeros(4)
    batch_count = 0
    logging.info('start to test!')
    for image,label in test_loader:
        # 预测
        image = image.to(device = device, dtype = torch.float32)
        label = label.to(device = device, dtype = torch.long)
        pred = net(image)[:,:-1,:,:]
        batch_metric = metric.all_metric(pred,label,ignore_index=255)
        logging.info(f'the batch metric is: {batch_metric}')
        if torch.isnan(batch_metric).any():
            continue
        else:
            test_metric += batch_metric
            batch_count += 1
        logging.debug(f'test_metric in this batch is: {test_metric}')
        logging.debug(f'batch_count is: {batch_count}')

    test_metric = test_metric/batch_count
    logging.info(f'pixel_accuracy: {test_metric[0].item()}')
    logging.info(f'mean_pixel_accuracy: {test_metric[1].item()}')
    logging.info(f'mean_iou: {test_metric[2].item()}')
    logging.info(f'frequency_weighted_iou: {test_metric[3].item()}')
        