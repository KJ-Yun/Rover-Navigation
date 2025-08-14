from datetime import datetime
import logging
import os
import torch
from model.unet_model import UNet
from utils.dataset import AI4MarsDataset
import utils.metric as metric
import numpy as np
from PIL import Image

def setup_logging(log_dir):
    # Configure logging
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'test_log_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.txt')
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def colorize_mask(mask, ori_img):
    """将语义 mask 映射为彩色图像"""
    # 你可以修改这里的颜色映射
    colors = np.array([
        [255, 0, 0], # soil (red)
        [255, 255, 255], # bedrock (white)
        [255, 255, 0], # sand (yellow)
        [0, 255, 0], # big rock (green)
        [0, 0, 0], # unknown (black)
    ], dtype=np.uint8)

    color_mask = colors[mask]
    color_mask = Image.fromarray(color_mask).convert('RGBA')
    ori_img = ori_img.convert('RGBA')
    alpha = 120
    color_mask.putalpha(alpha)
    mask_img = Image.alpha_composite(ori_img, color_mask)

    return mask_img

if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # setup_logging('logs')
    net = UNet(n_channels=1, n_classes=5).to(device)
    net.load_state_dict(torch.load('model/UNet/checkpoints/best_model.pth', map_location=device, weights_only=True))
    image_dir="data/ai4mars-dataset-merged-0.1/msl/images/edr"
    test_label_dir = 'data/ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min3-100agree'
    test_dataset = AI4MarsDataset(image_dir=image_dir, label_dir=test_label_dir, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    # 测试模式
    net.eval()
    # 遍历素有图片
    test_metric = torch.zeros(4)
    batch_count = 0
    logging.info('start to test!')
    output_dir = 'output/ai4mars-unet-inference'
    for image,label in test_loader:
        # 预测
        image = image.to(device = device, dtype = torch.float32)
        label = label.to(device = device, dtype = torch.long)
        # pred = net(image)[:,:-1,:,:]
        pred = net(image)
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        label[label == 255] = 4
        print(f'pred shape: {pred.shape}, label shape: {label.shape}')

        # save 3 images which are pred, label and original image
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pred_image = pred.squeeze(0).cpu().numpy().astype(np.uint8)
        label_image = label.squeeze(0).cpu().numpy().astype(np.uint8)
        original_image = image.squeeze(0).squeeze(0).cpu().numpy() * 255
        original_image = original_image.astype(np.uint8)

        print(f'pred_image shape: {pred_image.shape}, label_image shape: {label_image.shape}, original_image shape: {original_image.shape}')

        pred_image_path = os.path.join(output_dir, f'{batch_count}_pred.png')
        label_image_path = os.path.join(output_dir, f'{batch_count}_label.png')
        original_image_path = os.path.join(output_dir, f'{batch_count}_original.png')

        original_image = Image.fromarray(original_image)
        pred_image = colorize_mask(pred_image, original_image)
        label_image = colorize_mask(label_image, original_image)
        
        pred_image.save(pred_image_path)
        label_image.save(label_image_path)
        original_image.save(original_image_path)
        batch_count += 1

        # batch_metric = metric.all_metric(pred,label,ignore_index=255)
        # logging.info(f'the batch metric is: {batch_metric}')
        # if torch.isnan(batch_metric).any():
        #     continue
        # else:
        #     test_metric += batch_metric
        #     batch_count += 1
        # logging.debug(f'test_metric in this batch is: {test_metric}')
        # logging.debug(f'batch_count is: {batch_count}')

    # test_metric = test_metric/batch_count
    # logging.info(f'pixel_accuracy: {test_metric[0].item()}')
    # logging.info(f'mean_pixel_accuracy: {test_metric[1].item()}')
    # logging.info(f'mean_iou: {test_metric[2].item()}')
    # logging.info(f'frequency_weighted_iou: {test_metric[3].item()}')
        