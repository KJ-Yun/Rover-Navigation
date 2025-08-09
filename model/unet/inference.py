import os

import cv2
import torch
import numpy as np
from PIL import Image
from model.unet_model import UNet
import torchvision.transforms as transforms
from datetime import datetime

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)

def load_image_as_tensor(img_path):
    """读取单通道灰度图，转为模型输入格式"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transform = transforms.Compose([transforms.ToTensor()])
    img_array = transform(img)

    img_tensor = img_array.unsqueeze(0)  # [B,C,H,W]

    return img_tensor, img_tensor.shape[-2:]  # 返回原始尺寸 (W, H)

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

    # 加载模型
    net = UNet(n_channels=1, n_classes=5).to(device)

    net.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device, weights_only=True))
    net.eval()

    # 输入目录
    image_dir = "predict"
    output_dir = os.path.join(os.path.dirname(image_dir), "semantic_seg")
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有 PNG
    with torch.no_grad():
        for fname in os.listdir(image_dir):
            if fname.lower().endswith(".png"):
                img_path = os.path.join(image_dir, fname)

                img_tensor, (w, h) = load_image_as_tensor(img_path)
                img_tensor = img_tensor.to(device)
                print("inferencing", fname)

                # 推理
                pred = net(img_tensor)
                print("pred shape", pred.shape)
                pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                print(pred_mask)
                break
                ori_img = Image.open(img_path)

                # 映射颜色
                mask_image = colorize_mask(pred_mask, ori_img)

                # 保存到输出目录
                save_path = os.path.join(output_dir, fname)
                mask_image.save(save_path)

    print(f"推理完成，结果已保存到: {output_dir}")
