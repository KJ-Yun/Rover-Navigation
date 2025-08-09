import random
import torch
import cv2
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


class AI4MarsDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=transforms.Compose([transforms.ToTensor()]), train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.train = train
        # 获取图像和标签文件的路径
        if train:
            self.image_files = [
                f for f in os.listdir(self.image_dir) if
                f.endswith('.JPG') and os.path.exists(os.path.join(self.label_dir, f.replace('.JPG', '.png')))
            ]
        else:
            self.image_files = [
                f for f in os.listdir(self.image_dir) if
                f.endswith('.JPG') and os.path.exists(os.path.join(self.label_dir, f.replace('.JPG', '_merged.png')))
            ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # 根据index读取图片
        image_file = self.image_files[index]

        if self.train:
            label_file = image_file.replace('.JPG', '.png')
        else:
            label_file = image_file.replace('.JPG', '_merged.png')

        # 读取训练图片和标签图片
        image = cv2.imread(os.path.join(self.image_dir, image_file))
        label = cv2.imread(os.path.join(self.label_dir, label_file))

        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        
        if self.transform:
            image = self.transform(image)
            # label = self.transform(label)
            label = torch.from_numpy(label)

        # 如果标签全为255，随机选择一个替代标签
        while torch.all(label == 255) and self.train: # 检查标签是否全为255
            random_index = random.randint(0, len(self.image_files) - 1)  # 随机选择一个索引
            image_file = self.image_files[random_index]
            if self.train:
                label_file = image_file.replace('.JPG', '.png')

            # 读取替代的训练图片和标签图片
            image = cv2.imread(os.path.join(self.image_dir, image_file))
            label = cv2.imread(os.path.join(self.label_dir, label_file))

            # 将数据转为单通道的图片
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

            if self.transform:
                image = self.transform(image)
                # label = self.transform(label)
                label = torch.from_numpy(label)

        return image, label


if __name__ == "__main__":
    print(os.getcwd())
    image_dir = './ai4mars-dataset-merged-0.1/msl/images/edr'
    train_label_dir = './ai4mars-dataset-merged-0.1/msl/labels/train'
    test_label_dir = './ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min3-100agree'
    # 图像预处理
    # transform = transforms.Compose([transforms.ToTensor()])
    # 创建训练数据集
    train_dataset = AI4MarsDataset(image_dir=image_dir, label_dir=train_label_dir, train=True)
    print("Train Data Size:", len(train_dataset))
    # 创建测试数据集
    test_dataset = AI4MarsDataset(image_dir=image_dir, label_dir=test_label_dir, train=False)
    print("Test Data Size:", len(test_dataset))

    # 将训练数据分为训练集和开发集
    train_indices, val_indices = train_test_split(list(range(len(train_dataset))), test_size=0.2, random_state=42)
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=2, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=2, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)

    batch_num = 0
    for images, labels in test_loader:
        print(images.shape)
        print(images.dtype)
        print(images)
        break
        batch_num+=1
    print('batch num is: ',batch_num)
    # 检查数据
    # print('start to check')
    # batch_num = 0
    # for images, labels in train_loader:
    #     batch_num += 1
    #     if torch.all(labels == 255):
    #         print('all label == 255, need to be dropped')
    #         break
    # print('batch number should be: ',len(train_loader))
    # print('total batch number: ', batch_num)
