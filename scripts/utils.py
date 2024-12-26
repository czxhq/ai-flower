import os
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch


class FlowerDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.csv_file = csv_file
        self.transform = transform
        self.data = pd.read_csv(csv_file)

        # 创建标签映射字典
        self.label_to_index = {label: idx for idx, label in enumerate(self.data['label'].unique())}
        print(self.label_to_index)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, f"{self.data.iloc[idx, 0]}.jpg")  # 根据 id 加载图像
        image = Image.open(img_name).convert('RGB')
        label_str = self.data.iloc[idx, 1]  # 获取字符串标签
        label = ''
        if self.label_to_index.keys().__contains__(label_str):
            label = self.label_to_index[label_str]  # 映射到整数标签

        if self.transform:
            image = self.transform(image)

        # 转换为 Tensor
        if (label != ''):
            label = torch.tensor(label, dtype=torch.long)

        return image, label


def get_data_loaders(train_csv, test_csv, train_image_dir, test_image_dir, batch_size=32):
    # 数据增强和预处理
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform1 = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 创建训练集和测试集
    train_dataset = FlowerDataset(train_image_dir, train_csv, transform=transform1)
    test_dataset = FlowerDataset(test_image_dir, test_csv, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
