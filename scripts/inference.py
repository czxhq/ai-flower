import torch
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import DataLoader
from utils import FlowerDataset
from torchvision.models import ResNet18_Weights

# 加载数值标签 ↔ 字符串类别的映射关系
label_mapping = {
    0: 'iris',
    1: 'buttercup',
    2: 'bluebell',
    3: 'windflower',
    4: 'snowdrop',
    5: 'tigerlily',
    6: 'colts_foot',
    7: 'daffodil',
    8: 'fritillary',
    9: 'pansy',
    10: 'tulip',
    11: 'daisy',
    12: 'dandelion',
    13: 'sunflower',
    14: 'cowslip',
    15: 'crocus',
    16: 'lily_valley'
    # 添加所有类别的映射
}

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, len(label_mapping))  # 确保分类数一致
model.load_state_dict(torch.load('../model/flower_classification.pth'))
model.to(device)
model.eval()

# 处理测试集
test_csv = '../test/submission.csv'
test_image_dir = '../test/images'
test_dataset = FlowerDataset(test_image_dir, test_csv, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 预测并生成提交文件
predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())

# 将数值标签映射回字符串类别
string_predictions = [label_mapping[label] for label in predictions]

# 保存预测结果
submission = pd.read_csv(test_csv)
submission['label'] = string_predictions
submission.to_csv('../submission.csv', index=False)