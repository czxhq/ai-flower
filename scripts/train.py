import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score
from utils import get_data_loaders
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet18_Weights

# 设置超参数
batch_size = 6
epochs = 15
learning_rate = 0.0001
num_classes = 17  # 17种花

# 加载数据
train_csv = '../train/train.csv'
test_csv = '../test/submission.csv'
train_image_dir = '../train/images'
test_image_dir = '../test/images'

# 获取训练集和测试集的 DataLoader
train_loader, test_loader = get_data_loaders(
    train_csv, test_csv, train_image_dir, test_image_dir, batch_size
)

# 构建模型
# 使用预训练的 ResNet18
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # 修改全连接层以适应17分类

# 使用GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", torch.cuda.get_device_name(0))
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# 训练过程
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # 将数据移动到指定设备
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100 * correct / total
    print(
        f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, "
        f"Accuracy: {train_accuracy:.2f}%"
    )

    # 更新学习率
    scheduler.step()

# 保存训练好的模型
torch.save(model.state_dict(), '../model/flower_classification.pth')