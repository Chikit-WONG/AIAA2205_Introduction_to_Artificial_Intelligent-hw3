import torch
import torch.nn as nn
from models import VideoResNet
from dataset import VideoDataset
from torch.utils.data import DataLoader

# 使用 CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据
train_dataset = VideoDataset(
    root="./data/hw3_16fpv", csv_file="./data/trainval.csv", stage="train", ratio=0.2
)
val_dataset = VideoDataset(
    root="./data/hw3_16fpv", csv_file="./data/trainval.csv", stage="val", ratio=0.2
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# 初始化模型
model = VideoResNet(num_classes=10)
model.to(device)

# 定义优化器与损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(1):
    model.train()
    for inputs, labels in train_loader:
        slow_path, fast_path = inputs
        slow_path, fast_path, labels = (
            slow_path.to(device),
            fast_path.to(device),
            labels.to(device),
        )
        print(f"Slow path shape: {slow_path.shape}, Fast path shape: {fast_path.shape}")

        # 前向传播
        outputs = model([slow_path, fast_path])
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: Loss = {loss.item()}")
