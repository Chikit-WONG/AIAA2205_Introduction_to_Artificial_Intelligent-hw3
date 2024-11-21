import os
import pandas as pd
from PIL import Image
from dataset import MyDataset
from models import VideoResNet
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义数据增强和转换
transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# 加载测试数据集
test_dataset = MyDataset(
    "./data/hw3_16fpv",
    "./data/test_for_student.csv",
    stage="test",
    ratio=0.2,
    transform=transforms,
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Length of test loader: {len(test_loader)}")

# 加载模型
model = VideoResNet(num_classes=10).to(device)
model.load_state_dict(torch.load("./models/ResNet18_last.pth"))
print("Model loaded successfully.")

# 加载视频 ID
fread = pd.read_csv("./data/test_for_student.csv")
video_ids = fread["Id"].tolist()

# 验证阶段
model.eval()
result = []
with torch.no_grad():
    for data in tqdm(test_loader):
        inputs, _ = data  # 测试阶段无需标签
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        result.extend(predicted.cpu().numpy())

# 确保结果与视频 ID 对齐
assert len(video_ids) == len(result), "Mismatch between video IDs and predictions."

# 保存结果为 CSV 文件
result_df = pd.DataFrame({"Id": video_ids, "Category": result})
output_path = "./results/result_ResNet18_3D.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 创建目录（如果不存在）
result_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
