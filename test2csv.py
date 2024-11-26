import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MyDataset
from models import VideoResNet

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

test_dataset = MyDataset(
    root="./data/hw3_16fpv",
    csv_file="./data/test_for_student.csv",
    stage="test",
    transform=transform,
)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 加载模型
model = VideoResNet(num_classes=10).to(device)
model.load_state_dict(torch.load("./models/best_model.pth"))

# 推理
model.eval()
results = []
with torch.no_grad():
    for inputs, _ in test_loader:
        slow_path, fast_path = inputs
        slow_path, fast_path = slow_path.to(device), fast_path.to(device)
        outputs = model([slow_path, fast_path])
        _, predicted = torch.max(outputs, 1)
        results.extend(predicted.cpu().numpy())
