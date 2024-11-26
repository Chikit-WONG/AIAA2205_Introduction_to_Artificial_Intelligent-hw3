import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MyDataset
from models import VideoResNet

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据增强
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# 加载测试数据集
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

# 从 `test_for_student.csv` 中读取视频 ID
video_ids = pd.read_csv("./data/test_for_student.csv")["Id"].tolist()

with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        results.extend(predicted.cpu().numpy())

# 保存结果
result_df = pd.DataFrame({"Id": video_ids, "Category": results})
result_df.to_csv("./results/result_x3d_s.csv", index=False)
print("Results saved to ./results/result_x3d_s.csv")
