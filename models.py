import torch
import torch.nn as nn


class VideoResNet(nn.Module):
    def __init__(self, num_classes):
        super(VideoResNet, self).__init__()
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "x3d_s",  # 使用轻量级 X3D 模型
            pretrained=True,
        )

        # 替换分类头以适配任务的类别数
        self.model.blocks[5].proj = nn.Linear(
            self.model.blocks[5].proj.in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)
