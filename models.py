import torch
import torch.nn as nn
from pytorchvideo.models.slowfast import create_slowfast


class VideoResNet(nn.Module):
    def __init__(self, num_classes):
        super(VideoResNet, self).__init__()
        # 使用 create_slowfast 创建 SlowFast 模型
        self.model = create_slowfast(
            slowfast_channel_reduction_ratio=(8,),  # SlowFast 模型通道比例
            slowfast_conv_channel_fusion_ratio=2,  # Slow 和 Fast 通道融合比例
            input_channels=(3, 3),  # Slow 和 Fast 的输入通道
            model_depth=50,  # 使用 ResNet-50
            model_num_class=num_classes,  # 分类数
            head_pool_kernel_sizes=((8, 7, 7), (32, 7, 7)),  # 池化核大小
        )

        # 调整 Fast Path 的通道数
        self.channel_adjustment = nn.Conv3d(
            in_channels=64, out_channels=80, kernel_size=1
        )

    def forward(self, x):
        slow_path, fast_path = x

        # 调整 Fast Path 的通道数
        fast_path = self.channel_adjustment(fast_path)

        return self.model([slow_path, fast_path])
