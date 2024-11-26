import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, root, csv_file, stage="train", ratio=0.2, transform=None):
        self.root = root
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file, header=None, skiprows=1)
        self.stage = stage
        self.ratio = ratio
        self.transforms = transform or transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        self.files = self._split_files()

    def _split_files(self):
        filenames = self.df[0].tolist()
        length = len(filenames)
        train_files = filenames[int(length * self.ratio) :]
        val_files = filenames[: int(length * self.ratio)]
        if self.stage == "train":
            return train_files
        elif self.stage == "val":
            return val_files
        else:
            raise ValueError("Invalid stage: 'train' or 'val' expected.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        vid = self.files[idx]
        video_path = os.path.join(self.root, "hw3_16fpv", f"{vid}.mp4")
        img_list = sorted(os.listdir(video_path))
        imgs = [
            self.transforms(Image.open(os.path.join(video_path, img)).convert("RGB"))
            for img in img_list
        ]

        video_tensor = torch.stack(imgs)  # [T, C, H, W]

        # Slow Path 取 1/4 帧
        slow_tensor = video_tensor[::8]
        # Fast Path 全部帧
        fast_tensor = video_tensor

        # 标签
        label = self.df.loc[self.df[0] == vid, 1].values[0]
        return (slow_tensor, fast_tensor), label
