import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root, csv_file, stage="train", ratio=0.2, transform=None):
        self.root = root
        self.transforms = transform
        self.df = pd.read_csv(csv_file, header=None, skiprows=1)
        self.stage = stage
        self.ratio = ratio
        self.files = self._get_files()

    def _get_files(self):
        filenames = self.df[0].tolist()
        length = len(filenames)
        train_files = filenames[int(length * self.ratio) :]
        val_files = filenames[: int(length * self.ratio)]
        if self.stage == "train":
            return train_files
        elif self.stage == "val":
            return val_files
        elif self.stage == "test":
            return filenames
        else:
            raise ValueError("Invalid stage: choose 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        vid = self.files[index]
        video_path = os.path.join(self.root, "hw3_16fpv", f"{vid}.mp4")
        img_list = sorted(os.listdir(video_path))

        # 读取所有帧
        imgs = [
            self.transforms(
                Image.open(os.path.join(video_path, img_path)).convert("RGB")
            )
            for img_path in img_list
        ]
        video_tensor = torch.stack(imgs)  # [T, C, H, W]

        # 如果帧数不足 32，重复补足
        if video_tensor.shape[0] < 32:
            repeat_times = (32 // video_tensor.shape[0]) + 1
            video_tensor = video_tensor.repeat(repeat_times, 1, 1, 1)[:32]

        # 调整维度为 [C, T, H, W]
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        return video_tensor, self.df.loc[self.df[0] == vid, 1].values[0]
