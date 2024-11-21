import os
import random
import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class MyDataset(Dataset):
    def __init__(
        self,
        root="./data/hw3_16fpv",
        csv_file=None,
        stage="train",
        ratio=0.2,
        transform=None,
    ):
        self.root = root
        self.transforms = transform
        self.df = pd.read_csv(csv_file, header=None, skiprows=1)
        self.stage = stage
        self.ratio = ratio
        self.files = self._get_files()

    def _get_files(self):
        filenames = self.df[0].tolist()
        length = len(filenames)
        print(f"All length is {length}")
        train_files = filenames[int(length * self.ratio) :]
        val_files = filenames[: int(length * self.ratio)]
        print(f"Len train is {len(train_files)}, len val is {len(val_files)}")
        if self.stage == "train":
            return train_files
        elif self.stage == "val":
            return val_files
        elif self.stage == "test":
            return filenames
        else:
            raise ValueError("Stage should be either 'train', 'val' or 'test'")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        vid = self.files[index]
        vid = vid.split(".mp4")[0]
        # Corrected path to match nested structure
        label = self.df.loc[self.df[0] == vid, 1].values[0]
        img_list = os.listdir(os.path.join(self.root, "hw3_16fpv", f"{vid}.mp4"))
        img_16fpv = [
            self.transforms(
                Image.open(
                    os.path.join(self.root, "hw3_16fpv", f"{vid}.mp4", img_path)
                ).convert("RGB")
            )
            for img_path in img_list
        ]
        img_16fpv_tensor = torch.stack(img_16fpv).permute(1, 0, 2, 3)
        return img_16fpv_tensor, label
