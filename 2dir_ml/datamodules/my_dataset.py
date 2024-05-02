import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class MyDataSet(Dataset):
    def __init__(self, npz_file, csv_file, transform=None):
        self.npz_file = npz_file
        self.protein_ss = pd.read_csv(csv_file, header=None)
        self.transform = transform
        self.img_names = [name.split('_')[1] for name in list(np.load(npz_file).keys())]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img = np.load(self.npz_file)[f"arr_{img_name}"]
        label_values = self.protein_ss.iloc[int(img_name), 1:4]
        label = torch.tensor(label_values.values.astype(np.float32), dtype=torch.float32)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_name
