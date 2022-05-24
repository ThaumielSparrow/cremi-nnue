import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class LoadData(Dataset):
    def __init__(self, em_dir = 'EM', seg_dir = 'SEG', transform=None):
        self.em_dir = em_dir
        self.seg_dir = seg_dir
        self.transform = transform
        self.em = os.listdir(em_dir)
        self.seg = os.listdir(seg_dir)

    def __len__(self):
        return len(self.em)

    def __getitem__(self, idx):
        em_path = os.path.join(self.em_dir, self.em[idx])
        seg_path = os.path.join(self.seg_dir, self.seg[idx])
        EM = np.array(Image.open(em_path).convert('RGB'))
        SEG = np.array(Image.open(seg_path).convert('L'), dtype=np.float32)
        SEG[SEG == 255.0] = 1.0

        # Albumentations augmentations
        if self.transform is not None:
            augmentations = self.transform(image=EM, mask=SEG)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask