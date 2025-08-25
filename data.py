import os, glob, random
import cv2
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms


def split_dataset(root_dir: str, train_ratio: float = 0.8, seed: int = 42):
    """Split dataset into train/val/test patient IDs."""
    patients = next(os.walk(root_dir))[1]
    random.seed(seed)
    train = random.sample(patients, k=int(train_ratio * len(patients)))
    remaining = list(set(patients) - set(train))
    val = random.sample(remaining, k=len(remaining) // 2)
    test = list(set(remaining) - set(val))
    return train, val, test


class BrainSegmentationDataset(data.Dataset):
    """LGG MRI segmentation dataset with optional augmentations."""

    def __init__(self, root_dir, patients, augment=False, image_size=256, seed=42):
        self.mask_files = []
        self.augment = augment
        self.image_size = image_size
        self.rng = np.random.default_rng(seed)

        for p in patients:
            self.mask_files += glob.glob(os.path.join(root_dir, p, "*mask.tif"))

        self.img_files = [f.replace("_mask", "") for f in self.mask_files]

        self.base_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        
        self.normalize = transforms.Normalize(mean=0.5, std=0.5)

        self.aug_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-15, 15)),
            transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=15)], p=0.5)
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.img_files[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_files[idx], 0)

        img = self.normalize(self.base_tf(img))
        mask = self.base_tf(mask)

        if self.augment:
            seed = int(self.rng.integers(1e9))
            torch.manual_seed(seed)
            img = self.aug_tf(img)
            torch.manual_seed(seed)
            mask = self.aug_tf(mask)

        return img, mask


def make_dataloaders(trainset, valset, testset, batch_size=16, num_workers=2):
    return (
        data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    )
