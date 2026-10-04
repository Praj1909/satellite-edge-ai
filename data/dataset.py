"""EuroSAT dataset: 27,000 Sentinel-2 images, 10 land-use classes, 64x64."""
import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "eurosat_data")
BATCH_SIZE  = 32
NUM_CLASSES = 10
IMG_SIZE    = 64
MEAN = [0.3444, 0.3803, 0.4078]
STD  = [0.2025, 0.1364, 0.1153]

CLASS_NAMES = [
    "AnnualCrop","Forest","HerbaceousVeg","Highway",
    "Industrial","Pasture","PermanentCrop","Residential",
    "River","SeaLake"
]

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf

def load_eurosat(data_dir=DATA_DIR, val_split=0.2, batch_size=BATCH_SIZE):
    os.makedirs(data_dir, exist_ok=True)
    train_tf, val_tf = get_transforms()
    full = datasets.EuroSAT(root=data_dir, download=True, transform=train_tf)
    n_val   = int(len(full) * val_split)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    val_ds.dataset = datasets.EuroSAT(root=data_dir, download=False, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, CLASS_NAMES

INFER_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
