"""
Data loader for Param2Image v3 training (1024x1024 RGBA).
"""
from __future__ import annotations

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from preprocessing_p2i import OUTPUT_COLUMNS, PARAM_COLUMNS


class ImpactDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None, img_channels=4, scaler=None, fit_scaler=False):
        required = ["img_after"] + PARAM_COLUMNS + OUTPUT_COLUMNS
        self.df = pd.read_csv(csv_path).dropna(subset=required).reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.img_channels = img_channels
        self.scaler = scaler
        if self.scaler is not None and fit_scaler:
            self.scaler.fit(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx]

        img_path = os.path.join(self.data_dir, str(row["img_after"]).replace("\\", "/"))
        if self.img_channels == 4:
            img = Image.open(img_path).convert("RGBA")
        else:
            img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        params = row[PARAM_COLUMNS].astype(float).to_numpy()
        outputs = row[OUTPUT_COLUMNS].astype(float).to_numpy()
        if self.scaler is not None:
            params = self.scaler.transform_params_array(params)[0]
            outputs = self.scaler.transform_outputs_array(outputs)[0]

        return (
            torch.tensor(params, dtype=torch.float32),
            img,
            torch.tensor(outputs, dtype=torch.float32),
        )


def load_dataset_pytorch(
    csv_path,
    data_dir,
    batch_size,
    img_height,
    img_width,
    num_workers=0,
    img_channels=4,
    scaler=None,
    fit_scaler=False,
    return_scaler=False,
    shuffle=True,
    val_split: float = 0.0,
):
    """
    Build a DataLoader (or train/val DataLoaders if val_split > 0).
    """
    if img_channels == 4:
        mean = (0.5, 0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5, 0.5)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset = ImpactDataset(
        csv_path,
        data_dir,
        transform=transform,
        img_channels=img_channels,
        scaler=scaler,
        fit_scaler=fit_scaler,
    )

    # Keep workers alive across epochs (avoid the respawn cost every epoch)
    # and prefetch a few batches ahead so the GPU is never starved.
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    if val_split > 0:
        n_total = len(dataset)
        n_val = max(1, int(round(n_total * val_split)))
        n_train = n_total - n_val
        gen = torch.Generator().manual_seed(42)
        train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=gen)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
        result = (train_loader, val_loader)
        if return_scaler:
            return result + (scaler,)
        return result

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)
    if return_scaler:
        return loader, scaler
    return loader
