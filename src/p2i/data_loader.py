"""
Data loader for the impact simulation dataset (PyTorch).

Loads ``results_master.csv`` and associated post-impact images into a
DataLoader suitable for multi-modal cGAN training.
"""
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from p2i.preprocessing import OUTPUT_COLUMNS, PARAM_COLUMNS


class ImpactDataset(Dataset):
    """
    PyTorch Dataset for impact simulations (RGBA images + tabular fields).
    """

    def __init__(
        self,
        csv_path,
        data_dir,
        transform=None,
        img_channels=4,
        scaler=None,
        fit_scaler=False,
    ):
        required_cols = [
            "img_after",
            "velocity_ms",
            "proj_radius_mm",
            "impact_x_mm",
            "impact_y_mm",
            "max_mises_mpa",
            "max_peeq",
            "max_deflection_mm",
        ]
        self.df = pd.read_csv(csv_path).dropna(subset=required_cols)
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

        after_img_path = os.path.join(self.data_dir, row["img_after"].replace("\\", "/"))
        if self.img_channels == 4:
            after_image = Image.open(after_img_path).convert("RGBA")
        else:
            after_image = Image.open(after_img_path).convert("RGB")

        if self.transform:
            after_image = self.transform(after_image)

        param_values = row[PARAM_COLUMNS].astype(float).to_numpy()
        output_values = row[OUTPUT_COLUMNS].astype(float).to_numpy()

        if self.scaler is not None:
            param_values = self.scaler.transform_params_array(param_values)[0]
            output_values = self.scaler.transform_outputs_array(output_values)[0]

        input_params = torch.tensor(param_values, dtype=torch.float32)
        numerical_outputs = torch.tensor(output_values, dtype=torch.float32)

        return input_params, after_image, numerical_outputs


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
):
    """
    Build a shuffled DataLoader for Param2Image training.
    """
    print(
        f"--- Loading PyTorch dataset for Multi-modal cGAN "
        f"({img_channels}-channel, {img_height}x{img_width}) ---"
    )

    if img_channels == 4:
        norm_mean = (0.5, 0.5, 0.5, 0.5)
        norm_std = (0.5, 0.5, 0.5, 0.5)
    else:
        norm_mean = (0.5, 0.5, 0.5)
        norm_std = (0.5, 0.5, 0.5)

    transform = transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )

    dataset = ImpactDataset(
        csv_path,
        data_dir,
        transform=transform,
        img_channels=img_channels,
        scaler=scaler,
        fit_scaler=fit_scaler,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Found {len(dataset)} complete simulation records.")
    print("--- PyTorch Dataset created successfully ---")
    if scaler is not None:
        print("--- Tabular scaling enabled for impact parameters and numerical outputs ---")
    if return_scaler:
        return dataloader, scaler
    return dataloader


def main():
    from p2i.paths import default_data_dir

    data_dir = default_data_dir(Path(__file__))
    csv_path = data_dir / "results_master.csv"
    dataloader = load_dataset_pytorch(str(csv_path), str(data_dir), 4, 256, 256)

    for params, after_images, numerical_outputs in dataloader:
        print("\n--- Verifying one batch ---")
        print("Parameters batch shape:", params.shape)
        print("After images batch shape:", after_images.shape)
        print("Numerical outputs batch shape:", numerical_outputs.shape)
        print("---------------------------\n")
        break


if __name__ == "__main__":
    main()
