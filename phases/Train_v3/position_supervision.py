"""
Differentiable supervision and metrics for impact location in generated images.

The rendered plate is mostly blue away from damage, while the impact response
contains warmer/brighter pixels. A soft centroid of that response gives the
generator a direct location signal without changing its inputs or output shape.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


DEFAULT_DOWNSAMPLE_SIZE = 128
DEFAULT_TEMPERATURE = 12.0
DEFAULT_BORDER_FRACTION = 0.04


def _response_logits(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or images.size(1) < 3:
        raise ValueError("images must have shape (batch, channels>=3, height, width)")
    rgb = (images[:, :3].float().clamp(-1.0, 1.0) + 1.0) * 0.5
    # The rendered impact rings are yellow/red while intact plate is blue.
    return rgb[:, 0] + rgb[:, 1] - 0.5 * rgb[:, 2]


def soft_impact_centroid(
    images: torch.Tensor,
    *,
    downsample_size: int = DEFAULT_DOWNSAMPLE_SIZE,
    temperature: float = DEFAULT_TEMPERATURE,
    border_fraction: float = DEFAULT_BORDER_FRACTION,
    normalized: bool = False,
) -> torch.Tensor:
    """Return soft impact centroids as ``(x, y)`` coordinates per image."""
    if downsample_size < 2:
        raise ValueError("downsample_size must be at least 2")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= border_fraction < 0.5:
        raise ValueError("border_fraction must be in [0, 0.5)")

    _, _, height, width = images.shape
    margin_y = min(int(round(height * border_fraction)), max(0, (height - 2) // 2))
    margin_x = min(int(round(width * border_fraction)), max(0, (width - 2) // 2))
    top, bottom = margin_y, height - margin_y
    left, right = margin_x, width - margin_x

    cropped = images[:, :, top:bottom, left:right]
    target_height = min(downsample_size, cropped.size(2))
    target_width = min(downsample_size, cropped.size(3))
    reduced = F.interpolate(
        cropped,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    logits = _response_logits(reduced).flatten(1)
    weights = F.softmax(logits * temperature, dim=1)

    xs = torch.linspace(float(left), float(right - 1), target_width, device=images.device, dtype=weights.dtype)
    ys = torch.linspace(float(top), float(bottom - 1), target_height, device=images.device, dtype=weights.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    centroids = torch.stack(
        [
            (weights * grid_x.flatten()).sum(dim=1),
            (weights * grid_y.flatten()).sum(dim=1),
        ],
        dim=1,
    )
    if normalized:
        scale = images.new_tensor([max(width - 1, 1), max(height - 1, 1)], dtype=centroids.dtype)
        centroids = centroids / scale * 2.0 - 1.0
    return centroids


def impact_location_loss(
    generated: torch.Tensor,
    target: torch.Tensor,
    *,
    downsample_size: int = DEFAULT_DOWNSAMPLE_SIZE,
    temperature: float = DEFAULT_TEMPERATURE,
    border_fraction: float = DEFAULT_BORDER_FRACTION,
) -> torch.Tensor:
    """Smooth-L1 loss between generated and target soft impact locations."""
    generated_centroid = soft_impact_centroid(
        generated,
        downsample_size=downsample_size,
        temperature=temperature,
        border_fraction=border_fraction,
        normalized=True,
    )
    target_centroid = soft_impact_centroid(
        target,
        downsample_size=downsample_size,
        temperature=temperature,
        border_fraction=border_fraction,
        normalized=True,
    ).detach()
    return F.smooth_l1_loss(generated_centroid, target_centroid)


@torch.no_grad()
def impact_position_error_pixels(
    generated: torch.Tensor,
    target: torch.Tensor,
    *,
    downsample_size: int = DEFAULT_DOWNSAMPLE_SIZE,
    temperature: float = DEFAULT_TEMPERATURE,
    border_fraction: float = DEFAULT_BORDER_FRACTION,
) -> torch.Tensor:
    """Return mean Euclidean impact-location error in image pixels."""
    generated_centroid = soft_impact_centroid(
        generated,
        downsample_size=downsample_size,
        temperature=temperature,
        border_fraction=border_fraction,
    )
    target_centroid = soft_impact_centroid(
        target,
        downsample_size=downsample_size,
        temperature=temperature,
        border_fraction=border_fraction,
    )
    return torch.linalg.vector_norm(generated_centroid - target_centroid, dim=1).mean()


def pixels_to_plate_mm(
    error_pixels: float,
    image_width: int,
    *,
    plate_width_mm: float = 2500.0,
    border_fraction: float = DEFAULT_BORDER_FRACTION,
) -> float:
    """Approximate rendered position error in millimetres across the plate."""
    usable_width = max(float(image_width) * (1.0 - 2.0 * border_fraction), 1.0)
    return float(error_pixels) * float(plate_width_mm) / usable_width
