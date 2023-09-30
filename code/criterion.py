import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import rasterio
import mercantile
from PIL import Image
import os
from rasterio.io import MemoryFile
import gc
from rasterio.transform import from_bounds
from rasterio.merge import merge
import numpy as np


class HanningLoss(nn.Module):
    """
    Computes a weighted Binary Cross-Entropy loss, emphasizing the central part
    of positive regions in the target tensor.
    """

    def __init__(
        self, kernel_size: int = 33, negative_weight: int = 1, device: str = "cuda:0"
    ) -> None:
        super(HanningLoss, self).__init__()
        self.kernel_size = kernel_size
        self.device = device
        self.negative_weight = negative_weight
        self._prepare_hann_kernel()

    def _prepare_hann_kernel(self):
        hann_kernel = torch.hann_window(
            self.kernel_size, periodic=False, dtype=torch.float, device=self.device
        )
        hann_kernel = hann_kernel.view(1, 1, -1, 1) * hann_kernel.view(1, 1, 1, -1)
        self.hann_kernel = hann_kernel

    def _get_bounds(self, mask: torch.Tensor):
        indices = torch.nonzero(mask)
        ymin, xmin = indices.min(dim=0)[0]
        ymax, xmax = indices.max(dim=0)[0]
        return xmin.item(), ymin.item(), (xmax + 1).item(), (ymax + 1).item()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        batch_size = target.shape[0]
        batch_loss = 0.0

        for i in range(batch_size):
            weights = torch.zeros_like(target[i])
            xmin, ymin, xmax, ymax = self._get_bounds(target[i] == 1)
            weights[ymin:ymax, xmin:xmax] = self.hann_kernel

            # Normalize positive weights
            weights /= weights.sum()

            # Compute negative weights
            num_negative = (weights == 0).sum()

            negative_weight = self.negative_weight / num_negative

            # Assign weights
            weights = torch.where(weights == 0, negative_weight, weights)

            # Normalize weights again
            weights /= weights.sum()

            bce_l = F.binary_cross_entropy_with_logits(
                pred[i].view(1, 1, *pred[i].shape),
                target[i].view(1, 1, *target[i].shape),
                weight=weights,
                reduction="sum",
            )
            batch_loss += bce_l

        return batch_loss / batch_size


def get_tiff_map(self, tile: mercantile.Tile) -> (np.ndarray, dict):
    """
    Returns a TIFF map of the given tile.
    """
    tile_data = []
    neighbors = mercantile.neighbors(tile)
    neighbors.append(tile)

    for neighbor in neighbors:
        west, south, east, north = mercantile.bounds(neighbor)
        tile_path = (
            f"{self.satellite_dataset_dir}/{neighbor.z}_{neighbor.x}_{neighbor.y}.jpg"
        )
        if not os.path.exists(tile_path):
            self.download_missing_tile(neighbor)

        with Image.open(tile_path) as img:
            width, height = img.size
        memfile = MemoryFile()
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype="uint8",
            crs="EPSG:3857",
            transform=from_bounds(west, south, east, north, width, height),
        ) as dataset:
            data = rasterio.open(tile_path).read()
            dataset.write(data)
        tile_data.append(memfile.open())
        memfile.close()

    mosaic, out_trans = merge(tile_data)

    out_meta = tile_data[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": "EPSG:3857",
        }
    )

    # Clean up MemoryFile instances to free up memory
    for td in tile_data:
        td.close()

    del neighbors
    del tile_data
    gc.collect()

    return mosaic, out_meta
