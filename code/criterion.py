import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import rasterio


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


class DistanceModule(nn.Module):
    """
    Computes the Metre Distance (MD) between predicted lat-lon and ground
    truth lat-lon coordinates and Relative Distance Score (RDS).
    """

    def __init__(self, rds_k_factor: int = 10) -> None:
        super(DistanceModule, self).__init__()
        self.rds_k_factor = rds_k_factor

    def forward(
        self, heatmaps_pred: torch.Tensor, drone_infos: dict
    ) -> (torch.Tensor, torch.Tensor):
        metre_distances = torch.zeros(
            len(heatmaps_pred)
        )  # Initialize tensor to store distances

        rds_scores = torch.zeros(len(heatmaps_pred))

        for idx in range(len(heatmaps_pred)):
            heatmap_width = heatmaps_pred[idx].shape[1]
            heatmap_height = heatmaps_pred[idx].shape[0]

            coords = torch.where(heatmaps_pred[idx] == heatmaps_pred[idx].max())
            y_pred, x_pred = coords[0][0].item(), coords[1][0].item()

            patch_transform = drone_infos["patch_transform"][idx]

            lat_pred, lon_pred = self.pixel_to_geo_coordinates(
                x_pred, y_pred, patch_transform
            )

            lat_gt, lon_gt = (
                drone_infos["coordinate"]["latitude"][idx].item(),
                drone_infos["coordinate"]["longitude"][idx].item(),
            )

            distance = self.haversine(lon_pred, lat_pred, lon_gt, lat_gt)
            metre_distances[idx] = distance
            dx = abs(lon_pred - lon_gt)
            dy = abs(lat_pred - lat_gt)
            rds_scores[idx] = torch.exp(
                -self.rds_k_factor
                * (
                    torch.sqrt(
                        ((dx / heatmap_width) ** 2) + ((dy / heatmap_height) ** 2)
                    )
                    / 2
                )
            )

        return metre_distances, rds_scores

    def haversine(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """
        Calculate the great circle distance in meters between two points
        on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = 6371 * c * 1000  # Multiply by 1000 to get meters

        return distance

    def pixel_to_geo_coordinates(
        self, x: int, y: int, transform: rasterio.transform.Affine
    ) -> (float, float):
        """
        Converts a pair of pixel coordinates to (lat, lon) coordinates.
        """
        lon, lat = transform * (x, y)
        return lat, lon
