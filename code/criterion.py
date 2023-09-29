import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


# class RDS(nn.Module):
#    """
#    Computes the Relative Distance Score (RDS) between predicted heatmaps and
#    ground truth coordinates.
#    """
#
#    def __init__(self, k=10):
#        super(RDS, self).__init__()
#        self.k = k
#
#    def forward(self, heatmaps_pred, xs_gt, ys_gt, hm_w, hm_h):
#        running_rds = 0.0
#        for heatmap_pred, x_gt, y_gt in zip(heatmaps_pred, xs_gt, ys_gt):
#            coords = torch.where(heatmap_pred == heatmap_pred.max())
#            y_pred, x_pred = coords[0][0], coords[1][0]
#            dx = torch.abs(x_pred - x_gt)
#            dy = torch.abs(y_pred - y_gt)
#            running_rds += torch.exp(
#                -self.k * (torch.sqrt(((dx / hm_w) ** 2 + (dy / hm_h) ** 2)) / 2)
#            )
#
#        return running_rds / len(heatmaps_pred)
#


# class MeterDistance(nn.Module):
#    """
#    Computes the Metre Distance (MD) between predicted lat-lon and ground
#    truth lat-lon coordinates.
#    """
#
#    def forward(self, heatmaps_pred, drone_infos):
#        """
#        Args:
#            heatmaps_pred: The predicted x, y offsets.
#            drone_infos: Information about drones which includes ground truth latitudes and longitudes, as well as satellite image details.
#
#        Returns:
#            Tensor of Meter Distances.
#        """
#
#        distances = torch.zeros(
#            len(heatmaps_pred)
#        )  # Initialize tensor to store distances
#
#        for idx in range(len(heatmaps_pred)):
#
#            coords = torch.where(heatmaps_pred[idx] == heatmaps_pred[idx].max())
#            y_pred, x_pred = coords[0][0].item(), coords[1][0].item()
#
#            sat_image_path = drone_infos["filename"][idx]
#            zoom_level = drone_infos["zoom_level"][idx]
#            x_offset = drone_infos["x_offset"][idx]
#            y_offset = drone_infos["y_offset"][idx]
#
#            with rasterio.open(f"{sat_image_path}_sat_{zoom_level}.tiff") as s_image:
#                sat_transform = s_image.transform
#                lon_pred, lat_pred = rasterio.transform.xy(
#                    sat_transform, y_pred + y_offset, x_pred + x_offset
#                )
#
#            lat_gt, lon_gt = (
#                drone_infos["coordinate"]["latitude"][idx].item(),
#                drone_infos["coordinate"]["longitude"][idx].item(),
#            )
#            distance = self.haversine(lon_pred[0], lat_pred[0], lon_gt, lat_gt)
#            distances[idx] = distance
#
#        return distances
#
#    def haversine(self, lon1, lat1, lon2, lat2):
#        """
#        Calculate the great circle distance in meters between two points
#        on the earth (specified in decimal degrees)
#        """
#        # Convert decimal degrees to radians
#        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
#
#        # Haversine formula
#        dlon = lon2 - lon1
#        dlat = lat2 - lat1
#        a = (
#            math.sin(dlat / 2) ** 2
#            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
#        )
#        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#        distance = 6371 * c * 1000  # Multiply by 1000 to get meters
#
#        return distance
