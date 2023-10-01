#! /usr/bin/env python3

from torchvision import transforms
import numpy as np
import torch
import rasterio
from matplotlib import pyplot as plt
import matplotlib.patches as patches


class PlottingUtils:
    def __init__(self):
        self.inverse_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[
                        -m / s
                        for m, s in zip(
                            [0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225],
                        )
                    ],
                    std=[1 / s for s in [0.229, 0.224, 0.225]],
                ),
                transforms.ToPILImage(),
            ]
        )

    def plot_results(
        self,
        drone_image: torch.Tensor,
        sat_image: torch.Tensor,
        heatmap_gt: torch.Tensor,
        heatmap_pred: torch.Tensor,
        x_gt: int,
        y_gt: int,
    ):
        heatmap_pred_np = heatmap_pred.cpu().numpy()
        y_pred, x_pred = np.unravel_index(
            np.argmax(heatmap_pred_np), heatmap_pred_np.shape
        )

        fig, axs = plt.subplots(1, 3, figsize=(30, 10))

        # Subplot 1: Drone Image
        axs[0].imshow(self.inverse_transform(drone_image))
        axs[0].axis("off")

        # Subplot 2: Satellite Image
        axs[1].imshow(self.inverse_transform(sat_image))
        axs[1].axis("off")

        # Subplot 3: Satellite Image with Predicted Heatmap and circles
        axs[2].imshow(self.inverse_transform(sat_image))
        axs[2].imshow(heatmap_pred.squeeze(0).cpu().numpy(), cmap="jet", alpha=0.55)
        pred_circle = patches.Circle(
            (x_pred, y_pred), radius=10, edgecolor="blue", facecolor="none", linewidth=4
        )
        gt_circle = patches.Circle(
            (x_gt, y_gt),
            radius=10,
            edgecolor="red",
            facecolor="none",
            linewidth=4,
        )
        axs[2].add_patch(pred_circle)
        axs[2].add_patch(gt_circle)
        axs[2].legend(
            [pred_circle, gt_circle], ["Prediction", "Ground Truth"], loc="upper right"
        )
        axs[2].axis("off")

        # PLot it
        return fig
