#!/usr/bin/env python3

# Pytorch: [batch_size, channels, height, width]
# numpy: [height, width, channels]

import torch.nn as nn
import timm
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from criterion import HanningLoss, DistanceModule
import pytorch_lightning as pl
import wandb
import pandas as pd
from plot_utils import PlottingUtils
import matplotlib.pyplot as plt
import os
import json


class Xcorr(nn.Module):
    """
    Cross-correlation module.
    """

    def __init__(self) -> None:
        super(Xcorr, self).__init__()

        self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, query: torch.Tensor, search_map: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-correlation between a query and a search map.
        """

        # Pad search map to maintain spatial resolution
        search_map_padded = F.pad(
            search_map,
            (
                query.shape[3] // 2,
                query.shape[3] // 2,
                query.shape[2] // 2,
                query.shape[2] // 2,
            ),
        )

        bs, c, h, w = query.shape
        _, _, H, W = search_map_padded.shape

        search_map_padded = search_map_padded.reshape(1, bs * c, H, W)

        corr_maps = F.conv2d(search_map_padded, query, groups=bs)

        corr_maps = corr_maps.permute(1, 0, 2, 3)
        corr_maps = self.batch_norm(corr_maps)

        return corr_maps


class Fusion(nn.Module):
    def __init__(
        self,
        in_channels: tuple,
        out_channels: int,
        upsample_size: tuple,
        fusion_dropout: float,
    ) -> None:
        """
        Fusion module which is a type of convolutional neural network.

        The Fusion class is designed to merge information from two separate
        input streams. In the case of a UAV (unmanned aerial vehicle) and
        SAT (satellite), the module uses a pyramid of features from both, and
        computes correlations between them to fuse them into a single output.
        This module utilizes several 1x1 convolutions and correlation layers
        for the fusion process. The fusion is controlled by learnable weights
        for each level of the pyramid.
        """

        super(Fusion, self).__init__()

        self.fusion_dropout = fusion_dropout

        if self.fusion_dropout is None:
            self.fusion_dropout = 0

        # UAV convolutions
        self.conv1_UAV = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[0],
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=out_channels,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )
        self.conv2_UAV = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[1],
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=out_channels,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )
        self.conv3_UAV = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[2],
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=out_channels,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )

        # SAT convolutions
        self.conv1_SAT = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[0],
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=out_channels,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )
        self.conv2_SAT = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[1],
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=out_channels,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )
        self.conv3_SAT = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[2],
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                groups=out_channels,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )

        self.corrU1 = Xcorr()
        self.corrU2 = Xcorr()
        self.corrU3 = Xcorr()

        self.convU1_UAV = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                groups=64,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )
        self.convU2_UAV = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                groups=64,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )
        self.convU3_UAV = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                groups=64,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )
        self.convU3_SAT = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                groups=64,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=self.fusion_dropout),
        )

        self.upsample_size = upsample_size
        self.fusion_weights = nn.Parameter(torch.ones(3))

    def forward(
        self, sat_feature_pyramid: list, UAV_feature_pyramid: list
    ) -> torch.Tensor:
        """
        Perform the forward pass of the Fusion module.
        """
        s1_UAV_feature = UAV_feature_pyramid[0]
        s2_UAV_feature = UAV_feature_pyramid[1]
        s3_UAV_feature = UAV_feature_pyramid[2]
        s1_sat_feature = sat_feature_pyramid[0]
        s2_sat_feature = sat_feature_pyramid[1]
        s3_sat_feature = sat_feature_pyramid[2]

        # UAV feature upsampling
        U1_UAV = self.conv1_UAV(s3_UAV_feature)
        U2_UAV = F.interpolate(
            U1_UAV, size=s2_UAV_feature.shape[-2:], mode="bicubic"
        ) + self.conv2_UAV(s2_UAV_feature)
        U3_UAV = F.interpolate(
            U2_UAV, size=s1_UAV_feature.shape[-2:], mode="bicubic"
        ) + self.conv3_UAV(s1_UAV_feature)

        # SAT feature upsampling
        U1_sat = self.conv1_SAT(s3_sat_feature)
        U2_sat = F.interpolate(
            U1_sat, size=s2_sat_feature.shape[-2:], mode="bicubic"
        ) + self.conv2_SAT(s2_sat_feature)
        U3_sat = F.interpolate(
            U2_sat, size=s1_sat_feature.shape[-2:], mode="bicubic"
        ) + self.conv3_SAT(s1_sat_feature)

        U1_UAV = self.convU1_UAV(U1_UAV)
        U2_UAV = self.convU2_UAV(U2_UAV)
        U3_UAV = self.convU3_UAV(U3_UAV)
        U3_sat = self.convU3_SAT(U3_sat)

        A1 = self.corrU1(U1_UAV, U3_sat)
        A2 = self.corrU2(U2_UAV, U3_sat)
        A3 = self.corrU3(U3_UAV, U3_sat)

        fw = self.fusion_weights / torch.sum(self.fusion_weights)

        fused_map = fw[0] * A1 + fw[1] * A2 + fw[2] * A3

        fused_map = F.interpolate(fused_map, size=self.upsample_size, mode="bicubic")

        return fused_map


class SaveLayerFeatures(nn.Module):
    """
    A module that saves the output of a layer during forward pass.
    """

    def __init__(self) -> None:
        super(SaveLayerFeatures, self).__init__()
        self.outputs = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.outputs = x.clone()
        return x

    def clear(self) -> None:
        self.outputs = None


class ModifiedPCPVT(nn.Module):
    """
    A modified PVT (Pyramid Vision Transformer) model which saves features from
    its first three blocks during forward pass.
    """

    def __init__(self, original_model: nn.Module, drops: dict) -> None:
        super(ModifiedPCPVT, self).__init__()

        # Change the structure of the PCPVT model
        self.model = original_model
        self.model.blocks = original_model.blocks[:3]  # Only use the first 3 blocks
        self.model.norm = nn.Identity()  # Remove the normalization layer
        self.model.head = nn.Identity()  # Remove the head layer
        self.model.patch_embeds[
            3
        ] = nn.Identity()  # Remove the last patch embedding layer
        self.model.pos_block[
            3
        ] = nn.Identity()  # Remove the last position embedding layer

        # Add the save_features layer to the first 3 blocks
        self.save_l0 = SaveLayerFeatures()
        self.save_l1 = SaveLayerFeatures()
        self.save_l2 = SaveLayerFeatures()
        self.model.pos_block[0].proj.add_module("save_l0", self.save_l0)
        self.model.pos_block[1].proj.add_module("save_l1", self.save_l1)
        self.model.pos_block[2].proj.add_module("save_l2", self.save_l2)

        if drops is not None:
            self._set_dropout_values(self.model, drops)

    def _set_dropout_values(self, model: nn.Module, dropout_values: dict) -> None:
        """
        Regulates the dropout values of the model.
        """
        for module in model.modules():
            if hasattr(module, "attn_drop"):
                module.attn_drop.p = dropout_values.get("attn_drop", module.attn_drop.p)
            if hasattr(module, "proj_drop"):
                module.proj_drop.p = dropout_values.get("proj_drop", module.proj_drop.p)
            if hasattr(module, "head_drop"):
                module.head_drop.p = dropout_values.get("head_drop", module.head_drop.p)
            if hasattr(module, "drop1"):
                module.drop1.p = dropout_values.get("mlp_drop1", module.drop1.p)
            if hasattr(module, "drop2"):
                module.drop2.p = dropout_values.get("mlp_drop2", module.drop2.p)
            if hasattr(module, "pos_drops"):
                for drop in module.pos_drops:
                    drop.p = dropout_values.get("pos_drops", drop.p)

    def forward(self, x: torch.Tensor) -> list:
        self.save_l0.clear()
        self.save_l1.clear()
        self.save_l2.clear()

        _ = self.model(x)

        return [  # Return the feature pyramids
            self.save_l0.outputs,
            self.save_l1.outputs,
            self.save_l2.outputs,
        ]


class CrossViewLocalizationModel(pl.LightningModule):
    """
    Cross-View Localization model that uses a satellite and UAV (Unmanned Aerial Vehicle)
    view for localization.

    This model uses two modified PVT models for feature extraction from the satellite
    and UAV views, respectively. The extracted features are then passed through a Fusion
    module to produce a fused feature map.
    """

    def __init__(
        self,
        satellite_resolution: tuple,
        drops_UAV: dict,
        drops_satellite: dict,
        fusion_dropout: float,
        pretrained_twins: bool = True,
        lr_backbone: float = 1e-4,
        lr_fusion: float = 1e-4,
        milestones: list = [2, 4, 6, 8],
        gamma: float = 0.1,
        predict_checkpoint_path: str = None,
        heatmap_kernel_size: int = 33,
    ) -> None:
        super(CrossViewLocalizationModel, self).__init__()

        self.satellite_resolution = satellite_resolution
        self.fusion_dropout = fusion_dropout
        self.lr_backbone = lr_backbone
        self.lr_fusion = lr_fusion
        self.gamma = gamma
        self.criterion = HanningLoss(kernel_size=heatmap_kernel_size)
        self.distance_module = DistanceModule()
        self.milestones = milestones
        self.predict_checkpoint_path = predict_checkpoint_path

        # Statistics
        self.num_val_samples = 0
        self.num_train_samples = 0
        self.train_metre_distances_distribution = {
            "below_10": 0,
            "below_20": 0,
            "below_30": 0,
            "below_40": 0,
            "below_50": 0,
            "below_60": 0,
            "below_70": 0,
            "below_80": 0,
            "below_90": 0,
            "below_100": 0,
        }
        self.val_metre_distances_distribution = (
            self.train_metre_distances_distribution.copy()
        )
        self.train_rds = 0.0
        self.val_rds = 0.0

        if pretrained_twins is None:
            pretrained_twins = True

        # Feature extraction module
        self.backbone_UAV = timm.create_model(
            "twins_pcpvt_small", pretrained=pretrained_twins
        )
        self.feature_extractor_UAV = ModifiedPCPVT(self.backbone_UAV, drops_UAV)

        self.backbone_satellite = timm.create_model(
            "twins_pcpvt_small", pretrained=pretrained_twins
        )

        self.feature_extractor_satellite = ModifiedPCPVT(
            self.backbone_satellite, drops_satellite
        )

        self.fusion = Fusion(
            in_channels=[320, 128, 64],
            out_channels=64,
            upsample_size=self.satellite_resolution,
            fusion_dropout=self.fusion_dropout,
        )

        self.plotting_utils = PlottingUtils()

        self.save_hyperparameters()

    def forward(self, x_UAV, x_satellite):
        feature_pyramid_UAV = self.feature_extractor_UAV(x_UAV)
        feature_pyramid_satellite = self.feature_extractor_satellite(x_satellite)

        fus = self.fusion(feature_pyramid_satellite, feature_pyramid_UAV)

        fused_map = fus.squeeze(1)  # remove the unnecessary channel dimension

        return fused_map

    def configure_optimizers(self):
        params_to_update_backbone = list(
            self.feature_extractor_UAV.parameters()
        ) + list(self.feature_extractor_satellite.parameters())
        params_to_update_fusion = list(self.fusion.parameters())

        optimizer = AdamW(
            [
                {"params": params_to_update_backbone, "lr": self.lr_backbone},
                {"params": params_to_update_fusion, "lr": self.lr_fusion},
            ],
            lr=self.lr_backbone,
        )

        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def _fill_metre_distances_distribution(
        self, metre_distances, metre_distances_distribution: dict
    ):
        thresholds = list(range(10, 101, 10))  # [10, 20, ..., 100]
        thresholds = thresholds[::-1]  # [100, 90, ..., 10]

        for distance in metre_distances:
            for threshold in thresholds:
                key = f"below_{threshold}"
                if distance < threshold:
                    metre_distances_distribution[key] += 1
                else:
                    break

    def training_step(self, batch, batch_idx):
        uav_images, uav_labels, sat_images, sat_gt_hm = batch

        fused_maps = self(uav_images, sat_images)
        loss = self.criterion(fused_maps, sat_gt_hm)

        metre_distances, rds_values = self.distance_module(fused_maps, uav_labels)
        self._fill_metre_distances_distribution(
            metre_distances, self.train_metre_distances_distribution
        )
        self.train_rds += rds_values.sum().item()

        self.num_train_samples += batch[0].shape[0]
        self.log(
            "hann_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return {
            "metre_distances": metre_distances,
            "rds_values": rds_values,
            "fused_maps": fused_maps,
            "uav_labels": uav_labels,
            "sat_images": sat_images,
            "uav_images": uav_images,
            "sat_gt_hm": sat_gt_hm,
            "loss": loss,
        }

    def validation_step(self, batch, batch_idx):
        uav_images, uav_labels, sat_images, sat_gt_hm = batch

        fused_maps = self(uav_images, sat_images)
        loss = self.criterion(fused_maps, sat_gt_hm)

        metre_distances, rds_values = self.distance_module(fused_maps, uav_labels)
        self._fill_metre_distances_distribution(
            metre_distances, self.val_metre_distances_distribution
        )
        self.val_rds += rds_values.sum().item()

        self.num_val_samples += batch[0].shape[0]
        self.log(
            "hann_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return {
            "metre_distances": metre_distances,
            "rds_values": rds_values,
            "fused_maps": fused_maps,
            "uav_labels": uav_labels,
            "sat_images": sat_images,
            "uav_images": uav_images,
            "sat_gt_hm": sat_gt_hm,
            "loss": loss,
        }

    def _normalize_metre_distances_distribution(
        self, num_samples: int, metre_distances_distribution: dict
    ):
        for key in metre_distances_distribution.keys():
            metre_distances_distribution[key] /= num_samples

    def _pretty_print_metre_distances_distribution(
        self,
        num_samples: int,
        metre_distances_distribution: dict,
        mode: str = "val",
    ):
        metre_distances_distribution_df = pd.DataFrame(
            metre_distances_distribution.items(),
            columns=["distance", "percentage"],
        )
        wandb.log(
            {
                f"{mode}_metre_distances_distribution": wandb.Table(
                    dataframe=metre_distances_distribution_df
                ),
            }
        )

        if mode == "train":
            wandb.log({f"{mode}_running_rds": self.train_rds / num_samples})
        else:
            wandb.log({f"{mode}_running_rds": self.val_rds / num_samples})

    def _clear_metre_distances_distribution(self, metre_distances_distribution: dict):
        for key in metre_distances_distribution.keys():
            metre_distances_distribution[key] = 0

    def on_train_epoch_end(self):
        num_samples = self.num_train_samples
        self._normalize_metre_distances_distribution(
            num_samples, self.train_metre_distances_distribution
        )
        self._pretty_print_metre_distances_distribution(
            num_samples, self.train_metre_distances_distribution, mode="train"
        )
        self._clear_metre_distances_distribution(
            self.train_metre_distances_distribution
        )
        self.num_train_samples = 0
        self.train_rds = 0.0

    def on_validation_epoch_end(self):
        num_samples = self.num_val_samples
        self._normalize_metre_distances_distribution(
            num_samples, self.val_metre_distances_distribution
        )
        self._pretty_print_metre_distances_distribution(
            num_samples, self.val_metre_distances_distribution, mode="val"
        )
        self._clear_metre_distances_distribution(self.val_metre_distances_distribution)
        self.num_val_samples = 0
        self.val_rds = 0.0

    def _on_batch_end(self, outputs, batch, batch_idx, mode: str = "train"):
        rds_values = outputs["rds_values"]
        metre_distances = outputs["metre_distances"]
        fused_maps = outputs["fused_maps"]
        uav_labels = outputs["uav_labels"]
        sat_images = outputs["sat_images"]
        uav_images = outputs["uav_images"]
        sat_gt_hms = outputs["sat_gt_hm"]

        for i, (
            rds_value,
            metre_distance,
            fused_map,
            sat_image,
            uav_image,
            sat_gt_hm,
        ) in enumerate(
            zip(
                rds_values,
                metre_distances,
                fused_maps,
                sat_images,
                uav_images,
                sat_gt_hms,
            )
        ):
            x_sat = uav_labels["x_sat"][i].item()
            y_sat = uav_labels["y_sat"][i].item()

            fig = self.plotting_utils.plot_results(
                uav_image,
                sat_image,
                sat_gt_hm,
                fused_map,
                x_sat,
                y_sat,
            )

            image_name = f"{mode}_batch_{batch_idx}_image_{i}"

            log_data = {
                f"{image_name}/rds_value": rds_value,
                f"{image_name}/metre_distance": metre_distance,
                f"{image_name}/plot": wandb.Image(fig),
            }
            plt.close(fig)

            wandb.log(log_data)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        if batch_idx != 0:
            return
        self._on_batch_end(outputs, batch, batch_idx, mode="val")

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #    if batch_idx != 0:
    #        return
    #    self._on_batch_end(outputs, batch, batch_idx, mode="train")

    def predict_step(self, batch, batch_idx):
        uav_images, uav_labels, sat_images, sat_gt_hm = batch

        fused_maps = self(uav_images, sat_images)

        metre_distances, rds_values = self.distance_module(fused_maps, uav_labels)
        log_dir = self.predict_checkpoint_path.rsplit("/", 1)[0]
        os.makedirs(f"{log_dir}/predictions", exist_ok=True)

        for i, (
            uav_image,
            sat_image,
            sat_gt_hm,
            fused_map,
            rds_value,
            metre_distance,
        ) in enumerate(
            zip(
                uav_images,
                sat_images,
                sat_gt_hm,
                fused_maps,
                rds_values,
                metre_distances,
            )
        ):
            x_sat = uav_labels["x_sat"][i].item()
            y_sat = uav_labels["y_sat"][i].item()
            rot_angle = uav_labels["rot_angle"][i].item()
            lat_gt = uav_labels["coordinate"]["latitude"][i].item()
            lon_gt = uav_labels["coordinate"]["longitude"][i].item()
            pt_values = [uav_labels["patch_transform"][j][i].item() for j in range(6)]
            drone_im_path = uav_labels["filename"][i]

            # fig = self.plotting_utils.plot_results(
            #    uav_image,
            #    sat_image,
            #    sat_gt_hm,
            #    fused_map,
            #    x_sat,
            #    y_sat,
            # )

            image_file = f"{log_dir}/predictions/predict_batch_{batch_idx}_{i}.png"
            # metadata_file = image_file.replace(".png", ".json")

            # WARNING: This is a hack!
            # TODO: Fix this, it's a hack, now works only for one batch
            metadata_file = f"{log_dir}/predictions/metadata.json"

            metadata = {
                "rds_value": rds_value.item(),
                "metre_distance": metre_distance.item(),
                "x_sat": x_sat,
                "y_sat": y_sat,
                "rot_angle": rot_angle,
                "lat_gt": lat_gt,
                "lon_gt": lon_gt,
                "pt_values": pt_values,
                "drone_im_path": drone_im_path,
            }

            # append to metadata file
            with open(metadata_file, "a") as f:
                f.write(json.dumps(metadata))

            # plt.savefig(image_file)
            # plt.close(fig)

        return {
            "metre_distances": metre_distances,
            "rds_values": rds_values,
            "fused_maps": fused_maps,
            "uav_labels": uav_labels,
            "sat_images": sat_images,
            "uav_images": uav_images,
            "sat_gt_hm": sat_gt_hm,
        }
