#! /usr/bin/env python3
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import GeoLocalizationDataset
from argparse import ArgumentParser
from typing import List


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        uav_dataset_dir: str,
        satellite_dataset_dir: str,
        sat_zoom_level: int,
        uav_patch_width: int,
        uav_patch_height: int,
        heatmap_kernel_size: int,
        test_from_train_ratio: float,
        uav_scales: List[float],
        transform_mean: List[float],
        transform_std: List[float],
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.uav_dataset_dir = uav_dataset_dir
        self.satellite_dataset_dir = satellite_dataset_dir
        self.sat_zoom_level = sat_zoom_level
        self.uav_patch_width = uav_patch_width
        self.uav_patch_height = uav_patch_height
        self.heatmap_kernel_size = heatmap_kernel_size
        self.test_from_train_ratio = test_from_train_ratio
        self.uav_scales = uav_scales
        self.transform_mean = transform_mean
        self.transform_std = transform_std

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--uav_dataset_dir", type=str, default="path_to_uav_dataset"
        )
        parser.add_argument(
            "--satellite_dataset_dir", type=str, default="path_to_satellite_dataset"
        )
        parser.add_argument("--sat_zoom_level", type=int, default=19)
        parser.add_argument("--uav_patch_width", type=int, default=256)
        parser.add_argument("--uav_patch_height", type=int, default=256)
        parser.add_argument("--heatmap_kernel_size", type=int, default=15)
        parser.add_argument("--test_from_train_ratio", type=float, default=0.2)
        parser.add_argument("--uav_scales", type=float, nargs="+", default=[1.0])
        parser.add_argument(
            "--transform_mean", type=float, nargs="+", default=[0.485, 0.456, 0.406]
        )
        parser.add_argument(
            "--transform_std", type=float, nargs="+", default=[0.229, 0.224, 0.225]
        )
        return parser

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = GeoLocalizationDataset(
                uav_dataset_dir=self.uav_dataset_dir,
                satellite_dataset_dir=self.satellite_dataset_dir,
                sat_zoom_level=self.sat_zoom_level,
                uav_patch_width=self.uav_patch_width,
                uav_patch_height=self.uav_patch_height,
                heatmap_kernel_size=self.heatmap_kernel_size,
                test_from_train_ratio=self.test_from_train_ratio,
                uav_scales=self.uav_scales,
                transform_mean=self.transform_mean,
                transform_std=self.transform_std,
                dataset="train",
            )

        self.val_dataset = GeoLocalizationDataset(
            uav_dataset_dir=self.uav_dataset_dir,
            satellite_dataset_dir=self.satellite_dataset_dir,
            sat_zoom_level=self.sat_zoom_level,
            uav_patch_width=self.uav_patch_width,
            uav_patch_height=self.uav_patch_height,
            heatmap_kernel_size=self.heatmap_kernel_size,
            test_from_train_ratio=self.test_from_train_ratio,
            uav_scales=self.uav_scales,
            transform_mean=self.transform_mean,
            transform_std=self.transform_std,
            dataset="test",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=8, shuffle=True, num_workers=16
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=8, shuffle=True, num_workers=16)
