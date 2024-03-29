#! /usr/bin/env python3
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from arcgis_dataset import GeoLocalizationDataset
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
        sat_patch_height: int,
        sat_patch_width: int,
        heatmap_kernel_size: int,
        test_from_train_ratio: float,
        transform_mean: List[float],
        transform_std: List[float],
        max_rotation_angle: int,
        sat_available_years: List[str],
        uav_image_scale: float,
        val_dataloader_batch_size: int,
        val_dataloader_num_workers: int,
        train_dataloader_batch_size: int,
        train_dataloader_num_workers: int,
        misslabeled_images_path: str,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.uav_dataset_dir = uav_dataset_dir
        self.satellite_dataset_dir = satellite_dataset_dir
        self.sat_zoom_level = sat_zoom_level
        self.uav_patch_width = uav_patch_width
        self.uav_patch_height = uav_patch_height
        self.sat_patch_height = sat_patch_height
        self.sat_patch_width = sat_patch_width
        self.heatmap_kernel_size = heatmap_kernel_size
        self.test_from_train_ratio = test_from_train_ratio
        self.transform_mean = transform_mean
        self.transform_std = transform_std
        self.val_dataloader_batch_size = val_dataloader_batch_size
        self.val_dataloader_num_workers = val_dataloader_num_workers
        self.train_dataloader_batch_size = train_dataloader_batch_size
        self.train_dataloader_num_workers = train_dataloader_num_workers
        self.max_rotation_angle = max_rotation_angle
        self.sat_available_years = sat_available_years
        self.uav_image_scale = uav_image_scale
        self.misslabeled_images_path = misslabeled_images_path

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
        parser.add_argument("--sat_patch_width", type=int, default=256)
        parser.add_argument("--sat_patch_height", type=int, default=256)
        parser.add_argument("--heatmap_kernel_size", type=int, default=15)
        parser.add_argument("--test_from_train_ratio", type=float, default=0.2)
        parser.add_argument("--uav_image_scale", type=float, default=1.0)
        parser.add_argument(
            "--transform_mean", type=float, nargs="+", default=[0.485, 0.456, 0.406]
        )
        parser.add_argument(
            "--transform_std", type=float, nargs="+", default=[0.229, 0.224, 0.225]
        )
        parser.add_argument("--val_dataloader_batch_size", type=int, default=4)
        parser.add_argument("--val_dataloader_num_workers", type=int, default=16)
        parser.add_argument("--train_dataloader_batch_size", type=int, default=4)
        parser.add_argument("--train_dataloader_num_workers", type=int, default=16)
        parser.add_argument(
            "--sat_available_years", type=str, nargs="+", default=["2019"]
        )
        parser.add_argument("--max_rotation_angle", type=int, nargs="+", default=0)

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
                transform_mean=self.transform_mean,
                transform_std=self.transform_std,
                dataset="train",
                uav_image_scale=self.uav_image_scale,
                max_rotation_angle=self.max_rotation_angle,
                sat_available_years=self.sat_available_years,
                sat_patch_height=self.sat_patch_height,
                sat_patch_width=self.sat_patch_width,
                misslabeled_images_path=self.misslabeled_images_path,
            )

        self.val_dataset = GeoLocalizationDataset(
            uav_dataset_dir=self.uav_dataset_dir,
            satellite_dataset_dir=self.satellite_dataset_dir,
            sat_zoom_level=self.sat_zoom_level,
            uav_patch_width=self.uav_patch_width,
            uav_patch_height=self.uav_patch_height,
            heatmap_kernel_size=self.heatmap_kernel_size,
            test_from_train_ratio=self.test_from_train_ratio,
            uav_image_scale=self.uav_image_scale,
            transform_mean=self.transform_mean,
            transform_std=self.transform_std,
            dataset="test",
            max_rotation_angle=self.max_rotation_angle,
            sat_available_years=self.sat_available_years,
            sat_patch_height=self.sat_patch_height,
            sat_patch_width=self.sat_patch_width,
            misslabeled_images_path=self.misslabeled_images_path,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_dataloader_batch_size,
            shuffle=True,
            num_workers=self.train_dataloader_num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_dataloader_batch_size,
            shuffle=False,
            num_workers=self.val_dataloader_num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_dataloader_batch_size,
            shuffle=False,
            num_workers=self.val_dataloader_num_workers,
        )
