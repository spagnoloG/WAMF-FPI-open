#!/usr/bin/env python3

import json
import torch
import os
from typing import Literal, List
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
from arcgis_sat_utils_v2 import SatUtils


class GeoLocalizationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        uav_dataset_dir: str,
        satellite_dataset_dir: str,
        dataset: Literal["train", "test"],
        sat_zoom_level: int = 16,
        uav_patch_width: int = 128,
        uav_patch_height: int = 128,
        sat_patch_width: int = 400,
        sat_patch_height: int = 400,
        heatmap_kernel_size: int = 33,
        test_from_train_ratio: float = 0.0,
        transform_mean: List[float] = [0.485, 0.456, 0.406],
        transform_std: List[float] = [0.229, 0.224, 0.225],
        sat_available_years: List[str] = ["2023", "2021", "2019", "2016"],
        rotation_angles: List[int] = [0, 45, 90, 135, 180, 225, 270, 315],
        uav_image_scale: float = 1,
    ):
        self.uav_dataset_dir = uav_dataset_dir
        self.satellite_dataset_dir = satellite_dataset_dir
        self.dataset = dataset
        self.sat_zoom_level = sat_zoom_level
        self.uav_patch_width = uav_patch_width
        self.uav_patch_height = uav_patch_height
        self.heatmap_kernel_size = heatmap_kernel_size
        self.test_from_train_ratio = test_from_train_ratio
        self.transform_mean = transform_mean
        self.transform_std = transform_std
        self.metadata_dict = {}
        self.total_uav_samples = self.count_total_uav_samples()
        self.entry_paths = self.get_entry_paths(self.uav_dataset_dir)
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.transform_mean, self.transform_std),
            ]
        )
        self.sat_available_years = sat_available_years
        self.rotation_angles = rotation_angles
        self.uav_image_scale = uav_image_scale
        self.sat_patch_width = sat_patch_width
        self.sat_patch_height = sat_patch_height

        self.inverse_transforms = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[
                        -m / s for m, s in zip(self.transform_mean, self.transform_std)
                    ],
                    std=[1 / s for s in self.transform_std],
                ),
                transforms.ToPILImage(),
            ]
        )

        self.sat_utils = SatUtils(
            self.satellite_dataset_dir, self.sat_zoom_level, self.heatmap_kernel_size
        )

    def __len__(self) -> int:
        return (
            len(self.entry_paths)
            * len(self.rotation_angles)
            * len(self.sat_available_years)
        )

    def __getitem__(self, idx) -> (torch.Tensor, dict, torch.Tensor, torch.Tensor):
        """
        Retrieves a sample given its index, returning the preprocessed UAV
        and satellite images, along with their associated heatmap and metadata.
        """

        image_path_index = idx // (
            len(self.rotation_angles) * len(self.sat_available_years)
        )
        sat_year = self.sat_available_years[idx % len(self.sat_available_years)]

        image_path = self.entry_paths[image_path_index]
        uav_image = Image.open(image_path).convert("RGB")  # Ensure 3-channel image

        lookup_str, file_number = self.extract_info_from_filename(image_path)
        img_info = self.metadata_dict[lookup_str][file_number]
        img_info["filename"] = image_path

        lat, lon = (
            img_info["coordinate"]["latitude"],
            img_info["coordinate"]["longitude"],
        )

        (
            satellite_patch,
            x_sat,
            y_sat,
            x_offset,
            y_offset,
            patch_transform,
        ) = self.sat_utils.get_random_tiff_patch(
            lat, lon, self.sat_patch_width, self.sat_patch_height, sat_year, 0
        )

        # Rotate crop center and transform image
        h = np.ceil(uav_image.height // self.uav_image_scale).astype(int)
        w = np.ceil(uav_image.width // self.uav_image_scale).astype(int)

        uav_image = F.resize(uav_image, [h, w])
        uav_image = F.center_crop(
            uav_image, (self.uav_patch_height, self.uav_patch_width)
        )
        uav_image = self.transforms(uav_image)

        satellite_patch = satellite_patch.transpose(1, 2, 0)
        satellite_patch = self.transforms(satellite_patch)

        heatmap = self.get_heatmap_gt(
            x_sat,
            y_sat,
            satellite_patch.shape[1],
            satellite_patch.shape[2],
            self.heatmap_kernel_size,
        )

        img_info["x_sat"] = x_sat
        img_info["y_sat"] = y_sat
        img_info["x_offset"] = x_offset
        img_info["y_offset"] = y_offset
        img_info["patch_transform"] = patch_transform
        img_info["uav_image_scale"] = self. uav_image_scale

        return uav_image, img_info, satellite_patch, heatmap

    def count_total_uav_samples(self) -> int:
        """
        Count the total number of uav image samples in the dataset
        (train + test)
        """
        total_samples = 0

        for dirpath, dirnames, filenames in os.walk(self.uav_dataset_dir):
            # Skip the test folder
            for filename in filenames:
                if filename.endswith(".jpeg"):
                    total_samples += 1
        return total_samples

    def get_number_of_city_samples(self) -> int:
        """
        TODO: Count the total number of city samples in the dataset
        """
        return 11

    def get_entry_paths(self, directory: str) -> List[str]:
        """
        Recursively retrieves paths to image and metadata files in the given directory.
        """
        entry_paths = []
        entries = os.listdir(directory)

        images_to_take_per_folder = int(
            self.total_uav_samples
            * self.test_from_train_ratio
            / self.get_number_of_city_samples()
        )

        for entry in entries:
            entry_path = os.path.join(directory, entry)

            # If it's a directory, recurse into it
            if os.path.isdir(entry_path):
                entry_paths += self.get_entry_paths(entry_path)

            # Handle train dataset
            elif (self.dataset == "train" and "Train" in entry_path) or (
                self.dataset == "train"
                and self.test_from_train_ratio > 0
                and "Test" in entry_path
            ):
                if entry_path.endswith(".jpeg"):
                    _, number = self.extract_info_from_filename(entry_path)
                else:
                    number = None
                if entry_path.endswith(".json"):
                    self.get_metadata(entry_path)
                if number is None:
                    continue
                if (
                    number >= images_to_take_per_folder
                ):  # Only include images beyond the ones taken for test
                    if entry_path.endswith(".jpeg"):
                        entry_paths.append(entry_path)

            # Handle test dataset
            elif self.dataset == "test":
                if entry_path.endswith(".jpeg"):
                    _, number = self.extract_info_from_filename(entry_path)
                else:
                    number = None
                if entry_path.endswith(".json"):
                    self.get_metadata(entry_path)

                if number is None:
                    continue
                if (
                    ("Test" in entry_path and number < images_to_take_per_folder)
                    or (number < images_to_take_per_folder and "Train" in entry_path)
                    or (self.test_from_train_ratio == 0.0 and "Test" in entry_path)
                ):
                    if entry_path.endswith(".jpeg"):
                        entry_paths.append(entry_path)

        return sorted(entry_paths, key=self.extract_info_from_filename)

    def get_metadata(self, path: str) -> None:
        """
        Extracts metadata from a JSON file and stores it in the metadata dictionary.
        """
        with open(path, newline="") as jsonfile:
            json_dict = json.load(jsonfile)
            path = path.split("/")[-1]
            path = path.replace(".json", "")
            self.metadata_dict[path] = json_dict["cameraFrames"]

    def extract_info_from_filename(self, filename: str) -> (str, int):
        """
        Extracts information from the filename.
        """
        filename_without_ext = filename.replace(".jpeg", "")
        segments = filename_without_ext.split("/")
        info = segments[-1]
        try:
            number = int(info.split("_")[-1])
        except ValueError:
            print("Could not extract number from filename: ", filename)
            return None, None

        info = "_".join(info.split("_")[:-1])

        return info, number

    def get_heatmap_gt(
        self, x: int, y: int, height: int, width: int, square_size: int = 33
    ) -> torch.Tensor:
        """
        Returns 2D heatmap ground truth for the given x and y coordinates,
        with the given square size.
        """
        x_map, y_map = x, y

        heatmap = torch.zeros((height, width))

        half_size = square_size // 2

        # Calculate the valid range for the square
        start_x = max(0, x_map - half_size)
        end_x = min(
            width, x_map + half_size + 1
        )  # +1 to include the end_x in the square
        start_y = max(0, y_map - half_size)
        end_y = min(
            height, y_map + half_size + 1
        )  # +1 to include the end_y in the square

        heatmap[start_y:end_y, start_x:end_x] = 1

        return heatmap


def test():
    import pytest

    dataset = GeoLocalizationDataset(
        uav_dataset_dir="/home/spagnologasper/Documents/projects/uav-localization-experiments/drone_dataset",
        satellite_dataset_dir="/home/spagnologasper/Documents/projects/historical_satellite_tiles_downloader/tiles",
        sat_available_years=["2023", "2021", "2019", "2016"],
        rotation_angles=[0],
        dataset="train",
        sat_zoom_level=17,
        uav_patch_width=128,
        uav_patch_height=128,
        sat_patch_width=500,
        sat_patch_height=500,
        heatmap_kernel_size=33,
        test_from_train_ratio=0.1,
        transform_mean=[0.485, 0.456, 0.406],
        transform_std=[0.229, 0.224, 0.225],
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4
    )

    print("Testing patch retrieval & transformation...")
    query_lat = 46.048992
    query_lon = 14.509215

    (
        sat_patch,
        x_sat,
        y_sat,
        x_offset,
        y_offset,
        patch_transform,
    ) = dataset.sat_utils.get_random_tiff_patch(
        query_lat,  # latitude
        query_lon,  # longitude
        500,
        500,
        "2023",
        0
    )

    lat, lon = dataset.sat_utils.pixel_to_geo_coordinates(
        x_sat + x_offset, y_sat + y_offset, patch_transform
    )

    print("Query lat: ", query_lat)
    print("Query lon: ", query_lon)
    print("Retrieved lat: ", lat)
    print("Retrieved lon: ", lon)

    lat, lon = dataset.sat_utils.pixel_to_geo_coordinates(x_sat, y_sat, patch_transform)
    print("Retrieved lat: ", lat)
    print("Retrieved lon: ", lon)

    pytest.approx(lat, query_lat, abs=1e-5)
    pytest.approx(lon, query_lon, abs=1e-5)
    print("Test passed.")
    #import matplotlib.pyplot as plt

    ## Plot with the query point
    #plt.imshow(sat_patch.transpose(1, 2, 0))
    #plt.scatter(x_sat, y_sat, c="r")
    #plt.savefig("test2023.png")
    #exit()

    ## Plot the satellite patch and the point on the satellite patch
    for i, (uav_image, img_info, satellite_patch, heatmap) in enumerate(dataloader):
        print(img_info)


if __name__ == "__main__":
    test()
