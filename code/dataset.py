#!/usr/bin/env python3

import json
import torch
import os
from typing import Literal, List
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
import mercantile
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.merge import merge
import rasterio
import time
import requests
import gc
import random


class GeoLocalizationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        uav_dataset_dir: str,
        satellite_dataset_dir: str,
        dataset: Literal["train", "test"],
        sat_zoom_level: int = 16,
        uav_patch_width: int = 128,
        uav_patch_height: int = 128,
        heatmap_kernel_size: int = 33,
        test_from_train_ratio: float = 0.0,
        uav_scales: List[float] = [1.0],
        transform_mean: List[float] = [0.485, 0.456, 0.406],
        transform_std: List[float] = [0.229, 0.224, 0.225],
    ):
        self.uav_dataset_dir = uav_dataset_dir
        self.satellite_dataset_dir = satellite_dataset_dir
        self.dataset = dataset
        self.sat_zoom_level = sat_zoom_level
        self.uav_patch_width = uav_patch_width
        self.uav_patch_height = uav_patch_height
        self.heatmap_kernel_size = heatmap_kernel_size
        self.test_from_train_ratio = test_from_train_ratio
        self.uav_scales = uav_scales
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

        self.headers = {
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Referer": "https://www.openstreetmap.org/",
            "Sec-Fetch-Dest": "image",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "cross-site",
            "Sec-GPC": "1",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "sec-ch-ua": '"Not.A/Brand";v="8", "Chromium";v="114", "Brave";v="114"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
        }

        self.params = {
            "access_token": "pk.eyJ1Ijoib3BlbnN0cmVldG1hcCIsImEiOiJjbGRlaGp1b3gwOGRtM250NW9sOHhuMmRjIn0.Y3mM21ciEP5Zo5amLJUugg",
        }

    def __len__(self) -> int:
        return len(self.entry_paths) * len(self.uav_scales)

    def __getitem__(self, idx) -> (torch.Tensor, dict, torch.Tensor, torch.Tensor):
        """
        Retrieves a sample given its index, returning the preprocessed UAV
        and satellite images, along with their associated heatmap and metadata.
        """

        image_path = self.entry_paths[idx // len(self.uav_scales)]
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
            zoom_level,
        ) = self.get_random_tiff_patch(
            lat, lon, 400, 400
        )  # TODO: make patch size a parameter

        uav_image_scale = self.uav_scales[(idx % len(self.uav_scales))]

        # Rotate crop center and transform image
        h = np.ceil(uav_image.height // uav_image_scale).astype(int)
        w = np.ceil(uav_image.width // uav_image_scale).astype(int)

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
        img_info["zoom_level"] = zoom_level
        img_info["uav_image_scale"] = uav_image_scale

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
                if "Test" in entry_path or (
                    number < images_to_take_per_folder and "Train" in entry_path
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

    #### Satellite image methods ####

    def download_missing_tile(self, tile: mercantile.Tile) -> None:
        """
        Downloads the given tile from Mapbox and saves it to the satellite dataset directory.
        """
        os.makedirs(f"{self.satellite_dataset_dir}", exist_ok=True)
        file_path = f"{self.satellite_dataset_dir}/{tile.z}_{tile.x}_{tile.y}.jpg"

        if os.path.exists(file_path):
            return

        max_attempts = 5
        for attempt in range(max_attempts):
            print(
                f"Downloading tile {tile.z}_{tile.x}_{tile.y} (attempt {attempt + 1}/{max_attempts})..."
            )
            try:
                response = requests.get(
                    f"https://c.tiles.mapbox.com/v4/mapbox.satellite/{tile.z}/{tile.x}/{tile.y}@2x.jpg",
                    params=self.params,
                    headers=self.headers,
                )
                response.raise_for_status()  # raises a Python exception if the response contains an HTTP error status code
            except (
                requests.exceptions.RequestException,
                requests.exceptions.ConnectionError,
            ) as e:
                if attempt < max_attempts - 1:  # i.e., if it's not the final attempt
                    print("Error downloading tile. Retrying...")
                    time.sleep(5)  # wait for 5 seconds before trying again
                    continue
                else:
                    print("Error downloading tile. Max retries exceeded.")
                    break
            else:  # executes if the try block didn't throw any exceptions
                with open(file_path, "wb") as f:
                    f.write(response.content)
                break
        else:
            print("Error downloading tile. Max retries exceeded.")

    def get_tiff_map(self, tile: mercantile.Tile) -> (np.ndarray, dict):
        """
        Returns a TIFF map of the given tile.
        """
        tile_data = []
        neighbors = mercantile.neighbors(tile)
        neighbors.append(tile)

        for neighbor in neighbors:
            found = False
            west, south, east, north = mercantile.bounds(neighbor)
            tile_path = f"{self.satellite_dataset_dir}/{neighbor.z}_{neighbor.x}_{neighbor.y}.jpg"
            if os.path.exists(tile_path):
                found = True
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

                del memfile

            if not found:
                self.download_missing_tile(neighbor)
                time.sleep(1)
                tile_path = f"{self.satellite_dataset_dir}/{neighbor.z}_{neighbor.x}_{neighbor.y}.jpg"
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
                del memfile

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

    def get_random_tiff_patch(
        self, lat: float, lon: float, patch_width: int, patch_height: int
    ) -> (np.ndarray, int, int, int, int, rasterio.transform.Affine):
        """
        Returns a random patch from the satellite image.
        """

        tile = self.get_tile_from_coord(lat, lon, self.sat_zoom_level)
        mosaic, out_meta = self.get_tiff_map(tile)

        # plot the mosaic
        transform = out_meta["transform"]

        x_pixel, y_pixel = self.geo_to_pixel_coordinates(lat, lon, transform)

        ks = self.heatmap_kernel_size // 2

        x_offset_range = [
            x_pixel - patch_width + ks + 1,
            x_pixel - ks - 1,
        ]
        y_offset_range = [
            y_pixel - patch_height + ks + 1,
            y_pixel - ks - 1,
        ]

        # Randomly select an offset within the valid range
        x_offset = random.randint(*x_offset_range)
        y_offset = random.randint(*y_offset_range)

        x_offset = np.clip(x_offset, 0, mosaic.shape[-1] - patch_width)
        y_offset = np.clip(y_offset, 0, mosaic.shape[-2] - patch_height)

        # Update x, y to reflect the clamping of x_offset and y_offset
        x, y = x_pixel - x_offset, y_pixel - y_offset
        patch = mosaic[
            :, y_offset : y_offset + patch_height, x_offset : x_offset + patch_width
        ]

        return patch, x, y, x_offset, y_offset, transform

    def get_tile_from_coord(
        self, lat: float, lng: float, zoom_level: int
    ) -> mercantile.Tile:
        """
        Returns the tile containing the given coordinates.
        """
        tile = mercantile.tile(lng, lat, zoom_level)
        return tile

    def geo_to_pixel_coordinates(
        self, lat: float, lon: float, transform: rasterio.transform.Affine
    ) -> (int, int):
        """
        Converts a pair of (lat, lon) coordinates to pixel coordinates.
        """
        x_pixel, y_pixel = ~transform * (lon, lat)
        return round(x_pixel), round(y_pixel)


def test():
    dataset = GeoLocalizationDataset(
        uav_dataset_dir="/home/spagnologasper/Documents/uav-localization-experiments/drone_dataset",
        satellite_dataset_dir="/home/spagnologasper/Documents/uav-localization-experiments/satellite_dataset",
        dataset="train",
        sat_zoom_level=16,
        uav_patch_width=128,
        uav_patch_height=128,
        heatmap_kernel_size=33,
        test_from_train_ratio=0.0,
        uav_scales=[1.0],
        transform_mean=[0.485, 0.456, 0.406],
        transform_std=[0.229, 0.224, 0.225],
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1
    )

    (
        sat_patch,
        x_sat,
        y_sat,
        x_offset,
        y_offset,
        transform,
    ) = dataset.get_random_tiff_patch(
        46.048992,
        14.509215,
        400,
        400,
    )

    import matplotlib.pyplot as plt

    ## Plot the satellite patch and the point on the satellite patch
    # fig, axs = plt.subplots(1, 1, figsize=(20, 6))
    # axs.imshow(sat_patch.transpose(1, 2, 0))
    # axs.scatter(x_sat, y_sat, c="r")
    # axs.set_title("Satellite patch")
    # plt.savefig("output.png")

    for i, (uav_image, img_info, satellite_patch, heatmap) in enumerate(dataloader):
        print(i)
        print(uav_image.shape)
        print(img_info)
        print(satellite_patch.shape)
        print(heatmap.shape)


if __name__ == "__main__":
    test()
