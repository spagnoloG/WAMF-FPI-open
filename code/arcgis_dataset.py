#!/usr/bin/env python3

import json
import torch
import os
from typing import Literal, List
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
from arcgis_sat_utils import SatUtils
import cv2
import warnings
import random


class GeoLocalizationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        uav_dataset_dir: str,
        satellite_dataset_dir: str,
        misslabeled_images_path: str,
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
        rotation_angles: List[float] = [0, 45, 90, 135, 180, 225, 270, 315],
        uav_image_scale: float = 1,
        use_heatmap: bool = True,
        subset_size: int = -1,
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
        self.misslabeled_images_path = misslabeled_images_path
        self.total_uav_samples = self.count_total_uav_samples()
        self.misslabelled_images = self.read_misslabelled_images(self.misslabeled_images_path)
        self.entry_paths = self.get_entry_paths(self.uav_dataset_dir)
        self.cleanup_misslabelled_images()
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.transform_mean, self.transform_std),
            ]
        )
        self.sat_available_years = sat_available_years
        self.rotation_angles = rotation_angles
        self.uav_image_scale = uav_image_scale
        self.use_heatmap = use_heatmap
        self.sat_patch_width = sat_patch_width
        self.sat_patch_height = sat_patch_height
        self.subset_size = subset_size

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

        if self.subset_size > 0:
            ssize = int(
                self.subset_size
                / len(self.rotation_angles)
                / len(self.sat_available_years)
            )
            ssize = min(ssize, len(self.entry_paths))
            self.entry_paths = self.entry_paths[:ssize]

    def read_misslabelled_images(
        self, path: str = "misslabels/misslabeled.txt"
    ) -> List[str]:
        with open(path, "r") as f:
            lines = f.readlines()
        return [line.strip() for line in lines]

    def cleanup_misslabelled_images(self) -> None:
        indices_to_delete = []

        for image in self.misslabelled_images:
            for image_path in self.entry_paths:
                if image in image_path:
                    index = self.entry_paths.index(image_path)
                    indices_to_delete.append(index)
                    break

        sorted_tuples = sorted(indices_to_delete, reverse=True)

        for index in sorted_tuples:
            self.entry_paths.pop(index)

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
        rot_angle = self.rotation_angles[
            (idx // len(self.sat_available_years)) % len(self.rotation_angles)
        ]

        image_path = self.entry_paths[image_path_index]
        uav_image = Image.open(image_path).convert("RGB")  # Ensure 3-channel image

        original_uav_image_width = uav_image.width
        original_uav_image_height = uav_image.height

        lookup_str, file_number = self.extract_info_from_filename(image_path)
        img_info = self.metadata_dict[lookup_str][file_number]

        lat, lon = (
            img_info["coordinate"]["latitude"],
            img_info["coordinate"]["longitude"],
        )

        uav_rot_x, uav_rot_y, uav_rot_z = (
            img_info["rotation"]["x"],
            img_info["rotation"]["y"],
            img_info["rotation"]["z"],
        )

        fov_vertical = img_info["fovVertical"]

        try:
            agl_altitude = float(image_path.split("/")[-1].split("_")[2].split("m")[0])
        except IndexError:
            agl_altitude = 150.0
            warnings.warn(
                "Could not extract AGL altitude from filename, using default value of 150m."
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

        uav_image = F.rotate(uav_image, rot_angle)
        uav_image = F.resize(uav_image, [h, w])
        uav_image = F.center_crop(
            uav_image, (self.uav_patch_height, self.uav_patch_width)
        )
        uav_image = self.transforms(uav_image)

        satellite_patch = satellite_patch.transpose(1, 2, 0)
        satellite_patch = self.transforms(satellite_patch)

        if self.use_heatmap:
            heatmap = self.get_heatmap_gt(
                x_sat,
                y_sat,
                satellite_patch.shape[1],
                satellite_patch.shape[2],
                self.heatmap_kernel_size,
            )

        cropped_uav_image_width = self.calculate_cropped_uav_image_width(
            fov_vertical,
            original_uav_image_width,
            original_uav_image_height,
            self.uav_patch_width,
            self.uav_patch_height,
            agl_altitude,
        )

        satellite_tile_width = self.calculate_cropped_sat_image_width(
            lat, self.sat_patch_width, patch_transform
        )

        scale_factor = cropped_uav_image_width / satellite_tile_width
        scale_factor *= 10

        homography_matrix = self.compute_homography(
            rot_angle,
            x_sat,
            y_sat,
            self.uav_patch_width,
            self.uav_patch_height,
            scale_factor,
        )

        if not self.use_heatmap:
            # Sample four points from the UAV image
            points = self.sample_four_points(
                self.uav_patch_width, self.uav_patch_height
            )

            # Transform the points
            warped_points = self.warp_points(points, homography_matrix)
            img_info["warped_points_sat"] = warped_points
            img_info["warped_points_uav"] = points

        img_info["cropped_uav_image_width"] = cropped_uav_image_width
        img_info["satellite_tile_width"] = satellite_tile_width
        img_info["scale_factor"] = scale_factor
        img_info["filename"] = image_path
        img_info["rot_angle"] = rot_angle
        img_info["x_sat"] = x_sat
        img_info["y_sat"] = y_sat
        img_info["x_offset"] = x_offset
        img_info["y_offset"] = y_offset
        img_info["patch_transform"] = patch_transform
        img_info["uav_image_scale"] = self.uav_image_scale
        img_info["homography_matrix_uav_to_sat"] = homography_matrix
        img_info["homography_matrix_sat_to_uav"] = np.linalg.inv(homography_matrix)
        img_info["agl_altitude"] = agl_altitude
        img_info["original_uav_image_width"] = original_uav_image_width
        img_info["original_drone_image_height"] = original_uav_image_height
        img_info["fov_vertical"] = fov_vertical

        if self.use_heatmap:
            return uav_image, img_info, satellite_patch, heatmap
        else:
            return uav_image, img_info, satellite_patch, warped_points

    def calculate_cropped_sat_image_width(self, latitude, patch_width, patch_transform):
        """
        Computes the width of the satellite image in world coordinate units
        """
        length_of_degree = 111320 * np.cos(np.radians(latitude))
        scale_x_meters = patch_transform[0] * length_of_degree
        satellite_tile_width = patch_width * scale_x_meters
        return satellite_tile_width

    def calculate_cropped_uav_image_width(
        self,
        fov_vertical,
        orig_width,
        orig_height,
        crop_width,
        crop_height,
        altitude=150.0,
    ):
        """
        Computes the width of the UAV image in world coordinate units
        """
        # Convert fov from degrees to radians
        fov_rad = np.radians(fov_vertical)

        # Calculate the full width of the UAV image
        full_width = 2 * (altitude * np.tan(fov_rad / 2))

        # Determine the cropping ratio
        crop_ratio_width = crop_width / orig_width
        crop_ratio_height = crop_height / orig_height

        # Calculate the adjusted horizontal fov
        fov_horizontal = 2 * np.arctan(np.tan(fov_rad / 2) * (orig_width / orig_height))
        adjusted_fov_horizontal = 2 * np.arctan(
            np.tan(fov_horizontal / 2) * crop_ratio_width
        )

        # Calculate the new full width using the adjusted horizontal fov
        full_width = 2 * (altitude * np.tan(adjusted_fov_horizontal / 2))

        # Adjust the width according to the crop ratio
        cropped_width = full_width * crop_ratio_width

        return cropped_width

    def compute_homography(
        self, rot_angle, x_sat, y_sat, uav_width, uav_height, scale_factor
    ):
        # Adjust rot_angle if it's greater than 180 degrees
        if rot_angle > 180:
            rot_angle -= 360
        # Convert rotation angle to radians
        theta = np.radians(rot_angle)

        # Rotation matrix
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

        # Scale matrix
        S = np.array([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])

        # Translation matrix to center the UAV image
        T_uav = np.array([[1, 0, -uav_width / 2], [0, 1, -uav_height / 2], [0, 0, 1]])

        # Translation matrix to move to the satellite image position
        T_sat = np.array([[1, 0, x_sat], [0, 1, y_sat], [0, 0, 1]])

        # Compute the combined homography matrix
        H = np.dot(T_sat, np.dot(R, np.dot(S, T_uav)))

        return H

    def sample_four_points(self, width: int, height: int) -> np.ndarray:
        """
        Samples four points from the UAV image.
        """
        PADDING = 10
        points = np.array(
            [
                [
                    random.randint(0, width - PADDING),
                    random.randint(0, height - PADDING),
                ]
                for _ in range(4)
            ]
        )
        return points

    def warp_points(
        self, points: np.ndarray, homography_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Warps the given points using the given homography matrix.
        """
        points = np.array(points)
        points = np.concatenate([points, np.ones((4, 1))], axis=1)
        points = np.dot(homography_matrix, points.T).T
        points = points[:, :2] / points[:, 2:]
        return points

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
    import matplotlib.pyplot as plt

    dataset = GeoLocalizationDataset(
        uav_dataset_dir="/home/spagnologasper/Documents/projects/uav-localization-experiments/drone_dataset",
        satellite_dataset_dir="/home/spagnologasper/Documents/projects/historical_satellite_tiles_downloader/tiles",
        sat_available_years=["2023", "2021", "2019", "2016"],
        rotation_angles=[1],
        # rotation_angles=[
        #    0,
        #    22.5,
        #    45,
        #    67.5,
        #    90,
        #    112.5,
        #    135,
        #    157.5,
        #    180,
        #    202.5,
        #    225,
        #    247.5,
        #    270,
        #    292.5,
        #    315,
        #    337.5,
        # ],
        dataset="train",
        misslabeled_images_path="misslabels/misslabeled.txt",
        sat_zoom_level=17,
        uav_patch_width=400,
        uav_patch_height=400,
        sat_patch_width=400,
        sat_patch_height=400,
        heatmap_kernel_size=33,
        test_from_train_ratio=0.1,
        transform_mean=[0.485, 0.456, 0.406],
        transform_std=[0.229, 0.224, 0.225],
        uav_image_scale=2.0,
        use_heatmap=False,
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
        query_lat, query_lon, 500, 500, "2023", 0  # latitude  # longitude
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
    # import matplotlib.pyplot as plt

    ## Plot with the query point
    # plt.imshow(sat_patch.transpose(1, 2, 0))
    # plt.scatter(x_sat, y_sat, c="r")
    # plt.savefig("test2023.png")
    # exit()

    ## Plot the satellite patch and the point on the satellite patch
    for i, (uav_images, img_info, satellite_patches, warped_points_batch) in enumerate(
        dataloader
    ):
        for j, (uav_image, satellite_patch, warped_points) in enumerate(
            zip(uav_images, satellite_patches, warped_points_batch)
        ):
            x_sat = img_info["x_sat"][j].item()
            y_sat = img_info["y_sat"][j].item()
            angle = img_info["rot_angle"][j].item()
            homography_matrix = img_info["homography_matrix_uav_to_sat"][j].numpy()

            # Inverse transform the satellite patch and UAV image
            sp = dataset.inverse_transforms(satellite_patch)
            uav_i = dataset.inverse_transforms(uav_image)

            # Convert PIL images to NumPy arrays if necessary
            sp_array = np.array(sp)
            uav_i_array = np.array(uav_i)

            # Dimensions of the satellite patch
            h, w = sp_array.shape[:2]

            # Warp the UAV image using the homography matrix
            uav_warped = cv2.warpPerspective(uav_i_array, homography_matrix, (w, h))

            # Create a 2x2 subplot
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))

            # Plot satellite patch and point on the first subplot
            ax[0, 0].imshow(sp)
            ax[0, 0].scatter(x_sat, y_sat, c="r")
            ax[0, 0].set_title("Satellite Patch with Point")

            # Plot UAV image on the second subplot
            ax[0, 1].imshow(uav_i)
            ax[0, 1].set_title("UAV Image")

            # Plot warped UAV image on the satellite patch on the third subplot
            ax[1, 0].imshow(sp)
            ax[1, 0].imshow(uav_warped, alpha=0.5)  # Adjust alpha for transparency
            ax[1, 0].set_title("Warped UAV Image on Satellite Patch")

            # Plot satellite patch with warped points on the fourth subplot
            ax[1, 1].imshow(sp)
            ax[1, 1].scatter(
                *warped_points.T, c="yellow", marker="o"
            )  # Plotting warped points
            ax[1, 1].set_title("Satellite Patch with Warped Points")

            # Save the figure
            os.makedirs("outs", exist_ok=True)
            plt.savefig(f"outs/test_{i}_{j}.png")
            plt.close(fig)  # Close the figure to free memory


if __name__ == "__main__":
    test()
