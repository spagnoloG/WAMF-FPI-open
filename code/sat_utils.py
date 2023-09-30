#! /usr/bin/env python3

import os
import mercantile
import rasterio
import requests
import time
import numpy as np
import random
import warnings
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.merge import merge
from PIL import Image
import gc


class SatUtils:
    def __init__(
        self, satellite_dataset_dir: str, sat_zoom_level: int, heatmap_kernel_size: int
    ):
        self.satellite_dataset_dir = satellite_dataset_dir
        self.sat_zoom_level = sat_zoom_level
        self.heatmap_kernel_size = heatmap_kernel_size

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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
            for neighbor in neighbors:
                west, south, east, north = mercantile.bounds(neighbor)
                tile_path = f"{self.satellite_dataset_dir}/{neighbor.z}_{neighbor.x}_{neighbor.y}.jpg"
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

    def get_random_tiff_patch(
        self,
        lat: float,
        lon: float,
        patch_width: int,
        patch_height: int,
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

        patch_transform = rasterio.transform.Affine(
            transform.a,
            transform.b,
            transform.c + x_offset * transform.a + y_offset * transform.b,
            transform.d,
            transform.e,
            transform.f + x_offset * transform.d + y_offset * transform.e,
        )

        return patch, x, y, x_offset, y_offset, patch_transform

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

    def pixel_to_geo_coordinates(
        self, x: int, y: int, transform: rasterio.transform.Affine
    ) -> (float, float):
        """
        Converts a pair of pixel coordinates to (lat, lon) coordinates.
        """
        lon, lat = transform * (x, y)
        return lat, lon
