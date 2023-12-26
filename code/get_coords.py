#!/usr/bin/env python3

import json
import os


def get_entries(dataset_path: str):
    entries = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".json"):
                entries.append(os.path.join(root, file))
    return entries


def main(dataset_path: str):
    with open("coords.txt", "w") as ff:
        for entry in get_entries(dataset_path):
            with open(entry, "r") as f:
                data = json.load(f)

                for entry in data["cameraFrames"]:
                    ff.write(
                        f"lat: {entry['coordinate']['latitude']}, lon: {entry['coordinate']['longitude']}"
                        + "\n"
                    )


if __name__ == "__main__":
    main(dataset_path="../drone_dataset/")
