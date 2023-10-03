#! /usr/bin/env python3

from argparse import ArgumentParser
import os
import json
from tqdm import tqdm


def main(pred_dir: str):
    all_files = os.listdir(pred_dir)
    all_files = [f for f in all_files if f.endswith(".json")]
    distances = []

    for filename in all_files:
        with open(os.path.join(pred_dir, filename), "r") as f:
            data = json.load(f)
            metre_distance = data.get("metre_distance", None)
            if metre_distance is not None:
                distances.append((filename, metre_distance))

    # Sort the list by metre_distance in descending order
    sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)

    for index, (json_file, _) in tqdm(enumerate(sorted_distances)):
        new_json_name = f"{index}.json"
        os.rename(
            os.path.join(pred_dir, json_file), os.path.join(pred_dir, new_json_name)
        )

        image_file = json_file.replace(".json", ".png")
        new_image_name = f"{index}.png"
        os.rename(
            os.path.join(pred_dir, image_file), os.path.join(pred_dir, new_image_name)
        )


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("pred_dir", type=str, help="Path to predictions directory")
    args = argparser.parse_args()
    main(args.pred_dir)
