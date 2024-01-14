from typing import Optional

import os
import tqdm
import random
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor


parser = argparse.ArgumentParser()
parser.add_argument("output_dir_path")
parser.add_argument("--parallel", default=False)
parser.add_argument("--keyword", default=None)


def read_stroke_paths():
    all_paths = subprocess.Popen(
        "gsutil -m ls 'gs://quickdraw_dataset/full/simplified'",
        shell=True,
        stdout=subprocess.PIPE
    ).stdout.read()
    all_paths = all_paths.decode("utf-8")
    all_paths = all_paths.split("\n")
    all_paths = all_paths[:-1]

    return all_paths


def download_strokes(
    category_path: str,
    output_dir_path: str,
    progress: Optional[tqdm.std.tqdm]
):
    category_name = category_path.split("/")[-1].split(".")[0]
    category_name = category_name.replace(" ", "\ ")

    save_path = os.path.join(output_dir_path, f"{category_name}.ndjson")
    if os.path.exists(save_path):
        progress.update(1)
        return

    process = subprocess.Popen(f"gsutil -m cp '{category_path}' {save_path}", shell=True)
    process.wait()

    progress.update(1)

if __name__ == "__main__":
    args = parser.parse_args()

    paths = read_stroke_paths()
    if args.keyword is not None:
        paths = [path for path in paths if args.keyword in path]
    random.shuffle(paths)

    progress = tqdm.tqdm(total=len(paths))

    os.makedirs(args.output_dir_path, exist_ok=True)

    if args.parallel:
        with ThreadPoolExecutor(max_workers=None) as executor:
            futures = [
                executor.submit(
                    download_strokes,
                    category_path,
                    args.output_dir_path,
                    progress
                )
                for category_path in paths
            ]

    else:
        for category_path in paths:
            download_strokes(
                category_path,
                args.output_dir_path,
                progress
            )
