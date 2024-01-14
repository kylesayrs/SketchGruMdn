from typing import Optional, List, Tuple

import os
import json
import tqdm
import random


def load_file_drawings(
    strokes_dir: str,
    file_name: str,
    index_lookup,
    all_drawings_strokes,
    progress: Optional[tqdm.tqdm] = None
): 
    file_path = os.path.join(strokes_dir, file_name)
    drawings = load_drawings(file_path, sparsity=20)

    num_prev_drawings = len(all_drawings_strokes)
    indices = [
        [num_prev_drawings + drawing_index, stroke_index]
        for drawing_index in range(len(drawings))
        for stroke_index in range(len(drawings[drawing_index]))
    ]

    index_lookup.extend(indices)
    all_drawings_strokes.extend(drawings)

    if progress is not None:
        progress.update(1)


def load_drawings_strokes(
    strokes_dir: str
) -> Tuple[
        List[List[List[float]]],
        List[List[int]]
    ]:
    """
    all_drawings_strokes[drawing_index][stroke_index] = (x, y, pen_down)
    index_lookup[index] = (drawing_index, stroke_index)

    Future work could multiprocess this function

    :param strokes_dir: TODO
    :return: TODO
    """
    all_drawings_strokes = []
    index_lookup = []

    file_names = os.listdir(strokes_dir)
    progress = tqdm.tqdm(desc="Classes loaded", total=len(file_names))
    for _file_index, file_name in enumerate(file_names):
        load_file_drawings(strokes_dir, file_name, index_lookup, all_drawings_strokes, progress)

    return all_drawings_strokes, index_lookup


def split_drawings_strokes(
    index_lookup: List[Tuple[int, int]],
    test_size: float,
    shuffle: bool = True
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    indices = list(range(len(index_lookup)))
    if shuffle:
        random.shuffle(indices)

    split_index = int(test_size * len(index_lookup))
    test_indices = indices[:split_index]
    train_indices = indices[split_index:]

    test_index_lookup = [index_lookup[index] for index in test_indices]
    train_index_lookup = [index_lookup[index] for index in train_indices]

    return train_index_lookup, test_index_lookup


def load_drawings(file_path: str, sparsity: int = 1) -> List[List[int]]:
    drawings = []

    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines[::sparsity]:
            data = json.loads(line)
            if data["recognized"]:
                drawings.append(data["drawing"])

    return drawings
