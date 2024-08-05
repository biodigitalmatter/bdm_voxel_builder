import os
from datetime import datetime
from pathlib import Path

import compas.geometry as cg
import numpy as np
import numpy.typing as npt
from compas.data import json_dump

from bdm_voxel_builder import TEMP_DIR


def _get_timestamp() -> str:
    return datetime.now().strftime("%F_%H_%M_%S")


def get_savepath(dir: Path, suffix: str, note: str = None):
    filename = _get_timestamp()

    if note:
        filename += f"_{note}"

    filename += suffix

    return Path(dir) / filename


def get_nth_newest_file_in_folder(folder_path: os.PathLike, n = 0):
    folder_path = Path(folder_path)

    # Get a list of files in the folder
    files = list(folder_path.glob('*'))

    # Sort the files by change time (modification time) in descending order
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return files[n]


def save_pointcloud(
    pointcloud: cg.Pointcloud, values: list[float] = None, note: str = None
):
    dict_ = {"pointcloud": pointcloud, "values": values}

    json_dump(dict_, get_savepath(TEMP_DIR, ".json", note=note))



def save_ndarray(arr: npt.NDArray, note: str = None):
    np.save(get_savepath(TEMP_DIR, ".npy", note=note), arr)
