import json

import numpy as np
import numpy.typing as npt

from bdm_voxel_builder import TEMP_DIR
from bdm_voxel_builder.helpers.savepaths import get_savepath


def convert_array_to_points(a, list_output=False):
    indicies = np.indices(a.shape)
    pt_location = np.logical_not(a == 0)
    coordinates = []
    for i in range(3):
        c = indicies[i][pt_location]
        coordinates.append(c)
    if list_output:
        pts = np.vstack(coordinates).transpose().tolist()
    else:
        pts = np.vstack(coordinates).transpose()
    return pts


def save_ndarray(arr: npt.NDArray, note: str = None):
    np.save(get_savepath(TEMP_DIR, ".npy", note=note), arr)


def sort_pts_by_values(arr: npt.NDArray, multiply=1):
    """returns sortedpts, values"""
    indicies = np.indices(arr.shape)
    pt_location = np.logical_not(arr == 0)
    coordinates = []
    for i in range(3):
        c = indicies[i][pt_location]
        coordinates.append(c)
    pts = np.vstack(coordinates).transpose().tolist()
    values = arr[pt_location].tolist()
    # sort:

    # Pair the elements using zip
    paired_lists = list(zip(values, pts))

    # Sort the paired lists based on the first element of the pairs (values values)
    sorted_paired_lists = sorted(paired_lists, key=lambda x: x[0])

    # Extract the sorted nested list

    sortedpts = [element[1] for element in sorted_paired_lists]
    values = [element[0] * multiply for element in sorted_paired_lists]

    return sortedpts, values


def convert_array_to_points(a, list_output=False):
    indicies = np.indices(a.shape)
    pt_location = np.logical_not(a == 0)
    coordinates = []
    for i in range(3):
        c = indicies[i][pt_location]
        coordinates.append(c)
    if list_output:
        pts = np.vstack(coordinates).transpose().tolist()
    else:
        pts = np.vstack(coordinates).transpose()
    return pts
