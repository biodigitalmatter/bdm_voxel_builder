import math
import os

import numpy as np
import pyopenvdb as vdb
from compas.geometry import Box

from bdm_voxel_builder.data_layer.base import DataLayer
from bdm_voxel_builder.helpers.vdb import xform_to_compas


class ImportedLayer(DataLayer):
    @classmethod
    def from_npy(cls, path: os.PathLike, name: str = None):
        arr = np.load(path)
        return cls(name=name, array=arr)

    @classmethod
    def from_vdb(cls, grid: os.PathLike | vdb.GridBase, name: str = None):
        if isinstance(grid, os.PathLike):
            grids = vdb.readAllGridMetadata(str(grid))

            if not name and len(grids) > 1:
                print(
                    "File contains more than one grid, ",
                    f"only processing first named {grids[0].name}"
                )

            name = name or grids[0].name
            grid = vdb.read(str(grid), name)

        bbox_min = grid.metadata["file_bbox_min"]
        bbox_max = grid.metadata["file_bbox_max"]

        shape = np.array(bbox_max) - np.array(bbox_min)
        arr = np.zeros(shape)

        # rotate the grid to make Z up
        grid.transform.rotate(math.pi / 2, vdb.Axis.X)

        grid.copyToArray(arr, ijk=bbox_min)

        return cls(
            name=name or grid.name,
            bbox=Box.from_diagonal((bbox_min, bbox_max)),
            array=arr,
            xform=xform_to_compas(grid.transform),
        )
