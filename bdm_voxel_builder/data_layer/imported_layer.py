import os
from compas.geometry import Box
import numpy as np
import pyopenvdb as vdb

from bdm_voxel_builder.data_layer.base import DataLayer


class ImportedLayer(DataLayer):
    @classmethod
    def from_vdb(
        cls, grid: os.PathLike | vdb.GridBase, name: str = None
    ):
        if isinstance(grid, os.PathLike):
            grids = vdb.readAllGridMetadata(str(grid))

            if not name and len(grids) > 1:
                print(
                    f"File contains more than one grid, only processing first named {grids[0].name}"
                )

            name = name or grids[0].name
            grid = vdb.read(str(grid), name)

        bbox_min = grid.metadata["file_bbox_min"]
        bbox_max = grid.metadata["file_bbox_max"]

        shape = np.array(bbox_max) - np.array(bbox_min)
        arr = np.zeros(shape)

        grid.copyToArray(arr, ijk=bbox_min)

        return cls(
            name=name or grid.name,
            bbox=Box.from_diagonal((bbox_min, bbox_max)),
            array=arr,
        )
