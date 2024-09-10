import math
import os
from collections.abc import Sequence
from copy import deepcopy
from typing import Self

import compas.geometry as cg
import numpy as np
import numpy.typing as npt
import pyopenvdb as vdb
import pypcd4
from compas.colors import Color

from bdm_voxel_builder.helpers import get_indices_from_map_and_origin


class Grid:
    def __init__(
        self,
        name: str = None,
        clipping_box: cg.Box = None,
        xform: cg.Transformation = None,
        color: Color = None,
        grid: vdb.GridBase = None,
    ):
        self.clipping_box = clipping_box
        self.name = name
        self.xform = xform or cg.Transformation()
        self.color = color or Color.black()

        self.vdb = grid or vdb.FloatGrid()

        if self.name:
            self.vdb.name = self.name

    @property
    def clipping_box(self):
        if not self._clipping_box:
            diagonal = self.vdb.evalActiveVoxelBoundingBox()
            self._clipping_box = cg.Box.from_diagonal(diagonal)
        return self._clipping_box

    @clipping_box.setter
    def clipping_box(self, value):
        if value is None or isinstance(value, cg.Box):
            self._clipping_box = value
        else:
            if isinstance(value, Sequence):
                diagonal = ([0, 0, 0], [v - 1 for v in value])
            elif isinstance(value, int | float):
                diagonal = ([0, 0, 0], [value - 1] * 3)
            else:
                raise ValueError("Invalid clipping box value")

            self._clipping_box = cg.Box.from_diagonal(diagonal)

    def get_value(self, ijk: tuple[int, int, int]):
        return self.vdb.getConstAccessor().getValue(ijk)

    def get_values(self, indices: list[tuple[int, int, int]]):
        accessor = self.vdb.getConstAccessor()
        return [accessor.getValue(ijk) for ijk in indices]

    def set_value(self, ijk: tuple[int, int, int], value: float):
        self.vdb.getAccessor().setValueOn(ijk, value)

    def set_values(
        self, indices: list[tuple[int, int, int]], values: float | np.ndarray[np.float_]
    ):
        accessor = self.vdb.getAccessor()

        if not isinstance(values, Sequence):
            values = np.full(len(indices), values)

        for index, value in zip(indices, values, strict=True):
            accessor.setValueOn(index, value)

    def set_value_by_index_map(
        self, index_map: np.ndarray, origin: npt.ArrayLike, value=1
    ):
        indices = get_indices_from_map_and_origin(index_map, origin)
        self.set_values(indices, value)

    def set_values_by_array(self, array: np.ndarray, origin=None):
        self.vdb.copyFromArray(array=array, ijk=(origin))

    def set_values_in_zone_xxyyzz(
        self, zone_xxyyzz: tuple[int, int, int, int, int, int], value=1.0
    ):
        """add or replace values within zone (including both end)
        add_values == True: add values in self.array
        add_values == False: replace values in self.array *default
        input:
            zone_xxyyzz = [x_start, x_end, y_start, y_end, z_start, z_end]
        """
        bmin, bmax = zone_xxyyzz[:3], zone_xxyyzz[3:]
        self.vdb.fill(bmin, bmax, value, active=True)

    def get_active_voxels(self) -> npt.NDArray:
        """returns indices of nonzero values
        list of coordinates
            shape = [3,n]"""
        return np.array([item.min for item in self.vdb.citerOnValues()], dtype=np.int_)

    def get_number_of_active_voxels(self) -> int:
        return self.vdb.activeVoxelCount()

    def get_pointcloud(self):
        return cg.Pointcloud(self.get_active_voxels())

    def get_world_pts(self) -> list[list[float]]:
        return cg.transform_points_numpy(self.get_active_voxels(), self.xform)

    def get_world_pointcloud(self) -> cg.Pointcloud:
        return cg.Pointcloud(self.get_world_pts())

    def merge_with(self, other: Self):
        if not isinstance(other, Sequence):
            other = [other]

        for grid in other:
            vdb = grid.vdb.deepCopy()
            self.vdb.combine(vdb, lambda a, b: max(a, b))

    def merged_with(self, other: Self):
        new = deepcopy(self)
        new.merge_with(other)
        return new

    def to_numpy(self):
        arr = np.zeros([int(s) for s in self.clipping_box.dimensions])

        self.vdb.copyToArray(arr)

        return arr

    @classmethod
    def from_numpy(
        cls,
        arr: npt.NDArray | os.PathLike,
        clipping_box: cg.Box = None,
        **kwargs,
    ):
        if isinstance(arr, os.PathLike):
            arr = np.load(arr)

        indices = np.array(np.nonzero(arr)).transpose()

        grid = vdb.FloatGrid()
        accessor = grid.getAccessor()

        for ijk in indices:
            accessor.setValueOn(ijk, 1)

        return cls(
            clipping_box=clipping_box or cg.Box.from_diagonal(([0, 0, 0], arr.shape)),
            grid=grid,
            **kwargs,
        )

    @classmethod
    def from_vdb(
        cls,
        grid: os.PathLike | vdb.GridBase,
        name: str = None,
        xform: cg.Transformation = None,
        **kwargs,
    ):
        if isinstance(grid, os.PathLike):
            grids = vdb.readAllGridMetadata(str(grid))

            if not name and len(grids) > 1:
                print(
                    "File contains more than one grid, ",
                    f"only processing first named {grids[0].name}",
                )

            name = name or grids[0].name
            grid = vdb.read(str(grid), name)

        # rotate the grid to make Z up
        grid.transform.rotate(math.pi / 2, vdb.Axis.X)

        return cls(grid=grid, name=name, xform=cg.Transformation() or xform, **kwargs)

    @classmethod
    def from_pointcloud(
        cls,
        pointcloud: cg.Pointcloud | os.PathLike,
        **kwargs,
    ):
        if isinstance(pointcloud, os.PathLike):
            pointcloud = cg.Pointcloud.from_json(pointcloud)

        cls = cls(**kwargs)

        for pt in pointcloud.points:
            ijk = tuple(map(round, pt))
            cls.set_value(ijk, 1)

        return cls

    @classmethod
    def from_ply(
        cls,
        ply_path: os.PathLike,
        **kwargs,
    ):
        pointcloud = cg.Pointcloud.from_ply(ply_path)
        return cls.from_pointcloud(pointcloud, **kwargs)

    @classmethod
    def from_pcd(
        cls,
        pcd_path: os.PathLike,
        **kwargs,
    ):
        pc = pypcd4.PointCloud.from_path(pcd_path)
        xyz = pc.metadata.viewpoint[:3]
        Tl = cg.Translation.from_vector(xyz)

        q = pc.metadata.viewpoint[3:]
        R = cg.Rotation.from_quaternion(q)

        pts = [cg.Point(*pt[:3]) for pt in pc.numpy()]
        pointcloud = cg.Pointcloud(pts)

        pointcloud.scale(1000)

        return cls.from_pointcloud(pointcloud, xform=Tl * R, **kwargs)
