import math
import os
from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import Self

import compas.geometry as cg
import numpy as np
import numpy.typing as npt
import pyopenvdb as vdb
import pypcd4
from compas.colors import Color
from more_itertools import grouper

from bdm_voxel_builder import TEMP_DIR, get_direction_dictionary
from bdm_voxel_builder.helpers import (
    get_localized_index_map,
    get_savepath,
    xform_to_vdb,
)


class Grid:
    def __init__(
        self,
        name: str | None = None,
        clipping_box: cg.Box | None = None,
        xform: cg.Transformation | None = None,
        color: Color | None = None,
        grid: vdb.GridBase | None = None,
        flip_colors: bool = False,
    ):
        self.clipping_box = clipping_box
        self.color = color or Color.black()

        self.vdb = grid or vdb.FloatGrid()
        self.xform = xform or cg.Transformation()

        self.name = name or "grid"

        self.flip_colors = flip_colors

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

    @property
    def name(self) -> str:
        return self.vdb.name

    @name.setter
    def name(self, value: str):
        self.vdb.name = value

    @property
    def xform(self) -> cg.Transformation:
        return self._xform

    @xform.setter
    def xform(self, value: cg.Transformation):
        self._xform = value
        self.vdb.transform = xform_to_vdb(value)

    def copy(self, name: str | None = None):
        new_copy = deepcopy(self)
        new_copy.name = name or self.name
        return new_copy

    def get_value(self, ijk: tuple[int, int, int]):
        return self.vdb.getConstAccessor().getValue(ijk)

    def get_values(self, indices: list[tuple[int, int, int]]):
        accessor = self.vdb.getConstAccessor()
        return [accessor.getValue(ijk) for ijk in indices]

    def set_value(self, ijk: tuple[int, int, int], value: float):
        self.vdb.getAccessor().setValueOn(ijk, value)

    def set_values(
        self,
        indices: list[tuple[int, int, int]],
        values: float | npt.NDArray[np.float_],
    ):
        accessor = self.vdb.getAccessor()

        if not isinstance(values, Sequence):
            values = np.full(len(indices), values)

        for index, value in zip(indices, values, strict=True):
            accessor.setValueOn(index, value)

    def set_value_using_index_map(
        self,
        index_map: npt.NDArray[np.int_],
        origin: tuple[int, int, int] = (0, 0, 0),
        values: list[float] | float = 1.0,
    ):
        localized_map = get_localized_index_map(index_map, origin)
        self.set_values(localized_map, values)

    def set_values_using_array(self, array: np.ndarray, origin=(0, 0, 0)):
        self.vdb.copyFromArray(array, ijk=origin)

    def set_values_in_zone_xxyyzz(
        self, zone_xxyyzz: tuple[int, int, int, int, int, int], value=1.0
    ):
        """add or replace values within zone (including both end)
        add_values == True: add values in self.array
        add_values == False: replace values in self.array *default
        input:
            zone_xxyyzz = [x_start, x_end, y_start, y_end, z_start, z_end]
        """
        xbounds, ybounds, zbounds = grouper(zone_xxyyzz, 2)
        bmin = [xbounds[0], ybounds[0], zbounds[0]]
        bmax = [xbounds[1], ybounds[1], zbounds[1]]
        self.vdb.fill(bmin, bmax, value, active=True)

    def get_active_voxels(self) -> npt.NDArray:
        """returns indices of nonzero values
        list of coordinates
            shape = [3,n]"""
        return np.array([item.min for item in self.vdb.citerOnValues()], dtype=np.int_)

    def get_active_voxels_values(self) -> npt.NDArray[np.float_]:
        return np.array(
            [item.value for item in self.vdb.citerOnValues()], dtype=np.float_
        )

    def iter_active_voxels(self) -> Iterable[tuple[int, int, int]]:
        for item in self.vdb.citerOnValues():
            yield item.min

    def enumerate_active_voxels(self) -> Iterable[tuple[int, int, int], float]:
        for item in self.vdb.citerOnValues():
            yield item.min, item.value

    def get_number_of_active_voxels(self) -> int:
        return self.vdb.activeVoxelCount()

    def get_neighbors(
        self, ijk: tuple[int, int, int], radius: int = 1
    ) -> Iterable[tuple[int, int, int]]:
        if radius != 1:
            raise NotImplementedError("Only radius 1 is supported")

        for direction in get_direction_dictionary().values():
            neighbor = tuple(map(sum, zip(ijk, direction, strict=True)))
            yield neighbor

    def get_active_neighbors(
        self, ijk: tuple[int, int, int], radius: int = 1
    ) -> Iterable[tuple[int, int, int]]:
        if radius != 1:
            raise NotImplementedError("Only radius 1 is supported")

        for neighbor in self.get_neighbors(ijk, radius=radius):
            if self.vdb.getConstAccessor().isValueOn(neighbor):
                yield neighbor

    def set_values_for_neighbors(self, ijk: tuple[int, int, int], value: float):
        for neighbor in self.get_neighbors(ijk):
            self.set_value(neighbor, value)

    def map_values(self, func):
        self.vdb.mapValues(func)

    def map_on_active_voxels(self, func):
        self.vdb.mapOnValues(func)

    def map_on_inactive_voxels(self, func):
        self.vdb.mapOffValues(func)

    def get_pointcloud(self):
        return cg.Pointcloud(self.get_active_voxels())

    def get_world_pts(self) -> list[list[float]]:
        return cg.transform_points_numpy(self.get_active_voxels(), self.xform)

    def get_world_pointcloud(self) -> cg.Pointcloud:
        return cg.Pointcloud(self.get_world_pts())

    def merge_with(self, other: Self):
        grids = [other] if not isinstance(other, Sequence) else other

        for grid in grids:
            vdb = grid.vdb.deepCopy()
            self.vdb.combine(vdb, lambda a, b: max(a, b))

    def block_grids(self, other_grids: list[Self]):
        """acts as a solid obstacle, stopping diffusion of other grid
        input list of grids"""
        for grid in other_grids:
            for ijk, value in self.enumerate_active_voxels():
                if value == 1:
                    grid.set_value(ijk, 0)

    def merged_with(self, other: Self):
        new = deepcopy(self)
        new.merge_with(other)
        return new

    def to_numpy(self):
        arr = np.zeros([int(s) for s in self.clipping_box.dimensions])

        self.vdb.copyToArray(arr)

        return arr

    def save_vdb(self, dir: os.PathLike = TEMP_DIR):
        path = get_savepath(dir, ".vdb", note=f"grid_{self.name}")

        grid = self.vdb.deepCopy()

        # rotate the grid to make Y up for vdb_view and houdini
        # grid.transform.rotate(-math.pi / 2, vdb.Axis.X)
        vdb.write(str(path), grids=[grid])

    @classmethod
    def from_numpy(
        cls,
        array_or_path: npt.NDArray | os.PathLike,
        clipping_box: cg.Box | None = None,
        name: str | None = None,
        **kwargs,
    ):
        if isinstance(array_or_path, os.PathLike):
            loaded_array: npt.NDArray = np.load(array_or_path)
            array = loaded_array
        else:
            array = array_or_path

        indices = np.array(np.nonzero(array)).transpose()

        grid = vdb.FloatGrid()
        accessor = grid.getAccessor()

        for ijk in indices:
            accessor.setValueOn(ijk, 1)

        return cls(
            clipping_box=clipping_box or cg.Box.from_diagonal(([0, 0, 0], array.shape)),
            grid=grid,
            name=name,
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
