import enum
from typing import Self

import compas.geometry as cg
import numpy as np
import numpy.typing as npt
import pyopenvdb as vdb
from compas.colors import Color
from compas.geometry import Box

from bdm_voxel_builder.grid import Grid
from bdm_voxel_builder.helpers import crop_array


class GravityDir(enum.Enum):
    LEFT = 0
    RIGHT = 1
    FRONT = 2
    BACK = 3
    DOWN = 4
    UP = 5


class DiffusiveGrid(Grid):
    def __init__(
        self,
        name: str,
        clipping_box: Box,
        xform: cg.Transformation | None = None,
        color: Color | None = None,
        grid: vdb.GridBase = None,
        diffusion_ratio: float = 0.12,
        diffusion_random_factor: float = 0.0,
        decay_random_factor: float = 0.0,
        decay_linear_value: float = 0.0,
        decay_ratio: float = 0.0,
        emission_factor: float = 0.1,
        gradient_resolution: float = 0.0,
        flip_colors: bool = False,
        emission_array: npt.NDArray | None = None,
        gravity_dir: GravityDir = GravityDir.DOWN,
        gravity_ratio: float = 0.0,
        voxel_crop_range=(0, 1),
    ):
        super().__init__(
            name=name,
            clipping_box=clipping_box,
            xform=xform,
            color=color,
            grid=grid,
            flip_colors=flip_colors,
        )
        self.diffusion_ratio = diffusion_ratio
        self.diffusion_random_factor = diffusion_random_factor
        self.decay_random_factor = decay_random_factor
        self.decay_linear_value = decay_linear_value
        self.decay_ratio = decay_ratio
        self.emission_factor = emission_factor
        self.gradient_resolution = gradient_resolution
        self.emission_array = emission_array
        self.gravity_dir = gravity_dir
        self.gravity_ratio = gravity_ratio
        self.voxel_crop_range = voxel_crop_range
        self.iteration_counter: int = 0

    def grade(self):
        if self.gradient_resolution == 0:
            pass
        else:
            self.vdb.mapOn(
                lambda value: round(value * self.gradient_resolution)
                / self.gradient_resolution
            )

    def diffuse(self, limit_by_Hirsh=True, reintroduce_on_the_other_end=False):
        """infinitive borders
        every value of the voxel cube diffuses with its face nb
        standard finite volume approach (Hirsch, 1988).
        in not limit_by_Hirsch: ph volume can grow
        diffusion change of voxel_x between voxel_x and y:
        delta_x = -a(x-y)
        where 0 <= a <= 1/6
        """
        array = self.to_numpy()

        if limit_by_Hirsh:
            self.diffusion_ratio = max(0, self.diffusion_ratio)
            self.diffusion_ratio = min(1 / 6, self.diffusion_ratio)

        shifts = [-1, 1]
        axes = [0, 0, 1, 1, 2, 2]
        # order: left, right, front
        # diffuse per six face_neighbors
        total_diffusions = np.zeros_like(array)

        # if isinstance(self.grid_size, int):
        #     self.grid_size = [self.grid_size, self.grid_size, self.grid_size]
        # else:
        #     self.grid_size = self.grid_size

        for i in range(6):
            # y: shift neighbor
            y = np.copy(array)
            y = np.roll(y, shifts[i % 2], axis=axes[i])
            if not reintroduce_on_the_other_end:  # TODO do it with np.pad
                e = self.grid_size[axes[i]] - 1
                # removing the values from the other end after rolling
                if i == 0:
                    y[:][:][e] = 0
                elif i == 1:
                    y[:][:][0] = 0
                elif 2 <= i <= 3:
                    m = y.transpose((1, 0, 2))
                    if i == 2:
                        m[:][:][e] = 0
                    elif i == 3:
                        m[:][:][0] = 0
                    y = m.transpose((1, 0, 2))
                elif 4 <= i <= 5:
                    m = y.transpose((2, 0, 1))
                    if i == 4:
                        m[:][:][e] = 0
                    elif i == 5:
                        m[:][:][0] = 0
                    y = m.transpose((1, 2, 0))
            # calculate diffusion value
            if self.diffusion_random_factor == 0:
                diff_ratio = self.diffusion_ratio
            else:
                diff_ratio = self.diffusion_ratio * (
                    1 - np.zeros_like(array) * self.diffusion_random_factor
                )
            # sum up the diffusions per faces
            total_diffusions += diff_ratio * (self.to_numpy() - y)

        self.set_values_with_array(array - total_diffusions)

    def gravity_shift(self, reintroduce_on_the_other_end=False):
        """
        infinitive borders
        every value of the voxel cube diffuses with its face nb
        standard finite volume approach (Hirsch, 1988).
        diffusion change of voxel_x between voxel_x and y:
        delta_x = -a(x-y)
        where 0 <= a <= 1/6
        """
        array = self.to_numpy()
        shifts = [-1, 1]
        axes = [0, 0, 1, 1, 2, 2]
        # order: left, right, front
        # diffuse per six face_neighbors
        total_diffusions = np.zeros_like(array)
        if self.gravity_ratio != 0:
            # y: shift neighbor
            y = np.copy(array)
            y = np.roll(y, shifts[self.gravity_dir % 2], axis=axes[self.gravity_dir])
            if not reintroduce_on_the_other_end:
                # TODO replace to padded array method
                # TODO fix for non square grids
                e, f, g = self.grid_size
                # removing the values from the other end after rolling
                match self.gravity_dir:
                    case GravityDir.LEFT:
                        y[:][:][e - 1] = 0

                    case GravityDir.RIGHT:
                        y[:][:][0] = 0

                    case GravityDir.FRONT:
                        m = y.transpose((1, 0, 2))
                        m[:][:][f - 1] = 0
                        y = m.transpose((1, 0, 2))

                    case GravityDir.BACK:
                        m = y.transpose((1, 0, 2))
                        m[:][:][0] = 0
                        y = m.transpose((1, 0, 2))

                    case GravityDir.DOWN:
                        m = y.transpose((2, 0, 1))
                        m[:][:][e] = 0
                        y = m.transpose((1, 2, 0))

                    case GravityDir.UP:
                        m = y.transpose((2, 0, 1))
                        m[:][:][g - 1] = 0
                        y = m.transpose((1, 2, 0))

            total_diffusions += self.gravity_ratio * (array - y) / 2

            self.set_values_with_array(array - total_diffusions)

    def emission_self(self, proportional=True):
        """updates array values based on self array values
        by an emission factor ( multiply / linear )"""

        if proportional:  # proportional
            self.array += self.array * self.emission_factor
        else:  # absolute
            self.array = np.where(
                self.array != 0, self.array + self.emission_factor, self.array
            )

    def emission_intake(self, external_emission_array, factor, proportional=True):
        """updates array values based on a given array
        and an emission factor ( multiply / linear )"""

        if proportional:  # proportional
            # self.array += external_emission_array * self.emission_factor
            self.array = np.where(
                external_emission_array != 0,
                self.array + external_emission_array * factor,
                self.array,
            )
        else:  # absolute
            self.array = np.where(external_emission_array != 0, factor, self.array)

    def block_grids(self, other_grids: list[Grid]):
        """acts as a solid obstacle, stopping diffusion of other grid
        input list of grids"""
        for grid in other_grids:
            grid.array = np.where(self.array == 1, 0 * grid.array, 1 * grid.array)

    def decay(self):
        def decay_voxel(value):
            return value - value * self.decay_ratio

        self.vdb.mapOn(decay_voxel)

    def decay_linear(self):
        s, e = self.voxel_crop_range
        self.array = crop_array(self.array - self.decay_linear_value, s, e)

    def iterate(
        self, diffusion_limit_by_Hirsh=False, reintroduce_on_the_other_end=False
    ):
        self.iteration_counter += 1
        # emission update
        self.emission_in()
        # decay
        self.decay()
        # diffuse
        self.diffuse(diffusion_limit_by_Hirsh, reintroduce_on_the_other_end)
        # emission_out
        self.emission_out_update()

    def diffuse_diffusive_grid(
        self,
        emission_array: npt.NDArray | list[npt.NDArray] | None = None,
        blocking_grids: Self | list[Self] | None = None,
        gravity_shift_bool: bool = False,
        diffuse_bool: bool = True,
        decay: bool = True,
        decay_linear: bool = False,
        grade=True,
        n_iterations: int = 1,
    ):
        """
        DIFFUSE AND DECAY GRIDS
        optionally multiple iterations
        diffusion steps:

        loads from emissive_grids
        diffuse
        apply gravity shift
        decay_linear
        decay_proportional
        get_blocked_by (one or more grids)
        apply gradient resolution

        gravity direction: 0:left, 1:right, 2:front, 3:back, 4:down, 5:up"""
        for _ in range(n_iterations):
            if isinstance(emission_array, np.ndarray):
                self.emission_intake(emission_array, 1, False)
            elif isinstance(emission_array, list):
                for i in range(emission_array):
                    self.emission_intake(emission_array[i], 1, False)

            # diffuse
            if diffuse_bool:
                self.diffuse()

            # gravity
            if gravity_shift_bool:
                self.gravity_shift()

            # decay
            if decay_linear:
                self.decay_linear()
            elif decay:
                self.decay()

            # collision

            if blocking_grids:
                if isinstance(blocking_grids, list):
                    for blocking_grid in blocking_grids:
                        blocking_grid.block_grids([self])
                else:
                    blocking_grids.block_grids([self])

            # apply gradient steps
            if self.gradient_resolution != 0 and grade:
                self.grade()
