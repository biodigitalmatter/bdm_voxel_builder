import enum

import numpy as np
import numpy.typing as npt
from compas.colors import Color
from compas.geometry import Box

from bdm_voxel_builder.grid import Grid
from bdm_voxel_builder.helpers import (
    create_random_array,
    crop_array,
    get_mask_zone_xxyyzz,
    remap,
)


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
        grid_size: int | tuple[int, int, int] | Box = None,
        name: str = None,
        color: Color = None,
        diffusion_ratio: float = 0.12,
        diffusion_random_factor: float = 0.0,
        decay_random_factor: float = 0.0,
        decay_linear_value: float = 0.0,
        decay_ratio: float = 0.0,
        emission_factor: float = 0.1,
        gradient_resolution: float = 0.0,
        flip_colors: bool = False,
        emmision_array: npt.NDArray = None,
        gravity_dir: GravityDir = GravityDir.DOWN,
        gravity_ratio: float = 0.0,
        voxel_crop_range=(0, 1),
    ):
        super().__init__(
            grid_size,
            name=name,
            color=color,
        )
        self.diffusion_ratio = diffusion_ratio
        self.diffusion_random_factor = diffusion_random_factor
        self.decay_random_factor = decay_random_factor
        self.decay_linear_value = decay_linear_value
        self.decay_ratio = decay_ratio
        self.emission_factor = emission_factor
        self.gradient_resolution = gradient_resolution
        self.flip_colors = flip_colors
        self.emmision_array = emmision_array
        self.gravity_dir = gravity_dir
        self.gravity_ratio = gravity_ratio
        self.voxel_crop_range = voxel_crop_range
        self.iteration_counter: int = 0

    @property
    def color_array(self):
        r, g, b = self.color.rgb
        colors = np.copy(self.array)
        min_ = np.min(self.array)
        max_ = np.max(self.array)
        colors = remap(colors, output_domain=[0, 1], input_domain=[min_, max_])
        if self.flip_colors:
            colors = 1 - colors

        newshape = self.grid_size + [1]

        reds = np.reshape(colors * (r), newshape=newshape)
        greens = np.reshape(colors * (g), newshape=newshape)
        blues = np.reshape(colors * (b), newshape=newshape)

        return np.concatenate((reds, greens, blues), axis=3)

    def conditional_fill(self, condition="<", value=0.5, override_self=False):
        """returns new voxel_array with 0,1 values based on condition"""
        if condition == "<":
            mask_inv = self.array < value
        elif condition == ">":
            mask_inv = self.array > value
        elif condition == "<=":
            mask_inv = self.array <= value
        elif condition == ">=":
            mask_inv = self.array >= value
        a = np.zeros_like(self.array)
        a[mask_inv] = 0
        if override_self:
            self.array = a
        return a

    def grade(self):
        if self.gradient_resolution == 0:
            pass
        else:
            self.array = (
                np.int64(self.array * self.gradient_resolution)
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
        if limit_by_Hirsh:
            self.diffusion_ratio = max(0, self.diffusion_ratio)
            self.diffusion_ratio = min(1 / 6, self.diffusion_ratio)

        shifts = [-1, 1]
        axes = [0, 0, 1, 1, 2, 2]
        # order: left, right, front
        # diffuse per six face_neighbors
        total_diffusions = np.zeros_like(self.array)
        for i in range(6):
            # y: shift neighbor
            y = np.copy(self.array)
            y = np.roll(y, shifts[i % 2], axis=axes[i])
            if not reintroduce_on_the_other_end:
                # TODO: make work with non square grid
                e = self.grid_size[0] - 1

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
                    1 - np.zeros_like(self.array) * self.diffusion_random_factor
                )
            # summ up the diffusions per faces
            total_diffusions += diff_ratio * (self.array - y)
        self.array -= total_diffusions
        return self.array

    def gravity_shift(self, reintroduce_on_the_other_end=False):
        """
        infinitive borders
        every value of the voxel cube diffuses with its face nb
        standard finite volume approach (Hirsch, 1988).
        diffusion change of voxel_x between voxel_x and y:
        delta_x = -a(x-y)
        where 0 <= a <= 1/6
        """
        shifts = [-1, 1]
        axes = [0, 0, 1, 1, 2, 2]
        # order: left, right, front
        # diffuse per six face_neighbors
        total_diffusions = np.zeros_like(self.array)
        if self.gravity_ratio != 0:
            # y: shift neighbor
            y = np.copy(self.array)
            y = np.roll(y, shifts[self.gravity_dir % 2], axis=axes[self.gravity_dir])
            if not reintroduce_on_the_other_end:
                # TODO replace to padded array method
                # TODO fix for non square grids
                e = self.grid_size[0] - 1
                # removing the values from the other end after rolling
                match self.gravity_dir:
                    case GravityDir.LEFT:
                        y[:][:][e] = 0

                    case GravityDir.RIGHT:
                        y[:][:][0] = 0

                    case GravityDir.FRONT:
                        m = y.transpose((1, 0, 2))
                        m[:][:][e] = 0
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
                        m[:][:][0] = 0
                        y = m.transpose((1, 2, 0))

            total_diffusions += self.gravity_ratio * (self.array - y) / 2
            self.array -= total_diffusions
        else:
            pass
        return self.array

    def emission_self(self, proportional=True):
        """updates array values based on self array values
        by an emission factor ( multiply / linear )"""

        if proportional:  # proportional
            self.array += self.array * self.emission_factor
        else:  # absolut
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
        else:  # absolut
            self.array = np.where(external_emission_array != 0, factor, self.array)

    def block_grids(self, other_grids: list[Grid]):
        """acts as a solid obstacle, stopping diffusion of other grid
        input list of grids"""
        for grid in other_grids:
            grid.array = np.where(self.array == 1, 0 * grid.array, 1 * grid.array)

    def decay(self):
        if self.decay_random_factor == 0:
            self.array -= self.array * self.decay_ratio
        else:
            randomized_decay = self.decay_ratio * (
                1 - create_random_array(self.grid_size) * self.decay_random_factor
            )
            randomized_decay = abs(randomized_decay) * -1
            self.array += self.array * randomized_decay

    def decay_linear(self):
        s, e = self.voxel_crop_range
        self.array = crop_array(self.array - self.decay_linear_value, s, e)

    def iterate(
        self, diffusion_limit_by_Hirsh=False, reintroduce_on_the_other_end=False
    ):
        self.iteration_counter += 1
        # emission update
        self.emmission_in()
        # decay
        self.decay()
        # diffuse
        self.diffuse(diffusion_limit_by_Hirsh, reintroduce_on_the_other_end)
        # emission_out
        self.emmission_out_update()

    def set_values_in_zone_xxyyzz(self, zone_xxyyzz, value=1, add_values=False):
        """add or replace values within zone (including both end)
        add_values == True: add values in self.array
        add_values == False: replace values in self.array *default
        input:
            zone_xxyyzz = [x_start, x_end, y_start, y_end, z_start, z_end]
        """
        if add_values:
            zone = get_mask_zone_xxyyzz(self.grid_size, zone_xxyyzz, return_bool=False)
            zone *= value
            self.array += zone
        else:
            mask = get_mask_zone_xxyyzz(self.grid_size, zone_xxyyzz, return_bool=True)
            self.array[mask] = value
