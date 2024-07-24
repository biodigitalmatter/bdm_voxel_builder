from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from bdm_voxel_builder.helpers.numpy import (
    create_random_array,
    create_zero_array,
    crop_array,
    get_mask_zone_xxyyzz,
)
from bdm_voxel_builder.helpers.math import remap


@dataclass
class DataLayer:
    name: str = None
    axis_order: str = "zyx"
    voxel_size: int = 20
    rgb: float = (1, 1, 1)
    diffusion_ratio: float = 0.12
    diffusion_random_factor: float = 0.0
    decay_random_factor: float = 0.0
    decay_linear_value: float = 0.0
    decay_ratio: float = 0.0
    emission_factor: float = 0.1
    gradient_resolution: float = 0.0
    flip_colors: bool = False

    def __post_init__(self):
        self.array: npt.NDArray = create_zero_array(self.voxel_size)
        self.voxel_crop_range = [0, 1]
        self._iteration_counter: int = 0
        self._emmision_array = None
        self._gravity_ratio = 0

    @property
    def color_array(self):
        r, g, b = self.rgb
        colors = np.copy(self.array)
        min_ = np.min(colors)
        max_ = np.max(colors)
        colors = remap(colors, output_domain=[0, 1], input_domain=[min_, max_])
        if self._flip_colors:
            colors = 1 - colors

        reds = np.reshape(colors * (r), [self._n, self._n, self._n, 1])
        greens = np.reshape(colors * (g), [self._n, self._n, self._n, 1])
        blues = np.reshape(colors * (b), [self._n, self._n, self._n, 1])
        colors = np.concatenate((reds, greens, blues), axis=3)
        return colors

    @property
    def gravity_dir(self):
        return self._gravity_dir

    @gravity_dir.setter
    def gravity_dir(self, v):
        """direction: 0:left, 1:right, 2:front, 3:back, 4:down, 5:up"""
        if not isinstance(v, (int)) or v > 5 or v < 0:
            raise ValueError("gravity ratio must be an integrer between 0 and 5")
        self._gravity_dir = v

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
        a = create_zero_array(self._n)
        a[mask_inv] = 0
        if override_self:
            self.array = a
        return a

    def set_layer_value_at_index(self, index=[0, 0, 0], value=1):
        index2 = np.mod(index, self.voxel_size)
        i, j, k = index2
        self.array[i][j][k] = value
        return self.array

    def get_value_at_index(self, index=[0, 0, 0]):
        i, j, k = index
        v = self.array[i][j][k]
        return v

    def get_nonzero_point_list(self, array):
        """returns indicies of nonzero values
        if list_of_points:
            shape = [n,3]
        else:
            shape = [3,n]"""
        non_zero_array = np.nonzero(array)
        return np.transpose(non_zero_array)

    def get_nonzero_index_coordinates(self, array):
        """returns indicies of nonzero values
        list of coordinates
            shape = [3,n]"""
        non_zero_array = np.nonzero(array)
        return non_zero_array

    def grade(self):
        if self.gradient_resolution == 0:
            pass
        else:
            self._array = (
                np.int64(self.array * self._gradient_resolution)
                / self._gradient_resolution
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
            self._diffusion_ratio = max(0, self.diffusion_ratio)
            self._diffusion_ratio = min(1 / 6, self.diffusion_ratio)

        shifts = [-1, 1]
        axes = [0, 0, 1, 1, 2, 2]
        # order: left, right, front
        # diffuse per six face_neighbors
        total_diffusions = create_zero_array(self.voxel_size)
        for i in range(6):
            # y: shift neighbor
            y = np.copy(self.array)
            y = np.roll(y, shifts[i % 2], axis=axes[i])
            if not reintroduce_on_the_other_end:
                e = self.voxel_size - 1
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
                    1 - create_random_array(self._n) * self.diffusion_random_factor
                )
            # summ up the diffusions per faces
            total_diffusions += diff_ratio * (self.array - y) / 2
        self.array -= total_diffusions
        return self.array

    def gravity_shift(self, reintroduce_on_the_other_end=False):
        """direction: 0:left, 1:right, 2:front, 3:back, 4:down, 5:up
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
        total_diffusions = create_zero_array(self._n)
        if self.gravity_ratio != 0:
            for i in [self.gravity_dir]:
                # y: shift neighbor
                y = np.copy(self._array)
                y = np.roll(y, shifts[i % 2], axis=axes[i])
                if not reintroduce_on_the_other_end:
                    # TODO replace to padded array method
                    e = self._n - 1
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
                total_diffusions += self.gravity_ratio * (self._array - y) / 2
            self._array -= total_diffusions
        else:
            pass
        return self._array

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
            self.array = np.where(
                external_emission_array != 0, self.array + factor, self.array
            )

    def block_layers(self, other_layers=[]):
        """acts as a solid obstacle, stopping diffusion of other layers
        input list of layers"""
        for i in range(len(other_layers)):
            layer = other_layers[i]
            layer.array = np.where(self.array == 1, 0 * layer.array, 1 * layer.array)
        pass

    def decay(self):
        if self.decay_random_factor == 0:
            self.array -= self.array * self.decay_ratio
        else:
            randomized_decay = self.decay_ratio * (
                1 - create_random_array(self._n) * self._decay_random_factor
            )
            randomized_decay = abs(randomized_decay) * -1
            self.array += self.array * randomized_decay

    def decay_linear(self):
        s, e = self.voxel_crop_range
        self._array = crop_array(self._array - self.decay_linear_value, s, e)

    def iterate(
        self, diffusion_limit_by_Hirsh=False, reintroduce_on_the_other_end=False
    ):
        self._iteration_counter += 1
        # emission update
        self.emmission_in()
        # decay
        self.decay()
        # diffuse
        self.diffuse(diffusion_limit_by_Hirsh, reintroduce_on_the_other_end)
        # emission_out
        self.emmission_out_update()

    def get_merged_array_with(self, other_layer):
        a1 = self.array
        a2 = other_layer.array
        return a1 + a2

    def add_values_in_zone_xxyyzz(self, zone_xxyyzz, value = 1, add_values = False):
        """add or replace values within zone (including both end)
        add_values == True: add values in self.array
        add_values == False: replace values in self.array *default
        input: 
            zone_xxyyzz = [x_start, x_end, y_start, y_end, z_start, z_end]
            """
        # np.zeros_like(self.array)
        if add_values:
            zone = get_mask_zone_xxyyzz(self.voxel_size, zone_xxyyzz, return_bool = False)
            zone *= value
            self.array += zone
        else:
            mask = get_mask_zone_xxyyzz(self.voxel_size, zone_xxyyzz, return_bool=True)
            self.array[mask] = value


