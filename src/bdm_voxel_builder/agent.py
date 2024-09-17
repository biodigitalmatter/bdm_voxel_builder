import random as r
from copy import copy
from dataclasses import dataclass
from math import ceil, cos, radians, sin, trunc

import compas.geometry as cg
import numpy as np
import numpy.typing as npt

from bdm_voxel_builder.agent_algorithms.common import (
    get_any_free_voxel_above_array,
    get_any_voxel_in_region,
    get_lowest_free_voxel_above_array,
    get_random_index_in_zone_xxyy_on_Z_level,
)
from bdm_voxel_builder.grid import Grid
from bdm_voxel_builder.helpers import (
    clip_indices_to_grid_size,
    get_array_density_from_zone_xxyyzz,
    get_array_density_using_map,
    get_localized_map,
    get_sub_array,
    index_map_cylinder,
    index_map_sphere,
    mask_index_map_by_nonzero,
    random_choice_index_from_best_n,
    transform_index_map_to_plane,
)
from bdm_voxel_builder.helpers.array import get_values_using_index_map


@dataclass
class Agent:
    """Object based voxel builder, a guided random_walker"""

    initial_pose: tuple[int, int, int] | None = None
    leave_trace: bool = None
    space_grid = None
    ground_grid = None

    def __init__(self):
        self.initial_pose: npt.NDArray[np.int_] | None = None
        self.leave_trace: bool = True

        self.track_grid = None
        self.move_history = []
        self.save_move_history = True
        self.track_flag = None
        self.step_counter = 0
        self.passive_counter = 0

        self.id = 0

        self.build_h = 2
        self.build_random_chance = 0.1
        self.build_random_gain = 0.6
        self.max_build_angle = 90

        self.walk_radius = 4
        self.min_walk_radius = 0
        self.build_radius = 3
        self.sense_radius = 5
        self.move_map = None
        self.build_map = None
        self.sense_map = None
        self.sense_inplane_map = None
        self.sense_depth_map = None
        self.sense_overhang_map = None
        self.sense_nozzle_map = None
        self.sense_topology_bool = True

        self.min_build_density = 0
        self.max_build_density = 1
        self.build_limit_mod_by_density = [0.5, -0.5, 0.5]
        self.build_by_density = False
        self.build_by_density_random_factor = 0
        self.max_shell_thickness = 15
        self.overhang_density = 0.5

        self.move_mod_z = 0
        self.move_mod_random = 0.1
        self.move_mod_follow = 1

        self.reset_after_build = False
        self.reset_after_erase = False

        self.inactive_step_count_limit = None

        self.last_move_vector = []
        self.move_turn_degree = None

    _normal_vector = cg.Vector(0, 0, 1)

    def __post_init__(self):
        self._pose = self.initial_pose or np.array([0, 0, 0], dtype=np.int_)

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, new_pose):
        old_pose = self.pose

        self.move_history.append(old_pose)

        if self.leave_trace:
            self.track_grid.set_value(old_pose, 1)

        self.space_grid.set_value(old_pose, 0)
        self.space_grid.set_value(new_pose, 1)
        self._pose = new_pose

    @property
    def build_chance(self):
        return self._build_chance

    @build_chance.setter
    def build_chance(self, value):
        if not isinstance(value, (float | int)):
            raise ValueError("Chance must be a number")
        self._build_chance = value

    @property
    def erase_chance(self):
        return self._erase_chance

    @erase_chance.setter
    def erase_chance(self, value):
        if not isinstance(value, float | int):
            raise ValueError("Chance must be a number")
        self._erase_chance = value

    @property
    def die_chance(self):
        return self._die_chance

    @die_chance.setter
    def die_chance(self, value):
        if not isinstance(value, (float | int)):
            raise ValueError("Chance must be a number")
        self._die_chance = value

    @property
    def normal_vector(self):
        """unit vector pointing towards surrounding mass centroid
        surrounding volume(ground_grid) within sense_map"""
        return self.calculate_normal_vector()

    @property
    def normal_angle(self):
        """angle of self.normal_vector and -World.Z"""
        v = self.get_normal_angle()
        return v

    @property
    def fab_plane(self):
        self._fab_plane = cg.Plane(self.pose, self.normal_vector)
        return self._fab_plane

    def copy(self):
        return copy(self)

    def analyze_relative_position(self, grid: Grid):
        """check if there is sg around the agent
        return list of bool:
            [below, aside, above]"""
        values = self.get_grid_nb_values_6(grid, self.pose)
        values = values.tolist()
        above = values.pop(0)
        below = values.pop(1)
        sides = sum(values)

        aside = sides > 0
        above = above > 0
        below = below > 0

        self.relative_booleans_bottom_up = [below, aside, above]
        return below, aside, above

    def direction_preference_6_pheromones(self, x=0.5, up=True):
        """up = 1
        side = x
        down = 0.1"""
        return np.asarray([1, x, 0.1, x, x, x]) if up else np.ones(6)

    def direction_preference_26_pheromones(self, up=1, side=0.5, down=0):
        """up = 1
        side = x
        down = 0.1"""

        u = [up] * 9
        m = [side] * 8
        b = [down] * 9

        return np.asarray(u + m + b)

    # INTERACTION WITH GRIDS
    def get_array_value_at_index(
        self,
        array: np.ndarray,
        index=(0, 0, 0),
        reintroduce=False,
        limit_in_bounds=True,
        round_=False,
        ceil_=False,
        eliminate_dec=False,
    ):
        if reintroduce:
            i, j, k = np.mod(index, array.shape)
        elif limit_in_bounds:
            a, b, c = array.shape
            i, j, k = np.clip(index, [0, 0, 0], [a - 1, b - 1, c - 1])
        else:
            i, j, k = index
        try:
            v = array[i][j][k]
        except Exception as e:
            print(e)
            v = 0
        if round_:
            v = round(v)
        elif ceil_:
            v = ceil(v)
        elif eliminate_dec:
            v = trunc(v)

        return v

    # TODO: move to grid class?
    def get_grid_value_at_index(
        self,
        grid: Grid,
        index=(0, 0, 0),
        reintroduce=False,
        limit_in_bounds=True,
        round_=False,
        ceil_=False,
        eliminate_dec=False,
    ):
        return self.get_array_value_at_index(
            grid.to_numpy(),
            index=index,
            reintroduce=reintroduce,
            limit_in_bounds=limit_in_bounds,
            round_=round_,
            ceil_=ceil_,
            eliminate_dec=eliminate_dec,
        )

    # TODO: move to grid class?
    def set_array_value_at_index(
        self, value, array, index=(0, 0, 0), reintroduce=False, limit_in_bounds=True
    ):
        if reintroduce:
            i, j, k = np.mod(index, array.shape)
        elif limit_in_bounds:
            a, b, c = array.shape
            i, j, k = np.clip(index, [0, 0, 0], [a - 1, b - 1, c - 1])
        else:
            i, j, k = index
        try:
            array[i][j][k] = value
        except Exception as e:
            print(e)

    def get_nb_indices_6(self, pose):
        """returns the list of nb cell indexes"""
        nb_cell_index_list = []
        for key in self.compass_array:
            d = self.compass_array[key]
            nb_cell_index_list.append(d + pose)
        return nb_cell_index_list

    def get_nb_indices_26(self, pose: tuple[int, int, int]):
        """returns the list of nb cell indexes"""
        nb_cell_index_list = []
        for d in self.cube_array:
            nb_cell_index_list.append((d + pose).tolist())
        return nb_cell_index_list

    def get_grid_nb_values_6(self, grid: Grid, pose=None, round_values=False):
        """return list of values"""
        value_list = []
        for key in self.compass_array:
            d = self.compass_array[key]
            nb_cell_index = d + pose
            # dont check index in boundary
            v = self.get_grid_value_at_index(grid, nb_cell_index)
            value_list.append(v)
        v = np.asarray(value_list)
        if round_values:
            v.round()
        return v

    def get_nb_values_6_of_array(self, array, grid_size, pose, round_values=False):
        """return list of values"""
        value_list = []
        for key in self.compass_array:
            d = self.compass_array[key]
            i, j, k = clip_indices_to_grid_size(d + pose, grid_size)
            # dont check index in boundary
            v = array[i][j][k]
            value_list.append(v)
        v = np.asarray(value_list)
        if round_values:
            v.round()
        return v

    def get_nb_values_26_of_array(self, array, voxel_size, pose, round_values=False):
        """return list of values"""
        nb_cells = self.get_nb_indices_26(pose)
        cells_to_check = list(nb_cells)
        value_list = []
        for nb_pose in cells_to_check:
            x, y, z = np.clip(np.asarray(nb_pose), 0, voxel_size - 1)
            nb_value = array[x][y][z]
            value_list.append(nb_value)
        v = np.asarray(value_list)
        if round_values:
            v.round()
        return v

    def get_nb_values_3x3_around_of_array(self, array, round_values=False):
        """return list of values"""
        nb_cells = self.get_nb_indices_26(self.pose)
        cells_to_check = list(nb_cells)[9:17]
        value_list = []
        for nb_pose in cells_to_check:
            x, y, z = nb_pose
            xc, yc, zc = np.clip(
                np.asarray(nb_pose), [0, 0, 0], np.asarray(array.shape) - 1
            )
            if x == xc and y == yc and z == zc:
                nb_value = array[xc][yc][zc]
                value_list.append(nb_value)
        v = np.asarray(value_list)
        if round_values:
            v.round()
        return v

    def get_nb_values_3x3_below_of_array(self, array, round_values=False):
        """return list of values"""
        nb_cells = self.get_nb_indices_26(self.pose)
        cells_to_check = list(nb_cells)[17:]
        value_list = []
        for nb_pose in cells_to_check:
            x, y, z = nb_pose
            xc, yc, zc = np.clip(
                np.asarray(nb_pose), [0, 0, 0], np.asarray(array.shape) - 1
            )
            if x == xc and y == yc and z == zc:
                nb_value = array[xc][yc][zc]
                value_list.append(nb_value)
        v = np.asarray(value_list)
        if round_values:
            v.round()
        return v

    def get_grid_nb_values_26(self, grid: Grid, pose=None, round_values=False):
        value_list = []
        for d in self.cube_array:
            nb_cell_index = d + pose
            v = self.get_grid_value_at_index(grid, nb_cell_index)
            value_list.append(v)
        v = np.asarray(value_list)
        if round_values:
            v.round()
        return v

    def get_grid_density(self, grid: Grid, print_=False, nonzero=False):
        """return clay density"""
        values = self.get_grid_nb_values_26(grid, self.pose, False)
        density = sum(values) / 26 if not nonzero else np.count_nonzero(values) / 26
        if print_:
            print(f"grid values:\n{values}\n")
            print(f"grid_density:{density} in pose:{self.pose}")
        return density

    def get_array_density(self, array: np.ndarray, print_=False, nonzero=False):
        """return clay density"""
        nb_indices = self.get_nb_indices_26(self.pose)
        values = []
        for pose in nb_indices:
            a, b, c = array.shape
            pose_2 = np.clip(pose, [0, 0, 0], [a - 1, b - 1, c - 1])
            if np.sum(pose - pose_2) == 0:
                values.append(array[*pose])
        if nonzero:
            density = np.count_nonzero(values) / len(values)
        else:
            density = sum(values) / len(values)

        if print_:
            print(f"grid values:\n{values}\n")
            print(f"grid_density:{density} in pose:{self.pose}")
        return density


    def get_array_slice_parametric(
        self,
        array,
        x_radius=1,
        y_radius=1,
        z_radius=0,
        x_offset=0,
        y_offset=0,
        z_offset=0,
        pose=None,
        format_values=0,
        pad_values=0,
    ):
        """takes sub array around pose, in x/y/z radius optionally offsetted
        format values: returns sum '0', average '1', or entire_array_slice: '2'"""
        if not isinstance(pose, np.ndarray | list):
            pose = self.pose

        x, y, z = pose
        x, y, z = [x + x_offset, y + y_offset, z + z_offset]

        pad_x = x_radius + abs(x_offset)
        pad_y = y_radius + abs(y_offset)
        pad_z = z_radius + abs(z_offset)

        c = pad_values
        np.pad(
            array,
            ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)),
            "constant",
            constant_values=((c, c), (c, c), (c, c)),
        )

        a = int(x - x_radius) + pad_x
        b = int(x + x_radius + 1) + pad_x
        c = int(y - y_radius) + pad_y
        d = int(y + y_radius + 1) + pad_y
        e = int(z - z_radius) + pad_z
        f = int(z + z_radius + 1) + pad_z

        v = array[a:b, c:d, e:f]

        if format_values == 0:
            return np.sum(v)

        if format_values == 1:
            return np.average(v)

        if format_values == 2:
            return v

        return v

    def get_array_density_in_slice_shape(
        self, array, slice_shape=(1, 1, 0, 0, 0, 1), nonzero=False, print_=False
    ):
        """returns grid density
        slice shape = [
        x_radius = 1,
        y_radius = 1,
        z_radius = 0,
        x_offset = 0,
        y_offset = 0,
        z_offset = 0
        ]
        *radius: amount of indices in both direction added. r = 1 at i = 0
        returns array[-1:2]
        """
        # get the sum of the values in the slice
        # print(array.shape)
        values = self.get_array_slice_parametric(
            array,
            *slice_shape,
            self.pose,
            format_values=2,
        )
        sum_values = np.sum(values)
        radiis = slice_shape[:3]
        slice_volume = 1
        for x in radiis:
            slice_volume *= (x + 0.5) * 2
        if not nonzero:
            density = sum_values / slice_volume
        else:
            density = np.count_nonzero(values) / slice_volume
        if density > 0 and print_:
            print(f"shape of input array{np.shape(array)}")
            print(
                f"slice_volume: {slice_volume}, density:{density}, n of nonzero = {np.count_nonzero(values)}, sum_values: {sum_values},values:{values}"  # noqa: E501
            )
        return density

    def get_grid_density_in_slice_shape(
        self, grid: Grid, slice_shape=(1, 1, 0, 0, 0, 1), nonzero=False, print_=False
    ):
        """returns grid density
        slice shape = [
        x_radius = 1,
        y_radius = 1,
        z_radius = 0,
        x_offset = 0,
        y_offset = 0,
        z_offset = 0
        ]
        *radius: amount of indices in both direction added. r = 1 at i = 0
        returns array[-1:2]
        """
        return self.get_array_density_in_slice_shape(
            grid.to_numpy, slice_shape=slice_shape, nonzero=nonzero, print_=print_
        )

    def get_array_density_in_box(self, array, box, nonzero=False):
        d = get_array_density_from_zone_xxyyzz(array, self.pose, box, nonzero=True)
        return d

    def get_move_mask_6(self, grid: Grid):
        """return ground directions as bools
        checks nbs of the nb cells
        if value > 0: return True"""
        # get nb cell indices
        nb_cells = self.get_nb_indices_6(self.pose)
        cells_to_check = list(nb_cells)

        check_failed = []
        # iterate through nb cells
        for nb_pose in cells_to_check:
            # print('nb_pose;', nb_pose)
            # check nbs of nb cell
            nbs_values = self.get_nb_cell_values(grid, nb_pose)
            # check nb cell
            nb_value = self.get_grid_value_at_index(grid, nb_pose)
            if np.sum(nbs_values) > 0 and nb_value == 0:
                check_failed.append(False)
            else:
                check_failed.append(True)
        exclude_pheromones = np.asarray(check_failed)
        return exclude_pheromones

    def get_move_mask_26(self, grid: Grid, fly=False, check_self_collision=False):
        """return move mask 1D array for the 26 nb voxel around self.pose
        the voxel
            most be non-solid
            must has at least one solid face_nb
        return:
            True if can not move there
            False if can move there

        if fly == True, cells do not have to be neighbors of solid
        """
        # get nb cell indices
        # nb_cells = self.get_nb_indices_6(self.pose)
        nb_cells = self.get_nb_indices_26(self.pose)
        cells_to_check = list(nb_cells)
        exclude = []  # FALSE if agent can move there, True if cannot
        # iterate through nb cells
        for nb_pose in cells_to_check:
            # check if nb cell is empty
            nb_value = self.get_grid_value_at_index(grid, nb_pose)
            if check_self_collision:
                nb_value_collision = self.get_grid_value_at_index(
                    self.space_grid, nb_pose
                )
                nb_value += nb_value_collision
            # print(nb_value)
            if nb_value == 0:
                if not fly:
                    # check if nb cells have any face_nb cell which is solid
                    nbs_values = self.get_grid_nb_values_6(grid, nb_pose)
                    # print(nbs_values)
                    if np.sum(nbs_values) > 0:
                        exclude.append(False)
                    else:
                        exclude.append(True)
                else:
                    exclude.append(False)
            else:
                exclude.append(True)
        # print(exclude)
        exclude_pheromones = np.asarray(exclude)
        return exclude_pheromones

    def get_move_mask_26_of_array(
        self, solid_array, grid_size, fly=False, check_self_collision=False
    ):
        """return move mask 1D array for the 26 nb voxel around self.pose
        the voxel
            most be non-solid
            must has at least one solid face_nb
        return:
            True if can not move there
            False if can move there

        if fly == True, cells do not have to be neighbors of solid
        """
        # get nb cell indices
        # nb_cells = self.get_nb_indices_6(self.pose)
        nb_cells = self.get_nb_indices_26(self.pose)
        cells_to_check = list(nb_cells)
        exclude = []  # FALSE if agent can move there, True if cannot
        # iterate through nb cells
        for nb_pose in cells_to_check:
            # check if nb cell is empty
            i, j, k = clip_indices_to_grid_size(nb_pose, grid_size)

            nb_value = solid_array[i][j][k]
            if check_self_collision:
                nb_value_collision = self.get_grid_value_at_index(
                    self.space_grid, nb_pose
                )
                nb_value += nb_value_collision
            # print(nb_value)
            if nb_value == 0:
                if not fly:
                    # check if nb cells have any face_nb cell which is solid
                    nbs_values = self.get_nb_values_6_of_array(
                        solid_array, grid_size, nb_pose
                    )
                    # print(nbs_values)
                    if np.sum(nbs_values) > 0:
                        exclude.append(False)
                    else:
                        exclude.append(True)
                else:
                    exclude.append(False)
            else:
                exclude.append(True)
        # print(exclude)
        exclude_pheromones = np.asarray(exclude)
        return exclude_pheromones

    def filter_move_map(
        self,
        solid_array,
        index_map_oriented,
        fly=False,
        agent_size=1,
        check_self_collision=False,
    ):
        """return move_option_indices, selection_index_list

        the grid in the index_map is checked:
        1. free of solid array
        2. self collision
        3. any solid close by (so agent doesn't fly, if fly false)
        """
        # get nb cell indices
        # nb_cells = self.get_nb_indices_6(self.pose)

        # # filter indices within solid
        # v = get_values_by_index_map(solid_array, index_map, nb_pose)
        # index_map = index_map[v == 0]
        cells_to_check = list(index_map_oriented)

        # exclude = []  # FALSE if agent can move there, True if cannot
        agent_size_index_map = index_map_sphere(agent_size, min_radius=0.1)

        # iterate through nb cells
        move_options = []
        # exclude = []
        option_index = []
        for i, nb_pose in enumerate(cells_to_check):
            # check if nb cell is empty
            e, f, g = nb_pose

            solid_value = solid_array[e, f, g]
            if solid_value > 0:
                # exclude.append(True)
                pass
            else:
                # check self collision
                indices = get_localized_map(
                    agent_size_index_map,
                    nb_pose,
                    clipping_box=self.space_grid.clipping_box,
                )
                self_collision_values = self.space_grid.get_values(indices)

                nb_value_collision = np.sum(self_collision_values)
                if nb_value_collision > 0:
                    # exclude.append(True)
                    pass
                # check move on ground
                else:
                    if not fly:
                        # check all nbs
                        nbs_values = get_values_using_index_map(
                            solid_array, agent_size_index_map, nb_pose
                        )
                        if np.sum(nbs_values) == 0:
                            move_options.append(nb_pose)
                            option_index.append(i)
                            # exclude.append(False)
                        else:
                            # exclude.append(True)
                            pass
                    else:
                        move_options.append(nb_pose)

                        option_index.append(i)
                        # exclude.append(False)

        return np.asarray(move_options), np.asarray(option_index)

    def get_localized_move_mask(self, array: npt.NDArray[np.float_]):
        localized_move_map = self.get_localized_move_map()
        filter = get_values_using_index_map(array, localized_move_map, self.pose)

        legal_move_mask = filter == 1

        return legal_move_mask

    def move_on_ground_by_pheromone_cube(
        self,
        ground,
        pheromone_cube,
        grid_size=None,
        fly=None,
        only_bounds=True,
        check_self_collision=False,
    ):
        cube = self.get_nb_indices_26(self.pose)

        # # limit options to inside
        # print(np.max(cube), np.min(cube))
        if only_bounds:
            cube = clip_indices_to_grid_size(cube, grid_size)

        # move on ground
        exclude = self.get_move_mask_26(
            ground, fly, check_self_collision=check_self_collision
        )
        pheromone_cube[exclude] = -1
        choice = np.argmax(pheromone_cube)
        # print('pose', self.pose)
        # print('choice:', choice)
        new_pose = cube[choice]
        # print('new_pose:', new_pose)

        # move
        self.pose = new_pose

        # update location in space grid
        self.space_grid.set_grid_value_at_index(self.pose, 1)
        return True

    def move_by_pheromones(
        self,
        solid_array,
        pheromone_cube,
        grid_size=None,
        fly=None,
        only_bounds=True,
        check_self_collision=False,
        random_batch_size: int = 1,
    ):
        """move in the direction of the strongest pheromone - random choice of
        best three
        checks invalid moves
        solid grid collision
        self collision
        selects a random direction from the 'n' best options
        return bool_
        """
        direction_cube = self.get_nb_indices_26(self.pose)

        # # limit options to inside
        if only_bounds:
            direction_cube = clip_indices_to_grid_size(direction_cube, grid_size)

        # add penalty for invalid moves based on an array
        exclude = self.get_move_mask_26_of_array(
            solid_array, grid_size, fly, check_self_collision=check_self_collision
        )
        pheromone_cube[exclude] = -1

        # select randomly from the best n value
        if random_batch_size <= 1:
            pass
            i = np.argmax(pheromone_cube)
        else:
            i = random_choice_index_from_best_n(pheromone_cube, random_batch_size)
        if pheromone_cube[i] == -1:
            return False

        # best option
        new_pose = direction_cube[i]

        # update space grid before move
        if self.leave_trace:
            self.track_grid.set_value_at_index(index=self.pose, value=1)
        self.space_grid.set_value_at_index(index=self.pose, value=0)

        # move
        self.pose = new_pose

        # update location in space grid
        self.space_grid.set_value_at_index(index=self.pose, value=1)
        return True

    def move_by_index_map(
        self,
        index_map_in_place,
        move_values,
        random_batch_size: int = 1,
    ):
        """move in the direction of the strongest pheromone - random choice of
        best three

        checks invalid moves
        solid grid collision
        self collision
        selects a random direction from the 'n' best options
        return bool_
        """
        # return early if move_values is empty
        if len(move_values) < 1:
            return False

        # if len(move_values) == 0 or
        # return if len(move_values) == 0 or np.max(move_values) == -1

        # CHOOSE WHERE TO MOVE
        # select randomly from the best n value
        if random_batch_size <= 1:
            i = np.argmax(move_values)
        else:
            i = random_choice_index_from_best_n(move_values, random_batch_size)

        # move
        self.pose = index_map_in_place[i]

        return True

    def get_direction_cube_values_for_grid_domain(self, grid: Grid, domain, strength=1):
        # mirrored above domain end and squeezed with the domain length
        # centered at 1
        ph_cube = self.get_grid_nb_values_26(grid, self.pose)
        start, end = domain
        center = (start + end) / 2
        # ph_cube -= center
        ph_cube = ((np.absolute(ph_cube - center) * -1) + center) * strength
        # print(ph_cube)
        return ph_cube

    def get_direction_cube_values_for_grid(self, grid: Grid, strength):
        ph_cube = self.get_grid_nb_values_26(grid, self.pose)
        return ph_cube * strength

    # METHODS TO CALCULATE BUILD PROBABILITIES

    def get_chances_by_density(
        self,
        diffusive_grid: Grid,
        build_if_over=0,
        build_if_below=5,
        erase_if_over=27,
        erase_if_below=0,
        build_strength=1,
        erase_strength=1,
    ):
        """
        returns build_chance, erase_chance
        if grid nb value sum is between
        """
        v = self.get_grid_nb_values_26(diffusive_grid, self.pose)
        v = np.sum(v)
        build_chance, erase_chance = [0, 0]
        if build_if_over < v < build_if_below:
            build_chance = build_strength
        if erase_if_over < v < erase_if_below:
            erase_chance = erase_strength
        return build_chance, erase_chance

    def get_chances_by_density_by_slice(
        self,
        diffusive_grid: Grid,
        slice_shape=(1, 1, 0, 0, 0, -1),
        build_if_over=0,
        build_if_below=5,
        erase_if_over=27,
        erase_if_below=0,
        build_strength=1,
        erase_strength=1,
    ):
        """
        returns build_chance, erase_chance
        if grid nb value sum is between
        [x_radius = 1,
        y_radius = 1,
        z_radius = 0,
        x_offset = 0,
        y_offset = 0,
        z_offset = 0] = slice_shape
        """
        # get the sum of the values in the slice
        rx, ry, rz, ox, oy, oz = slice_shape
        v = self.get_array_slice_parametric(
            diffusive_grid.array, rx, ry, rz, ox, oy, oz, self.pose, format_values=0
        )

        build_chance, erase_chance = [0, 0]
        if build_if_over < v < build_if_below:
            build_chance = build_strength
        if erase_if_over < v < erase_if_below:
            erase_chance = erase_strength
        return build_chance, erase_chance

    def get_chances_by_density_normal_by_slice(
        self,
        diffusive_grid: Grid,
        slice_shape=(1, 1, 0, 0, 0, -1),
        build_if_over=0,
        build_if_below=0.5,
        erase_if_over=0.9,
        erase_if_below=1,
        build_strength=1,
        erase_strength=1,
        _print=False,
    ):
        """
        returns build_chance, erase_chance
        if grid nb value sum is between
        [x_radius = 1,
        y_radius = 1,
        z_radius = 0,
        x_offset = 0,
        y_offset = 0,
        z_offset = 0] = slice_shape
        """
        # get the sum of the values in the slice
        sum_values = self.get_array_slice_parametric(
            diffusive_grid.array,
            *slice_shape,
            self.pose,
            format_values=0,
        )
        # print(v)
        radiis = slice_shape[:3]
        slice_volume = 1
        for x in radiis:
            slice_volume *= (x + 0.5) * 2
        density = sum_values / slice_volume
        if _print:
            print("slice v", density, slice_volume)
        build_chance, erase_chance = [0, 0]
        if build_if_over < density < build_if_below:
            build_chance = build_strength
        if erase_if_over < density < erase_if_below:
            erase_chance = erase_strength
        return build_chance, erase_chance

    def get_chance_by_relative_position(
        self,
        Grid: Grid,
        build_below=-1,
        build_aside=-1,
        build_above=1,
        build_strength=1,
    ):
        b, s, t = self.analyze_relative_position(Grid)
        build_chance = (
            build_below * b + build_aside * s + build_above * t
        ) * build_strength
        return build_chance

    def get_chance_by_pheromone_strength(
        self, diffusive_grid: Grid, limit1, limit2, strength, flat_value=True
    ):
        """gets pheromone v at pose.
        if in limits, returns strength or strength * value"""
        v = diffusive_grid.get_value(self.pose)
        # build
        if limit1 is None:
            flag = v <= limit2
        elif limit2 is None:
            flag = v >= limit1
        else:
            flag = limit1 <= v <= limit2
        if flag:
            if flat_value:
                return strength
            else:
                return v * strength
        else:
            return 0

    def match_vertical_move_history(self, last_moves_pattern):
        "chance is returned based on the direction values and chance_weight"
        n = len(last_moves_pattern)
        if len(self.move_history) < n:
            return False
        else:
            for i in range(n):
                x, y, z = self.move_history[-i - 1]
                pattern = last_moves_pattern[-i - 1]
                if (
                    pattern == "up"
                    and z > 0
                    or pattern == "side"
                    and z == 0
                    or pattern == "down"
                    and z < 0
                ):
                    flag = True
                else:
                    flag = False
            return flag

    #  BUILD/ERASE FUNCTIONS

    def build(self):
        grid = self.ground_grid
        try:
            grid.set_value_at_index(
                index=self.pose, value=1, wrapping=False, clipping=True
            )
            bool_ = True
            self.build_chance = 0
        except Exception as e:
            print(e)
            print("cant build here:", self.pose)
            bool_ = False
        return bool_

    def build_on_grid(self, grid: Grid, value=1.0):
        try:
            grid.set_value_at_index(
                index=self.pose, value=value, wrapping=False, clipping=True
            )
            self.build_chance = 0
            return True
        except Exception as e:
            print(e)
            print("cant build here:", self.pose)
            return False

    def erase(self, grid: Grid, only_face_nb=True):
        if only_face_nb:
            v = self.get_grid_nb_values_6(grid, self.pose, reintroduce=False)
            places = self.get_nb_indices_6(self.pose)
            places = np.asarray(places)
            choice = np.argmax(v)
            place = places[choice]
        else:
            v = self.get_grid_nb_values_26()
            choice = np.argmax(v)
            cube = self.get_nb_indices_26(self.pose)
            vector = cube[choice]
            place = self.pose + vector

        try:
            grid.set_value_at_index(index=place, value=0, wrapping=False, clipping=True)
            bool_ = True
            self.erase_chance = 0
        except Exception as e:
            print(e)
            print("cant erase this:", place)
            # print(places)
            # print(choice)
            # print(v)
            # x,y,z = place
            bool_ = False
        return bool_

    def set_grid_value_at_nbs_26(self, grid: Grid, value):
        nbs = self.get_nb_indices_26(self.pose)
        for pose in nbs:
            grid.set_value_at_index(
                index=pose, value=value, wrapping=False, clipping=True
            )

    def set_grid_value_at_nbs_6(self, grid: Grid, value):
        nbs = self.get_nb_indices_6(self.pose)
        for pose in nbs:
            grid.set_value_at_index(
                index=pose, value=value, wrapping=False, clipping=True
            )

    def set_grid_value_cross_shape(self, grid: Grid, value):
        dirs = [
            np.asarray([0, 0, 0]),
            np.asarray([-1, 0, 0]),
            np.asarray([1, 0, 0]),
            np.asarray([0, -1, 0]),
            np.asarray([0, 1, 0]),
        ]
        for dir in dirs:
            pose = self.pose + dir
            grid.set_value_at_index(
                index=pose, value=value, wrapping=False, clipping=True
            )

    def erase_6(self, grid: Grid):
        self.set_grid_value_at_nbs_6(grid, 0)

    def erase_26(self, grid: Grid):
        self.set_grid_value_at_nbs_26(grid, 0)

    def check_build_conditions(self, grid: Grid, only_face_nbs=True):
        if only_face_nbs:
            v = self.get_grid_nb_values_6(grid, self.pose)
            if np.sum(v) > 0:
                return True
        else:
            if get_sub_array(grid, 1, self.pose, format_values=0) > 0:
                return True
        return False

    def deploy_airborne(
        self, grid_deploy_on, grid_deploy_in, keep_move_history=False, ground_level_Z=0
    ):
        pose = get_any_free_voxel_above_array(
            grid_deploy_on.array, np.ones_like(grid_deploy_in.array)
        )

        if not isinstance(pose, np.ndarray | list):  # failed
            e, f, _g = grid_deploy_in.grid_size
            pose = get_random_index_in_zone_xxyy_on_Z_level(  # noqa: F821
                [0, e - 1, 0, f - 1], grid_deploy_in.grid_size, ground_level_Z
            )

        self.reset(pose_after_reset=pose, keep_move_history=keep_move_history)

        return pose

    def deploy_airborne_min(
        self, grid_deploy_on, grid_deploy_in, keep_move_history=False, ground_level_Z=0
    ):
        pose = get_lowest_free_voxel_above_array(
            grid_deploy_on.array, grid_deploy_in.array
        )
        print(f"reset to {pose}")
        # if not isinstance(pose, np.ndarray | list):
        #     e, f, _g = grid_deploy_in.grid_size
        #     pose = get_random_index_in_zone_xxyy_on_Z_level(  # noqa: F821
        #         [0, e - 1, 0, f - 1], grid_deploy_in.grid_size, ground_level_Z
        #     )

        self.reset(pose_after_reset=pose, keep_move_history=keep_move_history)

        return pose

    def reset(self, pose_after_reset=(0, 0, 0), keep_move_history=False):
        self.pose = pose_after_reset

        self.build_chance = 0
        self.erase_chance = 0
        self.track_flag = None
        self.passive_counter = 0
        if not keep_move_history:
            self.move_history = []

    def deploy_in_region(self, region, reset_move_history=True):
        pose = get_any_voxel_in_region(region)
        self.pose = pose
        self.build_chance = 0
        self.erase_chance = 0
        if reset_move_history:
            self.move_history = []

    def update_build_chance_random(self):
        a, b = self.build_prob_rand_range
        random_gain = r.random() * (b - a) + a
        self.build_chance += random_gain

    def update_erase_chance_random(self):
        a, b = self.erase_gain_random_range
        random_gain = np.random.random(1) * (b - a) + a
        self.erase_chance += random_gain

    def update_build_chance_max_density_above(
        self, volume_array, max_density_limit, radius, height, rate=0.5
    ):
        map = index_map_cylinder(radius, height)
        density_above = get_array_density_using_map(
            volume_array,
            map,
            self._pose + [0, 0, +1],
            nonzero=True,
        )
        if density_above > max_density_limit:
            self.build_chance = 0
        else:
            self.build_chance += rate

    def update_build_chance_max_density_around(
        self,
        volume_array,
        density_range,
        radius=2,
        offset_vector=(0, 0, 0),
        gain=0.5,
        penalty=-0.5,
    ):
        map = index_map_sphere(radius)
        density = self.get_array_density_by_index_map(
            volume_array,
            map,
            self._pose + offset_vector,
            nonzero=True,
        )
        min_, max_ = density_range
        if min_ <= density <= max_:
            self.build_chance += gain
        else:
            if not penalty:
                self.build_chance = 0
            else:
                self.build_chance += penalty

    def calculate_last_move_vector(self):
        last_pose = self.move_history[-1]
        pose = self._pose
        self.last_move_vector = pose - last_pose
        return self.last_move_vector

    def limit_move_mask_by_turn_degree(self):
        angle_limit = self.move_turn_degree
        last_move_v = cg.Vector(self.calculate_last_move_vector())
        map = self.move_shape_map
        mask = []
        for i in range(len(map)):
            v = map[i]
            map_vector = cg.Vector(v)
            angle = map_vector.angle(last_move_v, degrees=True)
            if angle > angle_limit:
                mask.append(False)
            else:
                mask.append(True)
        return np.array(mask, dtype=np.bool)

    def check_solid_collision(self, solid_grids: list[Grid]):
        "returns True if collision"

        if len(solid_grids) > 1:
            grid = solid_grids[0].merged_with(solid_grids[1:])
        else:
            grid = solid_grids[0]

        return grid.get_value(self._pose) != 0

    def get_localized_index_map(
        self,
        map_: npt.NDArray[np.int_],
        clipping_box: cg.Box | None = None,
    ):
        """Returns indices of the surrounding voxels based on the map.

        Formerly known as get_surrounding_indices_using map"""
        return get_localized_map(
            map_, self.pose, clipping_box=clipping_box or self.space_grid.clipping_box
        )

    def get_surrounding_values_using_map(
        self,
        grid: Grid,
        map: np.ndarray[np.int_ | bool],
    ):
        map = self.get_localized_index_map(map, clipping_box=grid.clipping_box)
        return grid.get_values(map)

    def get_localized_move_map(self):
        return self.get_localized_index_map(
            self.move_shape_map, clipping_box=self.space_grid.clipping_box
        )

    def modify_limit_in_density_range(
        self,
        array,
        radius=4,
        min_density=0.2,
        max_density=0.75,
        mod_below_range=-0.1,
        mod_in_range=0.5,
        mod_above_range=-0.1,
        nonzero=True,
    ):
        surrounding_map = index_map_sphere(radius)
        d = get_array_density_using_map(
            array, surrounding_map, self.pose, nonzero=nonzero
        )
        a, b, _ = [min_density, max_density, self.build_probability]
        if d < a:
            return mod_below_range
        elif d < b:
            return mod_in_range
        else:
            return mod_above_range

    def calculate_move_values_random__z_based(self):
        """moves agents in a calculated direction
        calculate weighted sum of slices of grids makes the direction_cube
        check and excludes illegal moves by replace values to -1
        move agent
        return True if moved, False if not or in ground
        """
        localized_move_map = self.get_localized_move_map()

        # random weights for map (0-1)
        random_map_values = np.random.random(size=len(localized_move_map)) + 0.5

        # global direction preference
        localized_z_component = localized_move_map[:, 2]
        # made global from the local map here, in case agent.move_shape_map is oriented
        z_component = localized_z_component.astype(np.float_) - self.pose[2]

        # MOVE PREFERENCE SETTINGS
        z_component *= self.move_mod_z
        random_map_values *= self.move_mod_random
        move_values = z_component + random_map_values  # + follow_map
        return move_values

    def calculate_move_values_random__z_based__follow(self, follow_grid: Grid):
        """Generate probability values for the movement index map to control
        movement.

        Weight add for Z component and follow component.
        """
        move_values = self.calculate_move_values_random__z_based()

        follow_map = follow_grid.get_values(self.get_localized_move_map())
        follow_map *= self.move_mod_follow

        move_values += follow_map

        return move_values

    def get_build_limit_by_density_range(self, array, radius, nonzero):
        """return build"""
        # check density

        surr_map = index_map_sphere(radius)
        d = self.get_array_density_by_index_map(array, surr_map, nonzero=nonzero)
        if self.build_by_density_random_factor != 0:
            r_mod = self.build_by_density_random_factor * r.random()

        if self.min_build_density <= d <= self.max_build_density:
            build_limit = 0 + r_mod
        else:
            build_limit = 1 - r_mod
        return build_limit

    # topology sense methods

    def get_nonzero_map_in_sense_range(self, ground_array=None, radius=None):
        if not ground_array:
            array = self.ground_grid.to_numpy()
        sense_map = self.sense_map.copy() if not radius else index_map_sphere(radius)
        filled_surrounding_indices = mask_index_map_by_nonzero(
            array, self.pose.tolist(), sense_map
        )

        return filled_surrounding_indices

    def get_filled_vectors_in_sense_map(self, ground_array=None, radius=None):
        filtered_nonzero_map = self.get_nonzero_map_in_sense_range(ground_array, radius)
        vectors = []
        for x, y, z in filtered_nonzero_map:
            vectors.append(cg.Vector(x, y, z))
        return vectors

    def calculate_normal_vector(self, ground_array=None, radius=None):
        """return self.normal_vector, self.normal_vector
        compas.geometry.Vector"""
        vectors = self.get_nonzero_map_in_sense_range(ground_array, radius)
        vectors = self.pose - vectors
        self.all_vectors = vectors
        if len(vectors) != 0:
            average_vector = np.sum(vectors, axis=0) / len(vectors)
            average_vector = cg.Vector(*average_vector)
            if average_vector.length != 0:
                average_vector.unitize()
                average_vector.invert()
                self._normal_vector = average_vector
                return average_vector
            else:
                print("normal vector calculation problem, average_vector.length = 0")
                self._normal_vector = cg.Vector.Zaxis().inverted()
                return self._normal_vector

        else:  # bug. originated from somewhere else. I think its fixed
            print(
                f"normal vector calculation problem. empty value list in array selection in pose: {self.pose}"
            )
            self._normal_vector = self._normal_vector = cg.Vector.Zaxis().inverted()
            return self._normal_vector

    def orient_index_map(self, index_map, new_origin=None, normal=None):
        """transforms shape map
        input:
        index_maps: list
        new_origins: None or Point
        normals: None or Vector"""
        if not new_origin:
            new_origin = self.pose
        if not normal:
            normal = self.normal_vector
        transformed_map = transform_index_map_to_plane(index_map, new_origin, normal)
        bounds = self.space_grid.grid_size
        clipped_map = clip_index_map(transformed_map, bounds)
        return clipped_map

    def orient_index_maps(self, index_maps, new_origins=None, normals=None):
        """transforms shape maps
        input:
        index_maps: list
        new_origins: None or list
        normals: None or list"""
        maps = []
        for i in range(len(index_maps)):
            new_origin = self.pose if not new_origins else new_origins[i]
            normal = self.normal_vector if not normals else normals[i]

            index_map = index_maps[i]

            transformed_map = self.orient_index_map(index_map, new_origin, normal)
            maps.append(transformed_map)
        return maps

    def orient_move_map(self):
        return self.orient_index_map(self.move_map)

    def orient_build_map(self):
        return self.orient_index_map(self.build_map)

    def orient_sense_map(self):
        return self.orient_index_map(self.sense_map)

    def orient_sense_depth_map(self):
        return self.orient_index_map(self.sense_depth_map)

    def orient_sense_overhang_map(self):
        map = self.orient_index_map(self.sense_overhang_map, normal=Vector(0, 0, 1))
        return map

    def orient_sense_nozzle_map(self, world_z=False):
        if not world_z and 180 > self.normal_angle > self.max_build_angle:
            x, y, z = self.normal_vector
            v2 = Vector(x, y, 0)
            v2.unitize()
            x, y, _ = v2
            z = sin(radians(self.max_build_angle)) * -1
            x = cos(radians(self.max_build_angle)) * x
            y = cos(radians(self.max_build_angle)) * y
            v = Vector(x, y, z)
            # print(f"adjusted_nozzle_angle {v.angle(Vector(0,0,-1), True)}")
        elif not world_z:
            v = self.normal_vector
            v.invert()
        else:
            v = Vector(0, 0, 1)
        map = self.orient_index_map(self.sense_nozzle_map, normal=v)
        return map

    def orient_sense_inplane_map(self):
        return self.orient_index_map(self.sense_inplane_map)

    def get_normal_angle(self):
        v = self.normal_vector
        angle = v.angle([0, 0, -1], degrees=True)
        self._normal_angle = angle
        return angle
