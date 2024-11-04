import random as r
from copy import copy
from dataclasses import dataclass

import compas.geometry as cg
import numpy as np
import numpy.typing as npt

from bdm_voxel_builder.agent_algorithms.common import (
    get_any_index_of_mask,
)
from bdm_voxel_builder.grid import Grid
from bdm_voxel_builder.helpers import (
    get_array_density_using_index_map,
    get_localized_index_map,
    get_values_using_index_map,
    index_map_cylinder,
    index_map_sphere,
    mask_index_map_by_nonzero,
    random_choice_index_from_best_n,
)


@dataclass
class Agent:
    """Object based voxel builder, a guided random_walker"""

    initial_pose: npt.NDArray[np.int_] | None = None
    leave_trace: bool = True

    space_grid: Grid = None
    track_grid: Grid = None
    ground_grid: Grid = None

    move_history: list[tuple[int, int, int]] = None
    save_move_history = True
    track_flag = None
    step_counter = 0
    passive_counter = 0

    id = 0

    build_h = 2
    build_random_chance = 0.1
    build_random_gain = 0.6
    max_build_angle = 90

    walk_radius = 4
    min_walk_radius = 0
    build_radius = 3
    sense_radius = 5
    sense_wall_radar_map = None
    sense_overhang_map = None
    sense_nozzle_map = None
    sense_topology_bool = True

    min_build_density = 0
    max_build_density = 1
    build_limit_mod_by_density = [0.5, -0.5, 0.5]
    build_by_density = False
    build_by_density_random_factor = 0
    max_shell_thickness = 15
    overhang_density = 0.5

    move_mod_z = 0
    move_mod_random = 0.1
    move_mod_follow = 1

    reset_after_build = False
    reset_after_erase = False

    inactive_step_count_limit = None

    last_move_vector = []
    move_turn_degree = None

    pref_build_angle = 25.0  # degrees
    pref_build_angle_gain = 0.0

    _normal_vector = cg.Vector(0, 0, 1)

    # backs properties
    _normal_vector = None
    _build_map = None
    _move_map = None
    _sense_map = None
    _sense_inplane_map = None
    _sense_depth_map = None

    def __post_init__(self):
        self._pose = self.initial_pose or np.array([0, 0, 0], dtype=np.int_)

        self.space_grid = self.space_grid or Grid()
        self.track_grid = self.track_grid or Grid()
        self.ground_grid = self.ground_grid or Grid()

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, new_pose):
        old_pose = self.pose

        if self.move_history is None:
            self.move_history = []

        self.move_history.append(old_pose)

        if self.leave_trace:
            self.track_grid.set_value(old_pose, 1)

        self.space_grid.set_value(old_pose, 0)
        self.space_grid.set_value(new_pose, 1)
        self._pose = new_pose

    @property
    def build_map(self):
        return self._build_map or index_map_cylinder(
            self.build_radius, self.build_h, min_radius=0, z_lift=-1
        )

    @build_map.setter
    def build_map(self, index_map):
        self._build_map = index_map

    @property
    def move_map(self):
        return self._move_map or index_map_sphere(
            self.walk_radius, min_radius=self.min_walk_radius
        )

    @move_map.setter
    def move_map(self, index_map):
        self._move_map = index_map

    @property
    def sense_map(self):
        return self._sense_map or index_map_sphere(self.sense_radius)

    @sense_map.setter
    def sense_map(self, index_map):
        self._sense_map = index_map

    @property
    def sense_inplane_map(self):
        return self._sense_inplane_map or index_map_cylinder(
            radius=3, height=2, min_radius=0, z_lift=1
        )

    @sense_inplane_map.setter
    def sense_inplane_map(self, index_map):
        self._sense_inplane_map = index_map

    @property
    def sense_depth_map(self):
        return self._sense_depth_map or index_map_cylinder(
            1, self.max_shell_thickness * 2, 0, 1
        )

    @sense_depth_map.setter
    def sense_depth_map(self, index_map):
        self._sense_depth_map = index_map

    def copy(self):
        return copy(self)

    def get_localized_move_mask(self, array: npt.NDArray[np.float_]):
        filter_ = get_values_using_index_map(
            array,
            self.move_map,
            self.pose,
            clipping_box=self.space_grid.clipping_box,
        )

        legal_move_mask = np.logical_not(filter_ == 0)

        return legal_move_mask

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

    #  BUILD/ERASE FUNCTIONS

    def reset(self, pose_after_reset=(0, 0, 0), keep_move_history=False):
        self.pose = pose_after_reset

        self.build_chance = 0
        self.erase_chance = 0
        self.track_flag = None
        self.passive_counter = 0
        if not keep_move_history:
            self.move_history = []

    def deploy_in_region(self, region, reset_move_history=True):
        pose = get_any_index_of_mask(region)
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
        density_above = get_array_density_using_index_map(
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
        pose = self.pose
        self.last_move_vector = pose - last_pose
        return self.last_move_vector

    def limit_move_mask_by_turn_degree(self):
        angle_limit = self.move_turn_degree
        last_move_v = cg.Vector(self.calculate_last_move_vector())
        map = self.move_map
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
        return get_localized_index_map(
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
            self.move_map, clipping_box=self.space_grid.clipping_box
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
        d = get_array_density_using_index_map(
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
        # made global from the local map here, in case agent.move_map is oriented
        z_component = localized_z_component.astype(np.float_) - self.pose[2]

        # MOVE PREFERENCE SETTINGS
        z_component *= self.move_mod_z
        random_map_values *= self.move_mod_random
        move_values = z_component + random_map_values  # + follow_map
        return move_values

    def get_follow_modifiers(self, follow_grid: Grid):
        """Set follow modifiers based on the follow grid"""
        follow_mods = np.array(follow_grid.get_values(self.get_localized_move_map()))
        return follow_mods * self.move_mod_follow

    def calculate_move_values_random__z_based__follow(self, follow_grid: Grid):
        """Generate probability values for the movement index map to control
        movement.

        Weight add for Z component and follow component.
        """
        return self.calculate_move_values_random__z_based() + self.get_follow_modifiers(
            follow_grid
        )

    def calculate_move_values_random__z_based__follow_with_low_density_weight(
        self,
        follow_grid: Grid,
        built_grid: Grid,
        low_density_threshold=0.1,
        low_density_weight=1e7,
    ):
        """Generate probability values for the movement index map to control
        movement.

        Weight add for Z component and follow component.
        """
        move_values = self.calculate_move_values_random__z_based()

        follow_mods = self.get_follow_modifiers(follow_grid)

        built_density = get_array_density_using_index_map(
            built_grid.to_numpy(),
            self.get_localized_move_map(),
            self.pose,
            nonzero=True,
        )

        if built_density < self.min_build_density:
            follow_mods += low_density_weight

        return move_values + follow_mods

    def get_build_limit_by_density_range(self, array, radius, nonzero):
        """return build"""
        # check density

        surr_map = index_map_sphere(radius)

        d = get_array_density_using_index_map(
            array,
            surr_map,
            self.pose,
            clipping_box=self.space_grid.clipping_box,
            nonzero=nonzero,
        )
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
            array, self.pose, sense_map, clipping_box=self.space_grid.clipping_box
        )

        return filled_surrounding_indices

    def get_filled_vectors_in_sense_map(self, ground_array=None, radius=None):
        filtered_nonzero_map = self.get_nonzero_map_in_sense_range(ground_array, radius)
        vectors = []
        for x, y, z in filtered_nonzero_map:
            vectors.append(cg.Vector(x, y, z))
        return vectors
