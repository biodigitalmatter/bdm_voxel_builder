from math import cos, radians, sin

import compas.geometry as cg
import numpy as np

from bdm_voxel_builder.agents import Agent
from bdm_voxel_builder.helpers import transform_index_map_to_plane


class OrientableAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._normal_vector = None

    @property
    def normal_vector(self):
        """Normal vector of agent, default is Z axis inverted."""
        return self._normal_vector or cg.Vector.Zaxis().inverted()

    @normal_vector.setter
    def normal_vector(self, vector):
        self._normal_vector = cg.Vector(*vector)

    def set_normal_vector_using_sense_map(self, ground_array=None, radius=None):
        """return self.normal_vector, self.normal_vector
        compas.geometry.Vector"""
        vectors = self.get_nonzero_map_in_sense_range(ground_array, radius)
        vectors = self.pose - vectors
        if len(vectors) != 0:
            average_vector = np.sum(vectors, axis=0) / len(vectors)
            average_vector = cg.Vector(*average_vector)
            if average_vector.length != 0:
                average_vector.unitize()
                average_vector.invert()
                self.normal_vector = average_vector
            else:
                print("normal vector calculation problem, average_vector.length = 0")

        else:  # bug. originated from somewhere else. I think its fixed
            print(
                "normal vector calculation problem."
                + f"empty value list in array selection in pose: {self.pose}"
            )
        return self.normal_vector

    def get_plane(self):
        return cg.Plane(self.pose, self.normal_vector)

    def orient_index_map(self, index_map, normal: cg.Vector | None = None):
        """transforms shape map
        input:
        index_maps: list
        new_origins: None or Point
        normals: None or Vector"""
        return transform_index_map_to_plane(
            index_map,
            self.pose,
            normal or self.normal_vector,
            clipping_box=self.space_grid.clipping_box,
        )

    def orient_move_map(self):
        return self.orient_index_map(self.move_map)

    def orient_build_map(self):
        return self.orient_index_map(self.build_map)

    def orient_sense_map(self):
        return self.orient_index_map(self.sense_map)

    def orient_sense_depth_map(self):
        return self.orient_index_map(self.sense_depth_map)

    def orient_sense_wall_radar_map(self):
        return self.orient_index_map(self.sense_wall_radar_map)

    def orient_sense_overhang_map(self):
        map = self.orient_index_map(self.sense_overhang_map)
        return map

    def orient_sense_nozzle_map(self, world_z=False):
        if not world_z and 180 > self.get_normal_angle() > self.max_build_angle:
            x, y, z = self.set_normal_vector_using_sense_map()
            v2 = cg.Vector(x, y, 0)
            v2.unitize()
            x, y, _ = v2
            z = sin(radians(self.max_build_angle)) * -1
            x = cos(radians(self.max_build_angle)) * x
            y = cos(radians(self.max_build_angle)) * y
            v = cg.Vector(x, y, z)
            # print(f"adjusted_nozzle_angle {v.angle(Vector(0,0,-1), True)}")
        elif not world_z:
            v = self.set_normal_vector_using_sense_map().inverted()
        else:
            v = cg.Vector.Zaxis()
        map = self.orient_index_map(self.sense_nozzle_map, normal=v)
        return map

    def orient_sense_inplane_map(self):
        return self.orient_index_map(self.sense_inplane_map)

    def get_normal_angle(self):
        return self.normal_vector.angle(cg.Vector.Zaxis().inverted(), degrees=True)
