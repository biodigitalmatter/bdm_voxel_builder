import numpy as np
from compas.geometry import Pointcloud

from bdm_voxel_builder.helpers.numpy import (
    convert_pointcloud_to_grid_array,
)

print("hello, lets test this!\n")


n = 5
pt_array = np.random.random(n * 3).reshape([n, 3]) * 1200 - 100
# pt_array2 = np.transpose(pt_array)

pts = pt_array.tolist()
print(pt_array.shape)
pointcloud = Pointcloud(pts)
print(pointcloud)
array = convert_pointcloud_to_grid_array(pointcloud, unit_in_mm=10)
print(array)
