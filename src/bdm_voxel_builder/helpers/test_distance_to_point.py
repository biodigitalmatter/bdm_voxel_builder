import numpy as np

from bdm_voxel_builder.helpers.array import distance_to_point

array = np.ones([3, 3, 3])
d = distance_to_point(array, [1, 215, 1])
print(d)
