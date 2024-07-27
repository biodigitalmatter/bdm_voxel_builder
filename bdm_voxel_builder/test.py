from bdm_voxel_builder import data_layer as dl
import numpy as np
print('hello')
array = np.zeros([3,3,3])
try:
    v = array[4][5][2]
except Exception as e:
    print(e)
    v = 0
print(v)