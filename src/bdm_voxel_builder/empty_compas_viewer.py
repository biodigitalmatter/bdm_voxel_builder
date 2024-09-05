import os
from compas import json_load
from compas_viewer import Viewer

# algo8c
path_1 = "/Users/laszlomangliar/Documents/GitHub/bdm_voxel_builder/docs/algo_8c/2024-07-30_16_48_16_algo_8c_testED_reset_True_i1000a10_a10_i1000.json"

# algo8d
path_1 = "/Users/laszlomangliar/Documents/GitHub/bdm_voxel_builder/docs/algo_8d/2024-08-01_09_58_34_algo_8d_test_i1000a35_a35_i1000.json"

# algo_8e
folder = '/Users/laszlomangliar/Documents/GitHub/bdm_voxel_builder/docs/algo_8e/'
file1 = "2024-08-06_17_27_32_test_8_e_build_ridge_decay_i2000a1_a1_i2000" # decay?
file2 = '2024-08-06_17_31_33_test_8_e_build_ridge_overhang_i1000a50_a50_i1000' # overhang
file3 = '2024-08-06_17_49_35_test_8_e_build_ridge_i1000a50_a50_i1000' # normal
path = os.path.join(folder, file3 + '.json')

# algo_9a erase
path = "/Users/laszlomangliar/Documents/GitHub/bdm_voxel_builder/docs/algo_9a/2024-08-01_10_58_00_algo_9a_test_reset_i100a5_a5_i100.json"


# make view from loaded scene
scene_file = json_load(path)
viewer = Viewer()
viewer.scene = scene_file

viewer.show()


