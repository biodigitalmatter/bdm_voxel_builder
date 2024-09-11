from compas import json_load  # noqa: F401
from compas_viewer import Viewer

from bdm_voxel_builder import REPO_DIR  # noqa: F401

viewer = Viewer()

# # scene path
# path = (
#     REPO_DIR
#     / "docs/algo_8c/2024-07-30_16_48_16_algo_8c_testED_reset_True_i1000a10_a10_i1000.json"  # noqa: E501
# )
# # make view from loaded scene
# viewer.scene = json_load(path)

viewer.show()
