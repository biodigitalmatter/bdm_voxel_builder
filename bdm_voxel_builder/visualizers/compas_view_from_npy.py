import numpy as np
from compas.geometry import Pointcloud

def convert_array_to_points(a, list_output = False):
    indicies = np.indices(a.shape)
    pt_location = np.logical_not(a == 0)
    coordinates = []
    for i in range(3):
        c = indicies[i][pt_location]
        coordinates.append(c)
    if list_output:
        pts = np.vstack(coordinates).transpose().tolist()
    else:
        pts = np.vstack(coordinates).transpose()
    return pts



# LOAD NPY

file = r"temp\2024-07-24_20_47_07_build_on_existing_a200_i3000.npy"

array = np.load(file)
print(array.size)

pts = convert_array_to_points(array, True)

pointcloud = Pointcloud(pts)


# =============================================================================
 # SHOW network
from compas_view2.app import App
viewer = App(width=600, height=600)
viewer.view.camera.rx = -60
viewer.view.camera.rz = 100
viewer.view.camera.ty = -2
viewer.view.camera.distance = 20

viewer.add(pointcloud)

print('show starts')
viewer.show()

print('show done')

