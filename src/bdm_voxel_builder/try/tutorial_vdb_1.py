import numpy as np
import pyopenvdb as vdb

cube = vdb.FloatGrid()
cube.fill(min=(100, 100, 100), max=(199, 199, 199), value=1)
cube.name = "density"

sphere = vdb.createLevelSetSphere(radius=50, center=(1, 2, 3))
sphere["radius"] = 50

sphere.name = "sphere"

filename = "temp\\vdb_tests\\mygrids.vdb"
vdb.write(filename, grids=[cube, sphere])
print("done?")
