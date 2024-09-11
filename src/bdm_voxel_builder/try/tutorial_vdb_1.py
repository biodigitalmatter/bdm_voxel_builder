import pyopenvdb as vdb

cube = vdb.FloatGrid()
s = 1
e = 3
cube.fill(min=(s, s, s), max=(e, e, e), value=1)
cube.name = "density"

cube2 = vdb.FloatGrid()
s = 5
e = 20
cube2.fill(min=(s, s, s), max=(e, e, e), value=1)
cube2.name = "density"

# sphere = vdb.createLevelSetSphere(radius=50, center=(1, 2, 3))
# sphere["radius"] = 50

# sphere.name = "sphere"

filename = "temp\\vdb_tests\\mygrids3.vdb"
vdb.write(filename, grids=[cube, cube2])
print("done?")
