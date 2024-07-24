from bdm_voxel_builder import data_layer as dl
print('hello')
layer = dl.DataLayer(voxel_size=5)

layer.add_values_in_zone_xxyyzz([1,6,1,2,3,9], value = 1, add_values = False)

print(layer.array)