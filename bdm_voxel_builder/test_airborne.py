import numpy as np

n = 5
array = np.zeros([n, n, n])

design = array.copy()
# design[1:4, 1:4, 0:5] = 1
printed = array.copy()
printed[1:3, 1:3, :] = 0.001
printed[:, :, 3:] = 0.000001
printed = np.ceil(printed)
design = printed.copy()
printed[1, 1, 3] = 0
printed[3, 3, 4] = 0
printed[4, 4, 3] = 0
# printed = np.ones_like(design)
print(f"printed {printed}")


def get_airborne_deployment_index(printed_array, design_array):
    """find the lowest free voxel on the printed array along Z vector(axis = 2)
    in the design boundary
    return index (np.ndarray)"""
    air = np.ones_like(printed_array)
    array = air - design_array
    array += printed
    print(f"array to fly on\n{array}")

    nonzeros_per_layer = np.count_nonzero(array, axis=1)
    print(f"nonzeros \n{nonzeros_per_layer}")

    a, b, c = air.shape
    print(f"a {a}, b {b}")
    area = a * b
    zeros_per_layer = area - (np.sum(nonzeros_per_layer, axis=0))
    if sum(zeros_per_layer) > 0:
        print(f"sum_nonzeros {zeros_per_layer}")
        # first_layer_with_a_free_voxel
        mins = np.argwhere(zeros_per_layer != 0)
        print(mins)
        layer_index = mins[0][0]
        layer = array[:, :, layer_index]
        # print(f"lowest layer with place: \n{layer}")
        options = np.argwhere(layer == 0)
        # print(options)
        x, y = options[0]
        z = layer_index
        place = [x, y, z]
        # print(place)
        return np.array(place)
    else:
        return None


def get_lowest_free_voxel_above_array(array_to_search_on, search_boundary):
    """finds the lowest index of a free voxel on an array along Z vector(axis = 2)
    in the boundary. Search where boundary = 1 and where array = 0.

    input:
        array_to_search_on : int array'
        search_boundary : int array

    returns the index np.ndarray([x, y, z])"""
    print(f"original_boundary:\n{search_boundary}")
    print(f"printed array:\n{array_to_search_on}")

    air = np.zeros_like(array_to_search_on)
    air += search_boundary
    array = air - array_to_search_on
    # print(f"array to fly on\n{array}")
    array = np.pad(array, (1, 1), "constant", constant_values=0)
    shifted = np.roll(array, 1, 2)
    array -= shifted
    array = array[1:-1, 1:-1, 1:-1]
    print(f"bound array\n{array}")

    sum_per_layer = np.sum(np.clip(array, -1, 0), axis=1)

    print(sum_per_layer)
    nonzeros_per_layer = np.count_nonzero(sum_per_layer, axis=0)
    if sum(nonzeros_per_layer) > 0:
        print(f"nonzeros \n{nonzeros_per_layer}")
        layer_index = np.argwhere(nonzeros_per_layer != 0)[0][0]
        print(layer_index)
        layer = array[:, :, layer_index]
        print(f"layer:\n{layer}")
        options = np.argwhere(layer == -1)[0]
        print(f"options {options}")

        x, y = options
        z = layer_index
        place = x, y, z
        print(f"place: {place}")
        return np.array(place)
    else:
        return None


place = get_lowest_free_voxel_above_array(printed, design)
print(place)
