import numpy as np


def get_any_index_of_mask(index_map_array, nonzero=True):
    """Returns a random index of a mask array"""
    if nonzero:
        indices = np.argwhere(index_map_array != 0)
    else:
        indices = np.argwhere(index_map_array == 1)
    map_len = len(indices)
    i = np.random.choice(map_len)
    return indices[i]


def get_lowest_free_voxel_above_array(array_to_search_on, search_boundary):
    """finds the lowest index of a free voxel on an array along Z vector(axis = 2)
    in the boundary. Search where boundary = 1 and where array = 0.

    input:
        array_to_search_on : int array'
        search_boundary : int array

    returns the index np.ndarray([x, y, z])"""
    printed = array_to_search_on.copy()
    printed = np.ceil(printed)
    design = search_boundary.copy()
    design = np.ceil(design)
    not_design = np.ones_like(printed) - design
    printed = np.pad(printed, (1, 1), "constant", constant_values=-1)
    shifted = np.roll(printed, 1, 2)
    options_array = printed - shifted
    options_array = options_array[1:-1, 1:-1, 1:-1] + not_design
    sum_per_layer = np.sum(np.clip(options_array, -1, 0), axis=1)
    nonzeros_per_layer = np.count_nonzero(sum_per_layer, axis=0)
    if sum(nonzeros_per_layer) > 0:
        layer_index = np.argwhere(nonzeros_per_layer != 0)[0][0]
        layer = options_array[:, :, layer_index]
        x, y = np.argwhere(layer == -1)[0]
        z = layer_index
        place = x, y, z
        return np.array(place)
    else:
        return None


def get_any_free_voxel_above_array(array_to_search_on, search_boundary):
    """finds the lowest index of a free voxel on an array along Z vector(axis = 2)
    in the boundary. Search where boundary = 1 and where array = 0.

    input:
        array_to_search_on : int array'
        search_boundary : int array

    returns the index np.ndarray([x, y, z])"""
    printed = array_to_search_on.copy()
    printed = np.ceil(printed)
    design = search_boundary.copy()
    design = np.ceil(design)
    not_design = np.ones_like(printed) - design
    printed = np.pad(printed, (1, 1), "constant", constant_values=-1)
    shifted = np.roll(printed, 1, 2)
    # print(f"ground sum: {np.sum(printed)}")
    options_array = printed - shifted
    options_array = options_array[1:-1, 1:-1, 1:-1] + not_design
    if np.sum(options_array) > 0:
        options_array = np.clip(options_array, -1, 0)
        # print(f"options_array {options_array}")
        place_options = np.argwhere(options_array == -1)
        # print(f"place_options {place_options}")
        i = np.random.choice(np.shape(place_options[0]))
        x, y, z = place_options[i]
    else:
        return None


def get_random_index_in_zone_xxyy_on_Z_level(
    deployment_zone_xxyy, grid_size: int | tuple[int, int, int], ground_level=0
):
    """
    return random voxel index in deployment zone, on ground (axis = 2)

    z = ground_level + 1
    x1, x2, y1, y2 = deployment_zone_xxyy (include, exclude, include, exclude)
    """
    x1, x2, y1, y2 = deployment_zone_xxyy
    if isinstance(grid_size, int):
        a = grid_size
        b = grid_size
    else:
        a, b, _ = grid_size
    x1 = max(x1, 0)
    x2 = min(x2, a - 1)
    x2 = max(x1 + 1, x2)
    y1 = max(y1, 0)
    y2 = min(y2, b - 1)
    y2 = max(y1 + 1, y2)
    x = np.random.randint(x1, x2)
    y = np.random.randint(y1, y2)
    z = ground_level + 1

    return np.array([x, y, z])
