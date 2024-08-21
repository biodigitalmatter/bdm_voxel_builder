import numpy as np

def index_map_cube(radius, min_radius = None):
    """including max size, excluding min size"""

    d = np.int_(np.floor(radius * 2)) + 1
    # print(d)
    x,y,z = np.indices([d,d,d])
    r2 = radius
    r2x, r2y, r2z = [np.floor(r2)] * 3
    x,y,z = x-r2x, y-r2y, z-r2z
    index_map = np.array([x,y,z], dtype=np.int64)
    x,y,z = index_map
    # print(f'base map: {x,y,z}')
    abs_x, abs_y, abs_z = np.absolute(index_map)
    if not min_radius: min_radius = 0
    mask = np.logical_or(abs_x > min_radius, abs_z > min_radius)
    mask = np.logical_or(abs_y > min_radius, mask)
    print(mask)


    return index_map.transpose()[mask]

# print(index_map_cube(1.1, 0.1))


def index_map_box(box_size, box_min_size = None):
    """including max size, excluding min size"""
    radius = np.array(box_size)
    if not box_min_size: box_min_size = [0,0,0]
    min_radius = np.array(box_min_size)
    d = np.int_(np.floor(radius * 2)) + 1
    # print(d)
    x,y,z = np.indices(d)
    rx, ry, rz = np.floor(radius)
    x,y,z = [x-rx, y-ry, z-rz]
    index_map = np.array([x,y,z], dtype=np.int64)
    x,y,z = index_map
    # print(f'base map: {x,y,z}')
    abs_x, abs_y, abs_z = np.absolute(index_map)
    mask1 = np.logical_or(abs_x > min_radius[0], abs_z > min_radius[2])
    mask2 = np.logical_or(abs_y > min_radius[1], abs_z > min_radius[2])
    mask = np.logical_or(mask1, mask2)

    filtered_index_map = index_map[:, mask].transpose()
    print(mask.shape)
    index_map = np.array(index_map).transpose()

    return filtered_index_map

print(index_map_box([1,1,1]))


def index_map_spheric(radius = 1.5, min_radius = None):
    d = int(np.ceil(radius) * 2) + 1
    x,y,z = np.indices([d,d,d])
    r2 = np.ceil(radius)
    indices = [x - r2, y - r2, z - r2]
    l = np.linalg.norm(indices, axis=0)
    # print(l)
    mask = l <= radius
    if min_radius:
        mask2 = l >= min_radius
        mask = np.logical_and(mask, mask2)
    # print(mask)
    # print(indices)
    x = indices[0][mask]
    y = indices[1][mask]
    z = indices[2][mask]
    sphere_array = np.array([x, y, z], dtype=np.int64)
    return sphere_array.transpose()
# r = 2.2
# sphere_array = get_spheric_indices(r)
# print(len(sphere_array))
# print(sphere_array)

def index_map_cylinder(radius = 3, h = 2, min_radius = None):
    d = int(np.ceil(radius) * 2) + 1
    x,y,z = np.indices([d,d,h])
    r2 = np.ceil(radius)
    x,y,z = [x - r2, y - r2, z]
    l1 = np.linalg.norm([x,y], axis=0)
    # print(l1)

    height_condition = z < h
    radius_condition = l1 <= radius

    if min_radius:
        radius_min_condition = l1 >= min_radius
        mask = np.logical_and(mask, radius_min_condition)
        mask = np.logical_and(mask, height_condition)
    else:
        mask = np.logical_and(radius_condition, height_condition)


    print(f'mask {mask}')
    indices = [x,y,z]
    x = indices[0][mask]
    y = indices[1][mask]
    z = indices[2][mask]
    index_map = np.array([x, y, z], dtype=np.int64)
    return index_map.transpose()

# print(get_cylinder_indices_limit(2.2,1.9,1))


def index_map_spheric_scale_NU(radius = 1.5, scale_NU = [1, 1, 0.5], min_radius = None):
    """returns index map"""
    original_radius = radius
    scale_NU = np.array(scale_NU)
    radius = radius * scale_NU
    scale_down = 1 / scale_NU

    d = np.int_(np.ceil(radius) * 2) + 1
    x,y,z = np.indices(d)

    r2 = np.ceil(radius)
    x2 = x - r2[0]
    y2 = y - r2[1]
    z2 = z - r2[2]
    sx, sy, sz = scale_down
    indices = np.array([x2 * sx, y2 * sy, z2 * sz])
    indices_real = np.array([x2, y2, z2])

    l = np.linalg.norm(indices, axis=0)
    # print(l)
    if min_radius:
        mask = min_radius <= l <= original_radius
    else:
        mask = l <= original_radius
    # print(mask)

    x = indices_real[0][mask]
    y = indices_real[1][mask]
    z = indices_real[2][mask]

    arr = np.array([x, y, z], dtype=np.int64)
    return arr.transpose()



def get_array_values_by_index_map_at_point(
        array: np.ndarray, point: tuple[int, int, int], index_map: np.ndarray
    ):
    shape = array.shape
    index_map += np.int_(point)
    print(index_map)
    index_map = (np.clip(index_map, [0,0,0], shape))
    index_map = np.unique(index_map, axis = 0)
    print(index_map)
    v = []
    for x,y,z in index_map:
        v.append(array[x,y,z])
    return v

def get_array_values_by_index_map_at_point(
        array: np.ndarray, point: tuple[int, int, int], index_map: np.ndarray
    ):
    index_map += np.int_(point)
    print(index_map)
    index_map = (np.clip(index_map, [0,0,0], array.shape))
    index_map = np.unique(index_map, axis = 0)
    print(index_map)
    v = []
    for x,y,z in index_map:
        v.append(array[x,y,z])
    return v


def get_value_by_index_map(array, index_map, index_map_origin = [0,0,0], return_list=False):
    """
    Filters the values of a 3D NumPy array based on an index map.
        numpy.ndarray: A 1D array containing the filtered values.
    """
    # Convert the index_map to a tuple of lists, suitable for NumPy advanced indexing
    index_map += index_map_origin
    index_map = np.unique(
        (np.clip(index_map, [0,0,0], array.shape)), axis = 0
        )
    indices = tuple(np.array(index_map).T)

    # Extract the elements using advanced indexing
    filtered_values = array[indices]
    if return_list:
        return filtered_values.tolist()
    else:
        return filtered_values


def set_value_by_index_map(array, index_map, index_map_origin = [0,0,0], value = 1, return_list=False):
    """
    Filters the values of a 3D NumPy array based on an index map.
        numpy.ndarray: A 1D array containing the filtered values.
    """
    # Convert the index_map to a tuple of lists, suitable for NumPy advanced indexing
    index_map += index_map_origin
    index_map = np.unique(
        (np.clip(index_map, [0,0,0], array.shape)), axis = 0
        )
    indices = tuple(np.array(index_map).T)

    # Extract the elements using advanced indexing
    array[indices] = value 
    return array

# index_map = index_map_spheric(1)

# # array = np.ones([4,4,4])
# n = 4
# array = np.arange(n*n*n).reshape([n,n,n])
# print(array)
# values = get_array_values_by_index_map_at_point(array, [0,0,0], index_map)
# print(values)



# # Example usage
# # index_map = [[0, 0, 0], [0, 0, 1], [2, 2, 3]]  # Example index map
# origin = [1,1,1]
# filtered_values = filter_array_by_index_map(array, index_map, origin = origin)
# print(f'filtered_values { filtered_values}')

