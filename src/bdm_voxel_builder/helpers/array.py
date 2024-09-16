from math import ceil

import numpy as np
import numpy.typing as npt

NB_INDEX_DICT = {
    "up": np.asarray([0, 0, 1]),
    "left": np.asarray([-1, 0, 0]),
    "down": np.asarray([0, 0, -1]),
    "right": np.asarray([1, 0, 0]),
    "front": np.asarray([0, -1, 0]),
    "back": np.asarray([0, 1, 0]),
}


def sort_pts_by_values(arr: npt.NDArray, multiply=1):
    """returns sortedpts, values"""
    indices = np.indices(arr.shape)
    pt_location = np.logical_not(arr == 0)
    coordinates = []
    for i in range(3):
        c = indices[i][pt_location]
        coordinates.append(c)
    pts = np.vstack(coordinates).transpose().tolist()
    values = arr[pt_location].tolist()
    # sort:

    # Pair the elements using zip
    paired_lists = list(zip(values, pts, strict=False))

    # Sort the paired lists based on the first element of the pairs (values values)
    sorted_paired_lists = sorted(paired_lists, key=lambda x: x[0])

    # Extract the sorted nested list

    sortedpts = [element[1] for element in sorted_paired_lists]
    values = [element[0] * multiply for element in sorted_paired_lists]

    return sortedpts, values


def create_random_array(shape: int | tuple[int]):
    if isinstance(shape, int):
        shape = [shape] * 3

    return np.random.default_rng().random(shape)


def get_cube_array_indices(self_contain=False):
    """26 nb indices, ordered: top-middle-bottom"""
    # horizontal
    f = NB_INDEX_DICT["front"]
    b = NB_INDEX_DICT["back"]
    le = NB_INDEX_DICT["left"]
    r = NB_INDEX_DICT["right"]
    u = NB_INDEX_DICT["up"]
    d = NB_INDEX_DICT["down"]
    # first_story in level:
    story_1 = [f + le, f, f + r, le, r, b + le, b, b + r]
    story_0 = [i + d for i in story_1]
    story_2 = [i + u for i in story_1]
    if self_contain:
        nbs_w_corners = (
            story_2 + [u] + story_1 + [np.asarray([0, 0, 0])] + story_0 + [d]
        )
    else:
        nbs_w_corners = story_2 + [u] + story_1 + story_0 + [d]
    return nbs_w_corners


def conditional_fill(array, condition="<", value=0.5):
    """returns new voxel_array with 0,1 values based on condition"""
    if condition == "<":
        mask_inv = array < value
    elif condition == ">":
        mask_inv = array > value
    elif condition == "<=":
        mask_inv = array <= value
    elif condition == ">=":
        mask_inv = array >= value
    a = np.zeros_like(array)
    a[mask_inv] = 0

    return a


def make_solid_box_z(grid_size, z_max):
    test_i = np.indices(grid_size)
    z = test_i[2, :, :, :] <= z_max
    d = np.zeros(grid_size)
    d[z] = 1
    return d


def make_solid_box_xxz(grid_size, x_min, x_max, z_max):
    test_i = np.indices(grid_size)
    x1 = test_i[0, :, :, :] >= x_min
    x2 = test_i[0, :, :, :] <= x_max
    z = test_i[2, :, :, :] <= z_max
    d = np.zeros(grid_size)
    d[x2 & x1 & z] = 1
    return d


def make_solid_box_xxyyzz(grid_size, x_min, x_max, y_min, y_max, z_min, z_max):
    """boolean box including limits"""
    test_i = np.indices(grid_size)
    x1 = test_i[0, :, :, :] >= x_min
    x2 = test_i[0, :, :, :] <= x_max
    y1 = test_i[1, :, :, :] >= y_min
    y2 = test_i[1, :, :, :] <= y_max
    z1 = test_i[2, :, :, :] >= z_min
    z2 = test_i[2, :, :, :] <= z_max
    d = np.zeros(grid_size)
    d[x2 & x1 & y1 & y2 & z1 & z2] = 1
    return d


def get_sub_array(array, offset_radius, center=None, format_values=None):
    """gets sub array around center, in 'offset_radius'
    format values: returns sum '0', avarage '1', or all_values: 'None'"""

    x, y, z = center
    n = offset_radius
    v = array[x - n : x + n][y - n : y + n][z - n : z - n]
    if format_values == 0:
        return np.sum(v)
    elif format_values == 1:
        return np.average(v)
    else:
        return v


def get_mask_zone_xxyyzz(
    grid_size: tuple[int, int, int],
    zone_xxyyzz: tuple[int, int, int, int, int, int],
    return_bool=True,
):
    """gets 3D boolean array within zone (including both end)
    return bool or int
    input:
        grid_size: tuple[i, j, k]
        zone_xxyyzz : [x_start, x_end, y_start, y_end, z_start, z_end]
        _bool_type: bool
    """
    # make sure params are in bounds
    zone_xxyyzz = clip_indices_to_grid_size(zone_xxyyzz, grid_size)

    x_min, x_max, y_min, y_max, z_min, z_max = zone_xxyyzz

    mask = np.zeros(grid_size, dtype=np.int32)
    mask[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1] = 1
    if return_bool:
        return mask.astype(np.bool_)
    return mask


def get_array_density_from_zone_xxyyzz(
    array,
    pose,
    relative_zone_xxyyzz: tuple[int, int, int, int, int, int],
    nonzero=False,
):
    """gets 3D boolean array within zone (including both end)
    return bool or int
    input:
        grid_size: tuple[i, j, k]
        zone_xxyyzz : [x_start, x_end, y_start, y_end, z_start, z_end]
        _bool_type: bool
    """
    # make sure params are in bounds
    shape = array.shape
    # grid_vol = array.size
    # print(f'shape {shape}, grid vol: {grid_vol}')
    x, y, z = pose
    x_min, x_max, y_min, y_max, z_min, z_max = relative_zone_xxyyzz
    zone_xxyyzz = [x_min + x, x_max + x, y_min + y, y_max + y, z_min + z, z_max + z]
    zone_xxyyzz = clip_indices_to_grid_size(zone_xxyyzz, shape)
    # print(zone_xxyyzz)
    x_min, x_max, y_min, y_max, z_min, z_max = zone_xxyyzz
    vol = (abs(x_min - x_max) + 1) * (abs(y_min - y_max) + 1) * (abs(z_min - z_max) + 1)
    # print("vol", vol)
    mask = np.zeros(shape, dtype=np.int32)
    mask[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1] = 1
    v = np.where(mask == 1, array, 0)
    print(f"v_sum {np.sum(v)}")
    # print('v', v)
    if not nonzero:
        d = np.sum(v) / vol
    else:
        n = np.count_nonzero(v)
        # print(f"n = {n}")
        # m = grid_vol - n
        d = n / vol
    # print(f"density:{d}")
    return d


def crop_array(arr, start=0, end=1):
    arr = np.minimum(arr, end)
    arr = np.maximum(arr, start)
    return arr


# ######################
# index_map_functions


def index_map_cube(radius, min_radius=None):
    """including max size, excluding min size"""

    d = np.int_(np.floor(radius * 2)) + 1
    # print(d)
    x, y, z = np.indices([d, d, d])
    r2 = radius
    r2x, r2y, r2z = [np.floor(r2)] * 3
    x, y, z = x - r2x, y - r2y, z - r2z
    index_map = np.array([x, y, z], dtype=np.int32)
    x, y, z = index_map
    # print(f'base map: {x,y,z}')
    abs_x, abs_y, abs_z = np.absolute(index_map)
    if not min_radius:
        min_radius = 0
    mask = np.logical_or(abs_x > min_radius, abs_z > min_radius)
    mask = np.logical_or(abs_y > min_radius, mask)
    print(mask)

    return index_map.transpose()[mask]


def index_map_box(box_size, box_min_size=None):
    """including max size, excluding min size"""
    radius = np.array(box_size)
    if not box_min_size:
        box_min_size = [0, 0, 0]
    min_radius = np.array(box_min_size)
    d = np.int_(np.floor(radius * 2)) + 1
    # print(d)
    x, y, z = np.indices(d)
    rx, ry, rz = np.floor(radius)
    x, y, z = [x - rx, y - ry, z - rz]
    index_map = np.array([x, y, z], dtype=np.int32)
    x, y, z = index_map
    # print(f'base map: {x,y,z}')
    abs_x, abs_y, abs_z = np.absolute(index_map)
    mask1 = np.logical_or(abs_x > min_radius[0], abs_z > min_radius[2])
    mask2 = np.logical_or(abs_y > min_radius[1], abs_z > min_radius[2])
    mask = np.logical_or(mask1, mask2)

    filtered_index_map = index_map[:, mask].transpose()
    print(mask.shape)
    index_map = np.array(index_map).transpose()

    return filtered_index_map


def index_map_sphere(radius: float, min_radius: float = None) -> np.ndarray[np.int32]:
    d = int(np.ceil(radius) * 2) + 1
    x, y, z = np.indices([d, d, d])
    r2 = np.ceil(radius)
    indices = [x - r2, y - r2, z - r2]
    norm = np.linalg.norm(indices, axis=0)
    # print(l)
    mask = norm <= radius
    if min_radius:
        mask2 = norm >= min_radius
        mask = np.logical_and(mask, mask2)
    # print(mask)
    # print(indices)
    x = indices[0][mask]
    y = indices[1][mask]
    z = indices[2][mask]
    sphere_array = np.array([x, y, z], dtype=np.int32)
    return sphere_array.transpose()


def index_map_cylinder(radius, height, min_radius=0, z_lift=0):
    d = int(np.ceil(radius) * 2) + 1
    x, y, z = np.indices([d, d, height])
    z += z_lift
    r2 = np.ceil(radius)
    x, y, z = [x - r2, y - r2, z]
    l1 = np.linalg.norm([x, y], axis=0)

    if min_radius > 0:
        mask = np.logical_and(l1 <= radius, l1 >= min_radius)
    else:
        mask = l1 <= radius

    # print(f"mask {mask}")
    indices = [x, y, z]
    x = indices[0][mask]
    y = indices[1][mask]
    z = indices[2][mask]
    index_map = np.array([x, y, z], dtype=np.int32)
    return index_map.transpose()


def index_map_sphere_scale_NU(
    radius=1.5,
    min_radius=None,
    scale_NU=(1, 1, 0.5),
):
    """returns index map"""
    original_radius = radius
    scale_NU = np.array(scale_NU)
    radius = radius * scale_NU
    scale_down = 1 / scale_NU

    d = np.int_(np.ceil(radius) * 2) + 1
    x, y, z = np.indices(d)

    r2 = np.ceil(radius)
    x2 = x - r2[0]
    y2 = y - r2[1]
    z2 = z - r2[2]
    sx, sy, sz = scale_down
    indices = np.array([x2 * sx, y2 * sy, z2 * sz])
    indices_real = np.array([x2, y2, z2])

    norm = np.linalg.norm(indices, axis=0)
    # print(l)
    if min_radius:
        mask = np.logical_and(min_radius < norm, norm <= original_radius)
    else:
        mask = norm <= original_radius
    # print(mask)

    x = indices_real[0][mask]
    y = indices_real[1][mask]
    z = indices_real[2][mask]

    arr = np.array([x, y, z], dtype=np.int32)
    return arr.transpose()


def index_map_move_and_clip(
    index_map_: np.ndarray,
    move_to: tuple[int, int, int] = [0, 0, 0],
    grid_size: tuple[int, int, int] = None,
):
    index_map = index_map_.copy()
    index_map += np.int_(move_to)
    if grid_size:
        index_map = np.clip(index_map, [0, 0, 0], grid_size - np.array([1, 1, 1]))
        index_map = np.unique(index_map, axis=0)
    return index_map


def clip_index_map(
    index_map_: np.ndarray,
    bounds: tuple[int, int, int] = None,
    erase_out_of_bound_indices: bool = False,
):
    index_map = index_map_.copy()
    index_map_clipped = np.clip(index_map, [0, 0, 0], bounds - np.array([1, 1, 1]))
    if erase_out_of_bound_indices:
        mask = index_map_clipped == index_map
        same = np.all(mask, axis=1)
        index_map_clipped = index_map[same]
    return index_map_clipped


def get_values_by_index_map(
    array, index_map_, origin, return_list=False, dtype=np.float64
):
    """
    Filters the values of a 3D NumPy array based on an index map.
        numpy.ndarray: A 1D array containing the filtered values.
    """
    # Convert the index_map to a tuple of lists, suitable for NumPy advanced indexing
    index_map = index_map_.copy()
    index_map += np.array(origin, dtype=np.int32)
    # array = np.array(array, dtype=np.int32)
    index_map = np.unique(
        np.clip(
            index_map, [0, 0, 0], array.shape - np.array([1, 1, 1]), dtype=np.int32
        ),
        axis=0,
    )
    # print(f"index_map in get_value function: {index_map}")
    indices = tuple(np.array(index_map).T)

    # Extract the elements using advanced indexing
    values = np.array(array[indices], dtype=dtype)
    # print(origin, values)
    # print(f"filtered values: {values}")

    if return_list:
        return values.tolist()
    else:
        return values


def set_value_by_index_map(array, index_map_, origin=None, value=1):
    """
    Filters the values of a 3D NumPy array based on an index map.
        numpy.ndarray: A 1D array containing the filtered values.
    """
    # Convert the index_map to a tuple of lists, suitable for NumPy advanced indexing
    index_map = index_map_.copy()
    if not origin:
        origin = [0, 0, 0]
    index_map += np.array(origin, dtype=np.int32)
    index_map = np.unique(
        (np.clip(index_map, [0, 0, 0], array.shape - np.array([1, 1, 1]))), axis=0
    )
    indices = tuple(np.array(index_map).T)
    # Extract the elements using  indexing
    array[indices] = value
    return array


def random_choice_index_from_best_n_old(list_array, n, print_=False):
    """double random choice to avoid finding only the index of the selected
    if tie"""
    array = list_array
    a = np.sort(array)
    best_of = a[(len(array) - n) :]
    random_choice_from_the_best_nth = np.random.choice(best_of)
    matching_i = np.argwhere(array == random_choice_from_the_best_nth).transpose()
    random_choice_index_from_best_n_ = np.random.choice(matching_i[0])
    # print('random value of the best [{}] value = {}, index = {}'.format(n, array[best_index], index_of_random_choice_of_best_n_value))  # noqa: E501
    if print_:
        print(f"""random selection. input array: {list_array}
        best options:{best_of}, choice value: {random_choice_from_the_best_nth}, 
        index:{random_choice_index_from_best_n_}""")

    return random_choice_index_from_best_n_


def random_choice_index_from_best_n(list_array, n):
    """add a random array with very small numbers to avoid only finding the
    index of the selected if equals"""
    random_sort = np.random.random(len(list_array)) * 1e-30
    array = list_array * random_sort
    a = np.sort(array)
    best_of = a[(len(array) - n) :]
    random_choice_from_the_best_nth = np.random.choice(best_of)
    matching_i = np.argwhere(array == random_choice_from_the_best_nth).transpose()
    random_choice_index_from_best_n_ = np.random.choice(matching_i[0])
    return random_choice_index_from_best_n_


def clip_indices_to_grid_size(
    index: npt.NDArray | tuple[int], grid_size: tuple[int, int, int]
):
    """Clips indices (i, j, k) to grid size"""
    return_nparray = isinstance(index, np.ndarray)

    index = np.asarray(index)

    if index.shape[-1] == 3:
        a_min = [0, 0, 0]
        a_max = np.array(grid_size) - 1
    elif index.shape[0] == 6:
        a_min = [0, 0, 0, 0, 0, 0]
        # a_max should be [maxi, maxi, maxj, maxj, maxk, maxk]
        a_max = (
            np.array(
                [
                    grid_size[0],
                    grid_size[0],
                    grid_size[1],
                    grid_size[1],
                    grid_size[2],
                    grid_size[2],
                ]
            )
            - 1
        )
    else:
        raise ValueError("Index shape must be 3 or 6")

    clipped = np.clip(index, a_min=a_min, a_max=a_max)

    if return_nparray:
        return clipped
    else:
        return clipped.tolist()


def distance_to_point(array: np.ndarray, point: tuple[float, float, float]):
    # index = np.clip(index, [0, 0, 0], np.array(index) - [1, 1, 1])
    indices = np.indices(array.shape)
    center = np.int_(point)
    x, y, z = indices
    d = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    return d**0.5


def roll_array(array, shift, axis):
    """Shifts the array along the specified axis."""
    if shift == 0:
        np.zeros_like(array)

    else:
        array = np.roll(array, shift, axis=axis)
    return array


def pad_array(array: np.ndarray, pad_width: int = 1, values=0):
    array = np.pad(
        array,
        [[pad_width, pad_width], [pad_width, pad_width], [pad_width, pad_width]],
        "constant",
        constant_values=values,
    )
    return array


def offset_array_from_corner(
    array: np.ndarray,
    corner: tuple[int, int, int],
    steps: int,
    clip_array: bool = False,
):
    """corner: index of corner, for example:
    [0,0,0]
    [0, -1, 0]
    [-1, -1, -1]"""
    pad = steps * 3
    padded_array = pad_array(array.copy(), pad, 0)

    for j in range(steps):
        for i in range(len(corner)):
            axis = i
            dir = corner[i]
            if dir == 0:
                dir = 1
            elif dir == -1:
                dir = -1
            else:
                # print("int values of corner should be 0 or -1")
                raise ValueError(corner)

            # print(f"axis:{axis}, dir: {dir}")

            shift = (j + 1) * dir
            # print(f"shift:{shift}")

            rolled_array = np.roll(padded_array, shift=shift, axis=axis)

            padded_array += rolled_array

    array = padded_array[pad:-pad, pad:-pad, pad:-pad]

    if clip_array:
        array = np.clip(array, 0, 1)
    return array


def extrude_array_from_point(
    array: np.ndarray,
    point: tuple[int, int, int],
    steps: int,
    clip_array: bool = True,
):
    shape = array.shape
    a, b, c = np.clip(point, [0, 0, 0], np.array(shape) - 1)
    # print(a, b, c)
    new_array = np.zeros_like(array)
    for _ in range(steps):
        for z in [-1, 0]:
            for i in range(4):
                y = -1 if i < 2 else 0
                x = -1 if i % 2 == 0 else 0
                corner = np.array([x, y, z])
                # print(f"corner:{corner}")
                sub_array = array.copy()
                if x == -1:
                    sub_array = sub_array[:a, :, :]
                    pad_width_x = [0, shape[0] - a]
                    # print(f"subarray:\n{sub_array}")
                elif x == 0:
                    sub_array = sub_array[a:, :, :]
                    pad_width_x = [a, 0]
                    # print(f"subarray:\n{sub_array}")
                if y == -1:
                    sub_array = sub_array[:, :b, :]
                    pad_width_y = [0, shape[1] - b]
                    # print(f"subarray:\n{sub_array}")
                elif y == 0:
                    sub_array = sub_array[:, b:, :]
                    pad_width_y = [b, 0]
                    # print(f"subarray:\n{sub_array}")
                if z == -1:
                    sub_array = sub_array[:, :, :c]
                    pad_width_z = [0, shape[2] - c]
                    # print(f"subarray:\n{sub_array}")
                elif z == 0:
                    sub_array = sub_array[:, :, c:]
                    pad_width_z = [c, 0]
                    # print(f"subarray:\n{sub_array}")
                # print(f"subarray:\n{sub_array}")
                if np.sum(sub_array) == 0:
                    pass
                else:
                    offsetted_sub_array = offset_array_from_corner(
                        sub_array, corner, 1, clip_array
                    )
                    # print(f"offsetted_sub_array\n{offsetted_sub_array}")
                    new_array += np.pad(
                        offsetted_sub_array,
                        [pad_width_x, pad_width_y, pad_width_z],
                        "constant",
                        constant_values=[[0, 0], [0, 0], [0, 0]],
                    )
    return new_array


def offset_array_radial(array_: np.ndarray, steps: int, clip_array: bool = True):
    array = array_.copy()
    array1 = extrude_array_in_direction_expanding(array, [1, 1, 1], steps)
    array1 += extrude_array_in_direction_expanding(array1, [-1, -1, -1], steps)
    if clip_array:
        array1 = np.clip(array1, 0, 1)
    return array1


def extrude_array_in_direction_expanding(
    array: np.ndarray,
    direction: tuple[int, int, int],
    length: int,
    clip_array: bool = True,
):
    """expanding, conical extrusion in int_.vector direction"""
    direction = np.clip(direction, [-1, -1, -1], [1, 1, 1])

    pad = length
    for _ in range(length):
        array = pad_array(array.copy(), pad, 0)
        for i in range(3):
            axis = i
            dir = direction[i]
            if dir == 0:
                continue
            rolled_array = np.roll(array, shift=dir, axis=axis)

            array += rolled_array
        array = array[pad:-pad, pad:-pad, pad:-pad]

    if clip_array:
        array = np.clip(array, 0, 1)

    return array


def extrude_array_linear(
    array: np.ndarray,
    direction: tuple[int, int, int],
    length: int,
    clip_array: bool = True,
):
    """
    extrude linear, limited to 45 degree directions
    direction is clipped to (-1, 1)
    """
    direction = np.clip(direction, [-1, -1, -1], [1, 1, 1])

    pad = length
    new_array = array.copy()
    for j in range(length):
        array_step = pad_array(array.copy(), pad, 0)
        for i in range(3):
            axis = i
            dir = direction[i]
            if dir == 0:
                continue
            shift = (j + 1) * dir
            array_step = np.roll(array_step, shift=shift, axis=axis)

        new_array += array_step[pad:-pad, pad:-pad, pad:-pad]

    if clip_array:
        new_array = np.clip(new_array, 0, 1)

    return new_array


def extrude_array_along_vector(
    array: np.ndarray,
    vector: tuple[float, float, float],
    length: int,
    clip_array: bool = True,
):
    """
    linear extrusion, direction angle is 'unlimited'
    direction = [1,5,-4] for example"""
    vector = np.array(vector, dtype=np.float64)
    u_vector = vector / np.linalg.norm(vector)
    index_directions = get_index_steps_along_vector(vector, length)
    n = len(index_directions)
    print(u_vector)
    pad = n
    padded_array = pad_array(array.copy(), pad, 0)
    new_array = padded_array.copy()
    for j in range(length):
        # print(f"index dir:{index_direction}")
        for i in range(3):
            axis = i
            shift = ceil(index_directions[j][i])
            if shift == 0:
                continue
            new_array += np.roll(padded_array, shift=shift, axis=axis)

    array = new_array[pad:-pad, pad:-pad, pad:-pad]

    if clip_array:
        array = np.clip(array, 0, 1)

    return array


def extrude_array_along_vector_b(
    array: np.ndarray,
    vector: tuple[float, float, float],
    length: int,
    clip_array: bool = True,
):
    """
    linear extrusion, direction angle is 'unlimited'
    direction = [1,5,-4] for example"""
    vector = np.array(vector, dtype=np.float64)
    u_vector = vector / np.linalg.norm(vector)
    print(u_vector)
    pad = 4
    # new_array = array.copy()
    for _j in range(length):
        # print(f"index dir:{index_direction}")
        array_step = pad_array(array.copy(), pad, 0)
        for i in range(3):
            axis = i
            shift = ceil(u_vector[i])
            if shift == 0:
                continue
            array_step = np.roll(array_step, shift=shift, axis=axis)

        array += array_step[pad:-pad, pad:-pad, pad:-pad]

    if clip_array:
        array = np.clip(array, 0, 1)

    return array


def get_index_steps_along_vector(
    vector: tuple[float, float, float],
    length: int,
):
    vector = np.array(vector, dtype=np.float64)
    v_hat = vector / np.linalg.norm(vector)
    index_steps = []
    for i in range(length):
        d = (i + 1) * v_hat
        # print(d)
        d = np.round(d)
        index_steps.append(d)
    return np.unique(np.array(index_steps, dtype=np.int32), axis=0)


def get_normal_vector(
    vector: tuple[float, float, float],
):
    vector = np.array(vector, dtype=np.float64)
    v_hat = vector / np.linalg.norm(vector)
    return v_hat


def mask_index_map_by_nonzero(
    array=None, origin: tuple[int, int, int] = None, sense_range_or_radius=None
):
    """return nonzero indices in location"""
    sense_map = (
        index_map_sphere(sense_range_or_radius)
        if isinstance(sense_range_or_radius, float | int)
        else sense_range_or_radius.copy()
    )
    sense_map_clipped = index_map_move_and_clip(sense_map, origin, array.shape)
    values = get_values_by_index_map(array, sense_map, origin, return_list=False)
    x = np.argwhere(values != 0)

    filled_surrounding_indices = sense_map_clipped[x.reshape([x.size])]
    return filled_surrounding_indices


def get_surrounding_offset_region(arrays, offset_thickness=1, exclude_arrays=None):
    """returns surrounding volumetric region of several volumes in given thickness"""
    arrays = np.array(arrays, np.int_)
    walk_on_array = np.clip(np.sum(arrays, axis=0), 0, 1)
    walk_on_array_offset = offset_array_radial(walk_on_array, offset_thickness)
    offset_region = walk_on_array_offset - walk_on_array
    if not exclude_arrays:
        arrays = np.array(arrays, np.int_)
        exclude_region = np.clip(np.sum(arrays, axis=0), 0, 1)
        offset_region -= exclude_region
    return offset_region
