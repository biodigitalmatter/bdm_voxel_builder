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
    indicies = np.indices(arr.shape)
    pt_location = np.logical_not(arr == 0)
    coordinates = []
    for i in range(3):
        c = indicies[i][pt_location]
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
    """26 nb indicies, ordered: top-middle-bottom"""
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

    mask = np.zeros(grid_size, dtype=np.int8)
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
    grid_vol = array.size
    # print(f'shape {shape}, grid vol: {grid_vol}')
    x, y, z = pose
    x_min, x_max, y_min, y_max, z_min, z_max = relative_zone_xxyyzz
    zone_xxyyzz = [x_min + x, x_max + x, y_min + y, y_max + y, z_min + z, z_max + z]
    zone_xxyyzz = clip_indices_to_grid_size(zone_xxyyzz, shape)

    x_min, x_max, y_min, y_max, z_min, z_max = zone_xxyyzz
    vol = (abs(x_min - x_max) + 1) * (abs(y_min - y_max) + 1) * (abs(z_min - z_max) + 1)
    print("vol", vol)
    mask = np.zeros(shape, dtype=np.int8)
    mask[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1] = 1
    v = np.where(mask == 1, array, 0)
    # print('v', v)
    if not nonzero:
        d = np.sum(v) / vol
    else:
        n = np.count_nonzero(v)
        print(f"n = {n}")
        m = grid_vol - n
        d = m / vol
    print(f"density:{d}")
    return d


def crop_array(arr, start=0, end=1):
    arr = np.minimum(arr, end)
    arr = np.maximum(arr, start)
    return arr


def random_choice_index_from_best_n_old(list_array, n, print_=False):
    """double random choice to to avoid only finding the index of the selected
    if equals"""
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


def random_choice_index_from_best_n(list_array, n, print_=False):
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
            # print(
            #     # f"rolled_array {rolled_array[steps:-steps, steps:-steps, steps:-steps]}"
            # )
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


def offset_array_radial(array: np.ndarray, steps: int, clip_array: bool = True):
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
    index_steps = get_index_steps_along_vector(vector, length)
    pad = length
    new_array = array.copy()
    for j in range(len(index_steps)):
        index_direction = index_steps[j]
        print(f"index dir:{index_direction}")
        array_step = pad_array(array.copy(), pad, 0)
        for i in range(3):
            axis = i
            shift = index_direction[i]
            if shift == 0:
                continue
            array_step = np.roll(array_step, shift=shift, axis=axis)

        new_array += array_step[pad:-pad, pad:-pad, pad:-pad]

    if clip_array:
        new_array = np.clip(new_array, 0, 1)

    return new_array


def get_index_steps_along_vector(
    vector: tuple[float, float, float],
    length: int,
):
    vector = np.array(vector, dtype=np.float64)
    v_hat = vector / np.linalg.norm(vector)
    index_steps = []
    for i in range(length):
        d = (i + 1) * v_hat
        print(d)
        d = np.round(d)
        index_steps.append(d)
    return np.unique(np.array(index_steps, dtype=np.int64), axis=0)


# index_steps = get_index_steps_along_vector(vector, length)
# print(index_steps)


# # """test offset_array_from_corner"""
# # array = np.zeros([6, 6, 6])
# # array[2, 2, 2] = 1

# # print(f"original array: \n{array}")
# # offset = 100
# # index = [2, 3, 4]

# # array = offset_array_from_index(array, index, offset, clip_array=True)

# # print(f"offsetted_array: \n{array}")

# # print(f"offsetted_array sum: \n{np.sum(array)}")

# """test extrude_array_linear"""
# array = np.zeros([7, 7, 7])
# # array[2:3, 2:4, 2:4] = 1
# array[2, 2, 2] = 1

# print(f"original array: \n{array}")
# length = 3
# vector = [0, 1, 0]

# array = extrude_array_along_vector(array, vector, length)

# array = offset_array_radial(array, 1)

# print(f"extruded_array: \n{array}")
