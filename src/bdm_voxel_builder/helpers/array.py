from math import ceil

import compas.geometry as cg
import numpy as np
import numpy.typing as npt


def sort_pts_by_values(arr: npt.NDArray, multiply=1):
    """returns sorted points, values"""
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

    sorted_pts = [element[1] for element in sorted_paired_lists]
    values = [element[0] * multiply for element in sorted_paired_lists]

    return sorted_pts, values


def create_random_array(shape: int | tuple[int]):
    if isinstance(shape, int):
        shape = [shape] * 3

    return np.random.default_rng().random(shape)


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
    format values: returns sum '0', average '1', or all_values: 'None'"""

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
    zone_xxyyzz: list[int, int, int, int, int, int],
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


def get_array_average_using_index_map(
    array: npt.NDArray,
    map_,
    origin=(0, 0, 0),
    clipping_box: cg.Box = None,
    nonzero=False,
):
    """return clay density"""
    values = get_values_using_index_map(array, map_, origin, clipping_box)

    if nonzero:
        return np.count_nonzero(values) / len(values)
    else:
        return np.sum(values) / len(values)


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


def index_map_box_xxyyzz(xxyyzz: tuple[int, int, int, int, int, int]):
    x_min, x_max, y_min, y_max, z_min, z_max = xxyyzz

    x = np.arange(x_min, x_max - 1)
    y = np.arange(y_min, y_max - 1)
    z = np.arange(z_min, z_max - 1)

    x, y, z = np.meshgrid(x, y, z, indexing="ij")

    return np.array([x.flatten(), y.flatten(), z.flatten()]).transpose()


def index_map_sphere(radius: float, min_radius: float | None = None):
    """
    The index_map_sphere function generates a 3D array of indices that represent
    the coordinates within a sphere of a given radius. Optionally, it can also
    filter out indices that fall within a minimum radius, creating a hollow
    sphere effect.
    """
    d = int(np.ceil(radius) * 2) + 1
    x, y, z = xyz = np.indices([d, d, d])
    r2 = np.ceil(radius)
    indices = xyz - r2
    norm_indices = np.linalg.norm(indices, axis=0)

    mask = norm_indices <= radius
    if min_radius:
        mask2 = norm_indices >= min_radius
        mask = np.logical_and(mask, mask2)

    x = indices[0][mask]
    y = indices[1][mask]
    z = indices[2][mask]
    sphere_array = np.array([x, y, z], dtype=np.int_).transpose()
    return sphere_array


def index_map_cylinder(radius, height, min_radius=0, z_lift=0):
    d = int(np.ceil(radius) * 2) + 1
    x, y, z = np.indices([d, d, height])
    z += z_lift
    r2 = np.ceil(radius)
    x, y, z = [x - r2, y - r2, z]
    xy_norm = np.linalg.norm([x, y], axis=0)
    # print(l1)

    radius_condition = xy_norm <= radius

    if min_radius:
        radius_min_condition = xy_norm >= min_radius
        mask = np.logical_and(radius_condition, radius_min_condition)
    else:
        mask = xy_norm <= radius

    indices = [x, y, z]
    x = indices[0][mask]
    y = indices[1][mask]
    z = indices[2][mask]

    index_map = np.array([x, y, z], dtype=np.int_)

    return index_map.transpose()


def index_map_sphere_half_quarter(
    radius: float, min_radius: float = None
) -> np.ndarray[np.int32]:
    """half quarter sphere mask, center in [0,0,0]"""
    r = int(np.ceil(radius)) + 1
    indices = np.indices([r, r, r])
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


# TODO
def index_map_cylinder_quarter(radius, height, min_radius=0, z_lift=0):
    """quarter cylinder mask, center in [0,0,0]"""
    r = int(np.ceil(radius)) + 1
    x, y, z = np.indices([r, r, height])
    z += z_lift
    distance = np.linalg.norm([x, y], axis=0)

    if min_radius > 0:
        mask = np.logical_and(distance <= radius, distance >= min_radius)
    else:
        mask = distance <= radius

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


def clip_index_map_to_box(
    index_map: np.ndarray[np.int_], bbox: cg.Box | tuple[tuple[float]]
) -> np.ndarray:
    if not isinstance(bbox, cg.Box):
        a_min, a_max = bbox
    else:
        a_min, a_max = bbox.diagonal

    a_min = np.floor(a_min).astype(np.int_)
    a_max = np.floor(a_max).astype(np.int_) - 1

    new_index_map = index_map.clip(a_min, a_max)

    return np.unique(new_index_map, axis=0)


def get_localized_index_map(
    index_map: npt.NDArray,
    origin: tuple[int, int, int],
    clipping_box: cg.Box | tuple[tuple[float]] | None = None,
):
    localized_map = index_map + np.array(origin, dtype=np.int_)
    if clipping_box:
        localized_map = clip_index_map_to_box(localized_map, clipping_box)
    return localized_map


def get_values_using_index_map(
    array: np.ndarray,
    index_map: npt.NDArray[np.int_],
    origin: tuple[int, int, int] | None = None,
    clipping_box: cg.Box | None = None,
):
    if len(index_map) == 0:
        raise ValueError("Index map is empty")

    if origin is None:
        origin = (0, 0, 0)

    clipping_box = clipping_box or cg.Box.from_diagonal(((0, 0, 0), (array.shape)))
    index_map = get_localized_index_map(index_map, origin, clipping_box=clipping_box)

    return array[tuple(index_map.transpose())]


def set_value_using_index_map(
    array: np.ndarray,
    map_: npt.NDArray[np.int_],
    origin: tuple[int, int, int] = (0, 0, 0),
    clipping_box: cg.Box | list[tuple[float]] | None = None,
    value=1.0,
):
    localized_map = get_localized_index_map(map_, origin, clipping_box=clipping_box)

    array[localized_map] = value
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
    return_array = isinstance(index, np.ndarray)

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

    if return_array:
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
        array.clip(0, 1, out=array)

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
        new_array.clip(0, 1, out=new_array)

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
        array.clip(0, 1, out=array)

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
        array.clip(0, 1, out=array)

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
    array: npt.NDArray,
    origin: tuple[int, int, int] = (0, 0, 0),
    sense_range_or_radius: float | npt.NDArray[np.int_] | None = None,
    clipping_box: cg.Box | None = None,
):
    """return nonzero indices in location"""
    sense_map = (
        index_map_sphere(sense_range_or_radius)
        if isinstance(sense_range_or_radius, float | int)
        else sense_range_or_radius.copy()
    )
    values = get_values_using_index_map(
        array, sense_map, origin, clipping_box=clipping_box
    )

    x = np.argwhere(values != 0)

    localized_sense_map = get_localized_index_map(
        sense_map, origin, clipping_box=clipping_box
    )
    filled_surrounding_indices = localized_sense_map[x.reshape([x.size])]
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


def count_neighbors(array, int_=True):
    array = np.clip(array(array, dtype=np.int_), 0, 1)

    padded_array = np.pad(array, pad_width=1, mode="constant", constant_values=0)
    summed_array = padded_array.copy()

    shifts = [-1, 1, -1, 1, -1, 1]
    axii = [0, 0, 1, 1, 2, 2]
    for shift, axis in zip(shifts, axii):  # noqa: B905
        rolled_array = np.roll(padded_array, shift, axis)
        summed_array += rolled_array
    neighbor_count_array = summed_array[1:-1, 1:-1, 1:-1]
    return neighbor_count_array
