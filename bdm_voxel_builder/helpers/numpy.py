import numpy as np
import numpy.typing as npt

from bdm_voxel_builder import TEMP_DIR
from bdm_voxel_builder.helpers.savepaths import get_savepath

NB_INDEX_DICT = {
    "up": np.asarray([0, 0, 1]),
    "left": np.asarray([-1, 0, 0]),
    "down": np.asarray([0, 0, -1]),
    "right": np.asarray([1, 0, 0]),
    "front": np.asarray([0, -1, 0]),
    "back": np.asarray([0, 1, 0]),
}


def _convert_array_to_pts_wo_data(arr: npt.NDArray) -> list[list[float]]:
    pts = []
    for i, j, k in zip(*np.nonzero(arr), strict=False):
        pts.append([i, j, k])
    return pts


def convert_array_to_pts(
    arr: npt.NDArray, get_data=True
) -> list[list[float] | npt.NDArray]:
    if not get_data:
        return _convert_array_to_pts_wo_data(arr)

    indicies = np.indices(arr.shape)
    pt_location = np.logical_not(arr == 0)

    coordinates = []
    for i in range(3):
        c = indicies[i][pt_location]
        coordinates.append(c)

    return np.vstack(coordinates).transpose()

def convert_pointcloud_to_grid_array(pointcloud, unit_in_mm = 10):
    pts = pointcloud.points
    coordinate_array = np.asarray(pts) # array = [[x,y,z][x,y,z][x,y,z]]
    index_array = np.floor_divide(coordinate_array, unit_in_mm)
    index_array = np.int64(index_array)
    # print(f'index array, floor divide:\n {index_array}')

    maximums = np.amax(index_array, axis = 0)
    minimums = np.amin(index_array, axis = 0)
    # print(f'max:{maximums}, mins:{minimums}')

    move_to_origin_vector = 0 - minimums
    index_array += move_to_origin_vector
    # print(f'index array, translated:\n {index_array}')

    bounds = np.int64(maximums - minimums)
    bounds += 1
    # print(f'grid_size {bounds}')

    grid_from_pointcloud = np.zeros(bounds)
    for point in index_array:
        x,y,z = point
        # print(f'coord: {x,y,z}')
        grid_from_pointcloud[x][y][z] = 1

    # print(f'grid_from_pointcloud=\n{grid_from_pointcloud}')
    return grid_from_pointcloud

def save_ndarray(arr: npt.NDArray, note: str = None):
    np.save(get_savepath(TEMP_DIR, ".npy", note=note), arr)


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
    x,y,z = pose
    x_min, x_max, y_min, y_max, z_min, z_max = relative_zone_xxyyzz
    zone_xxyyzz = [x_min + x, x_max + x, y_min + y, y_max + y, z_min + z, z_max + z]
    zone_xxyyzz = clip_indices_to_grid_size(zone_xxyyzz, shape)

    x_min, x_max, y_min, y_max, z_min, z_max = zone_xxyyzz
    vol = ((abs(x_min - x_max) + 1) * (abs(y_min - y_max) + 1) * (abs(z_min - z_max) + 1))
    print('vol', vol)
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
    print(f'density:{d}')
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
