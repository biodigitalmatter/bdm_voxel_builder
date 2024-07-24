import numpy as np
import numpy.typing as npt

from bdm_voxel_builder import TEMP_DIR
from bdm_voxel_builder.helpers.savepaths import get_savepath

NB_INDEX_DICT = {
    'up' : np.asarray([0,0,1]),
    'left' : np.asarray([-1,0,0]),
    'down' : np.asarray([0,0,-1]),
    'right' : np.asarray([1,0,0]),
    'front' : np.asarray([0,-1,0]),
    'back' : np.asarray([0,1,0])
}

def convert_array_to_points(a, list_output=False):
    indicies = np.indices(a.shape)
    pt_location = np.logical_not(a == 0)
    coordinates = []
    for i in range(3):
        c = indicies[i][pt_location]
        coordinates.append(c)
    if list_output:
        pts = np.vstack(coordinates).transpose().tolist()
    else:
        pts = np.vstack(coordinates).transpose()
    return pts


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
    paired_lists = list(zip(values, pts))

    # Sort the paired lists based on the first element of the pairs (values values)
    sorted_paired_lists = sorted(paired_lists, key=lambda x: x[0])

    # Extract the sorted nested list

    sortedpts = [element[1] for element in sorted_paired_lists]
    values = [element[0] * multiply for element in sorted_paired_lists]

    return sortedpts, values

def create_zero_array(n):
    # create voxel-like array
    a = np.zeros(n ** 3)  # 3 dimensional numpy array representing voxel voxel
    a = np.reshape(a, [n, n, n])
    return a

def create_random_array(n):
    # create voxel-like array
    a = np.random.random(n ** 3)
    a = np.reshape(a, [n, n, n])
    return a

def set_value_at_index(layer, index = [0,0,0], value = 1):
    # print('set value at index', index)
    i,j,k = index
    try:
        layer.array[i][j][k] = value
    except Exception as e:
        print('set value error:%s' %e)
    return layer

def get_cube_array_indices(self_contain = False):
    """26 nb indicies, ordered: top-middle-bottom"""
    # horizontal
    f = NB_INDEX_DICT['front']
    b = NB_INDEX_DICT['back']
    le = NB_INDEX_DICT['left']
    r = NB_INDEX_DICT['right']
    u = NB_INDEX_DICT['up']
    d = NB_INDEX_DICT['down']
    # first_story in level:
    story_1 = [ f + le, f, f + r, le, r, b + le, b, b + r]
    story_0 = [i + d for i in story_1]
    story_2 = [i + u for i in story_1]
    if self_contain:
        nbs_w_corners = story_2 + [u] + story_1 + [np.asarray([0,0,0])] + story_0 + [d]
    else:
        nbs_w_corners = story_2 + [u] + story_1 + story_0 + [d]
    return nbs_w_corners

def conditional_fill(array, n, condition = '<', value = 0.5, override_self = False):
    """returns new voxel_array with 0,1 values based on condition"""
    if condition == '<':
        mask_inv = array < value
    elif condition == '>':
        mask_inv = array > value
    elif condition == '<=':
        mask_inv = array <= value
    elif condition == '>=':
        mask_inv = array >=  value
    a = create_zero_array(n)
    a[mask_inv] = 0
    if override_self:
        array = a
    return a

def make_solid_box_z(voxel_size, z_max):
    n = voxel_size
    test_i = np.indices((n,n,n))
    z = test_i[2,:,:,:] <= z_max
    d = np.zeros((n,n,n))
    d[z] = 1
    return d

def make_solid_box_xxz(voxel_size, x_min, x_max, z_max):
    n = voxel_size
    test_i = np.indices((n,n,n))
    x1 = test_i[0,:,:,:] >= x_min
    x2 = test_i[0,:,:,:] <= x_max
    z = test_i[2,:,:,:] <= z_max
    d = np.zeros((n,n,n))
    d[x2 & x1 & z] = 1
    return d

def make_solid_box_xxyyzz(voxel_size, x_min, x_max, y_min, y_max, z_min, z_max):
    """boolean box including limits"""
    n = voxel_size
    test_i = np.indices((n,n,n))
    x1 = test_i[0,:,:,:] >= x_min
    x2 = test_i[0,:,:,:] <= x_max
    y1 = test_i[1,:,:,:] >= y_min
    y2 = test_i[1,:,:,:] <= y_max
    z1 = test_i[2,:,:,:] >= z_min
    z2 = test_i[2,:,:,:] <= z_max
    d = np.zeros((n,n,n))
    d[x2 & x1 & y1 & y2 & z1 & z2] = 1
    return d

def get_sub_array(array, offset_radius, center = None, format_values = None):
    """gets sub array around center, in 'offset_radius'
    format values: returns sum '0', avarage '1', or all_values: 'None'"""

    x,y,z = center
    n = offset_radius
    v = array[x - n : x + n][y - n : y + n][z - n : z - n]
    if format_values == 0:
        return np.sum(v)
    elif format_values == 1:
        return np.average(v)
    else: 
        return v