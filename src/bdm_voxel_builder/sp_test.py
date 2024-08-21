import numpy as np

def get_spheric_indices(radius = 1.5):
    d = int(np.ceil(radius) * 2) + 1
    x,y,z = np.indices([d,d,d])
    r2 = np.ceil(radius)
    indices = [x - r2, y - r2, z - r2]
    l = np.linalg.norm(indices, axis=0)
    # print(l)
    mask = l <= radius
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


def get_cylinder_indices(radius = 1.5, h = 2):
    d = int(np.ceil(radius) * 2) + 1
    x,y,z = np.indices([d,d,h])
    r2 = np.ceil(radius)
    x,y,z = [x - r2, y - r2, z]
    l1 = np.linalg.norm([x,y], axis=0)
    print(l1)
    radius_c = l1 <= radius
    height_c = z < h
    mask = np.logical_and(radius_c, height_c)
    print(f'mask {mask}')
    indices = [x,y,z]
    x = indices[0][mask]
    y = indices[1][mask]
    z = indices[2][mask]
    sphere_array = np.array([x, y, z], dtype=np.int64)
    return sphere_array.transpose()

def get_cylinder_indices_limit(radius = 3, min_radius = 1, h = 2):
    d = int(np.ceil(radius) * 2) + 1
    x,y,z = np.indices([d,d,h])
    r2 = np.ceil(radius)
    x,y,z = [x - r2, y - r2, z]
    l1 = np.linalg.norm([x,y], axis=0)
    print(l1)
    radius_c = l1 <= radius
    radius_c_min = l1 >= min_radius
    height_c = z < h
    mask = np.logical_and(radius_c, height_c)
    mask = np.logical_and(mask, radius_c_min)
    print(f'mask {mask}')
    indices = [x,y,z]
    x = indices[0][mask]
    y = indices[1][mask]
    z = indices[2][mask]
    sphere_array = np.array([x, y, z], dtype=np.int64)
    return sphere_array.transpose()

# print(get_cylinder_indices_limit(2.2,1.9,1))


def get_spheric_indices_scale_NU(radius = 1.5, scale_NU = [1, 1, 0.5]):
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
    mask = l <= original_radius
    # print(mask)

    x = indices_real[0][mask]
    y = indices_real[1][mask]
    z = indices_real[2][mask]

    arr = np.array([x, y, z], dtype=np.int64)
    return arr.transpose()

# r = 2
# bullet = get_spheric_indices_scale_NU(r, [1.3,1,2.1])

# print(len(bullet))
# print(bullet.transpose())