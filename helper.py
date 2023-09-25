from numba import cuda
import numpy as np


@cuda.jit
def vox2world_cuda(voxel_index, spacing, origin):
    return voxel_index * spacing + origin


@cuda.jit
def world2vox_cuda(world_index, spacing, origin):
    return (world_index - origin) / spacing


@cuda.jit
def interpolate1d_cuda(array, pos):
    pos_floor = int(pos)
    pos_floor_plus_one = pos_floor + 1
    delta = pos - pos_floor

    # Check boundary conditions
    if pos_floor < 0 or pos_floor > len(array) - 1:
        pos_floor = None
    if pos_floor_plus_one < 0 or pos_floor_plus_one > len(array) - 1:
        pos_floor_plus_one = None

    # Get function values of the data points
    a = 0.0
    b = 0.0
    if pos_floor is not None:
        a = array[int(pos_floor)]
    if pos_floor_plus_one is not None:
        b = array[int(pos_floor_plus_one)]

    return delta * b + (1 - delta) * a


@cuda.jit
def interpolate2d_cuda(array, pos_x, pos_y):
    pos_x_floor = int(pos_x)
    pos_x_floor_plus_one = pos_x_floor + 1
    delta_x = pos_x - pos_x_floor
    pos_y_floor = int(pos_y)
    pos_y_floor_plus_one = pos_y_floor + 1
    delta_y = pos_y - pos_y_floor

    # Check boundary conditions
    if pos_x_floor < 0 or pos_x_floor > array.shape[0] - 1:
        pos_x_floor = None
    if pos_x_floor_plus_one < 0 or pos_x_floor_plus_one > array.shape[0] - 1:
        pos_x_floor_plus_one = None
    if pos_y_floor < 0 or pos_y_floor > array.shape[1] - 1:
        pos_y_floor = None
    if pos_y_floor_plus_one < 0 or pos_y_floor_plus_one > array.shape[1] - 1:
        pos_y_floor_plus_one = None

    # Get function values of the data points
    a = 0.0
    b = 0.0
    c = 0.0
    d = 0.0
    if pos_x_floor is not None and pos_y_floor is not None:
        a = array[int(pos_x_floor), int(pos_y_floor)]
    if pos_x_floor_plus_one is not None and pos_y_floor is not None:
        b = array[int(pos_x_floor_plus_one), int(pos_y_floor)]
    if pos_x_floor is not None and pos_y_floor_plus_one is not None:
        c = array[int(pos_x_floor), int(pos_y_floor_plus_one)]
    if pos_x_floor_plus_one is not None and pos_y_floor_plus_one is not None:
        d = array[int(pos_x_floor_plus_one), int(pos_y_floor_plus_one)]

    tmp1 = delta_x * b + (1 - delta_x) * a
    tmp2 = delta_x * d + (1 - delta_x) * c

    return delta_y * tmp2 + (1 - delta_y) * tmp1


def params_2_proj_matrix(angles, dsd, dsi, tx, ty, det_spacing, det_origin):
    ''' compute fan beam projection matrices from parameters for circular trajectory

    :param angles: projection angles in radians
    :param dsd: source to detector distance
    :param dsi: source to isocenter distance
    :param tx: additional detector offset in x (usually 0 for motion free, ideal trajectory)
    :param ty: additional detector offset in y (usually 0 for motion free, ideal trajectory)
    :param det_spacing: detector pixel size
    :param det_origin: attention!! this is (-detector_origin / detector_spacing) or simply (image_size - 0.5)!!
    :return:
    '''
    num_angles = len(angles)
    matrices = np.zeros((num_angles, 2, 3))
    matrices[:, 0, 0] = -dsd * np.sin(angles) / det_spacing + det_origin * np.cos(angles)
    matrices[:, 0, 1] = dsd * np.cos(angles) / det_spacing + det_origin * np.sin(angles)
    matrices[:, 0, 2] = dsd * tx / det_spacing + det_origin * (dsi + ty)
    matrices[:, 1, 0] = np.cos(angles)
    matrices[:, 1, 1] = np.sin(angles)
    matrices[:, 1, 2] = dsi + ty

    intrinsics = np.zeros((num_angles, 2, 2))
    intrinsics[:, 0, 0] = dsd / det_spacing
    intrinsics[:, 0, 1] = det_origin
    intrinsics[:, 1, 1] = 1.

    extrinsics = np.zeros((num_angles, 2, 3))
    extrinsics[:, 0, 0] = - np.sin(angles)
    extrinsics[:, 0, 1] = np.cos(angles)
    extrinsics[:, 0, 2] = tx
    extrinsics[:, 1, 0] = np.cos(angles)
    extrinsics[:, 1, 1] = np.sin(angles)
    extrinsics[:, 1, 2] = dsi + ty

    assert np.allclose(matrices, np.einsum('aij,ajk->aik', intrinsics, extrinsics))

    # normalize w.r.t. lower right entry
    matrices = matrices / matrices[:, 1, 2][:, np.newaxis, np.newaxis]

    return matrices, extrinsics, intrinsics
