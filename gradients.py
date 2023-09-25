from numba import cuda
from helper import interpolate1d_cuda


@cuda.jit
def compute_grad_proj_wrt_pos_cuda(x, y, a, sinogram_derived, sampled_detector_positions):
    u = sampled_detector_positions[x, y, a]
    sino_derived = sinogram_derived[a, :]
    grad_proj_wrt_pos = interpolate1d_cuda(sino_derived, u)

    return grad_proj_wrt_pos


@cuda.jit
def compute_gradient_pos_wrt_angle_cuda(px, py, s, c, source_detector_distance, source_isocenter_distance):
    nom = px ** 2 + py ** 2 + c * px * source_isocenter_distance + s * py * source_isocenter_distance
    denom = (c * px + s * py + source_isocenter_distance) ** 2
    grad_pos_wrt_angle = nom / denom
    grad_pos_wrt_angle *= -source_detector_distance

    return grad_pos_wrt_angle


@cuda.jit
def compute_gradient_pos_wrt_dsd_cuda(px, py, s, c, source_isocenter_distance):
    nom = -s * px + c * py
    denom = c * px + s * py + source_isocenter_distance
    grad_pos_wrt_dsd = nom / denom

    return grad_pos_wrt_dsd


@cuda.jit
def compute_gradient_pos_wrt_dsi_cuda(px, py, s, c, source_detector_distance, source_isocenter_distance):
    nom = s * px - c * py
    denom = (c * px + s * py + source_isocenter_distance) ** 2
    grad_pos_wrt_dsi = nom / denom
    grad_pos_wrt_dsi *= source_detector_distance

    return grad_pos_wrt_dsi