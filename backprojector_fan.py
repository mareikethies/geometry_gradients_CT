import torch
from numba import cuda
from helper import vox2world_cuda
from torch.autograd import Function
from helper import interpolate1d_cuda
from torch.autograd.function import once_differentiable


class DifferentiableFanBeamBackprojector(Function):
    @staticmethod
    def forward(ctx, sinogram, projection_matrices, geometry):
        ''' Performs 2d fan beam backprojection, saves detector sampling positions for backward. Geometry
        object does not need gradient.

        :param ctx: context to save variables for backward
        :param sinogram: Sinogram tensor, shape [num_angles x detector_size]
        :param projection_matrices: Matrix tensor, shape [num_angles x 2 x 3]
        :param geometry: Collector for all geometry parameters that have no derivative implemented
        :return:
        '''
        projection_matrices = projection_matrices.detach()

        reco = DifferentiableFanBeamBackprojector.call_forward_kernel(sinogram, projection_matrices, geometry)

        ctx.save_for_backward(sinogram, projection_matrices)
        ctx.geometry = geometry

        return reco

    @staticmethod
    def call_forward_kernel(sinogram, projection_matrices, geometry):
        blocks_per_grid = (32, 32, 32)
        threads_per_block = (4, 4, 8)
        gpu = torch.device('cuda')

        reco = cuda.as_cuda_array(torch.zeros((geometry.volume_shape[0], geometry.volume_shape[1]), device=gpu))
        DifferentiableFanBeamBackprojector.forward_loop[blocks_per_grid, threads_per_block](sinogram,
                                                                                            projection_matrices,
                                                                                            reco,
                                                                                            geometry.volume_shape,
                                                                                            geometry.volume_spacing,
                                                                                            geometry.volume_origin)
        reco = torch.as_tensor(reco, device=gpu)

        return reco

    @staticmethod
    @cuda.jit
    def forward_loop(sinogram, projection_matrices, reco, volume_shape, volume_spacing, volume_origin):
        start_x, start_y, start_a = cuda.grid(3)
        stride_x, stride_y, stride_a = cuda.gridsize(3)

        for x in range(start_x, volume_shape[0], stride_x):
            for y in range(start_y, volume_shape[1], stride_y):
                for a in range(start_a, sinogram.shape[0], stride_a):
                    # convert image pixel to world coordinates
                    point1 = vox2world_cuda(x, volume_spacing[0], volume_origin[0])
                    point0 = vox2world_cuda(y, volume_spacing[1], volume_origin[1])
                    point2 = 1.
                    # forward project using projection matrix
                    p = projection_matrices[a, :, :]
                    u = p[0, 0] * point0 + p[0, 1] * point1 + p[0, 2] * point2
                    v = p[1, 0] * point0 + p[1, 1] * point1 + p[1, 2] * point2
                    # sample sinogram at position (a, u/v)
                    sino = sinogram[a, :]
                    # add to reco image
                    cuda.atomic.add(reco, (x, y), interpolate1d_cuda(sino, u / v))

    @staticmethod
    @once_differentiable
    def backward(ctx, volume_error):
        sinogram, projection_matrices = ctx.saved_tensors
        geometry = ctx.geometry

        proj_matrix_error = torch.zeros((sinogram.shape[0], 2, 3), device='cuda')
        sinogram_derived = torch.gradient(sinogram, dim=1)[0]

        DifferentiableFanBeamBackprojector.call_backward_kernel(geometry.volume_shape,
                                                                geometry.volume_spacing,
                                                                geometry.volume_origin,
                                                                projection_matrices,
                                                                sinogram_derived,
                                                                proj_matrix_error,
                                                                volume_error)

        return None, proj_matrix_error, None

    @staticmethod
    def call_backward_kernel(volume_shape, volume_spacing, volume_origin, projection_matrices, sinogram_derived,
                             proj_matrix_error, volume_error):
        blocks_per_grid = (32, 32, 32)
        threads_per_block = (4, 4, 8)

        DifferentiableFanBeamBackprojector.backward_loop[blocks_per_grid, threads_per_block](volume_shape,
                                                                                             volume_spacing,
                                                                                             volume_origin,
                                                                                             projection_matrices,
                                                                                             sinogram_derived,
                                                                                             proj_matrix_error,
                                                                                             volume_error)

    @staticmethod
    @cuda.jit
    def backward_loop(volume_shape, volume_spacing, volume_origin, projection_matrices, sinogram_derived,
                      proj_matrix_error, volume_error):
        start_x, start_y, start_a = cuda.grid(3)
        stride_x, stride_y, stride_a = cuda.gridsize(3)

        for a in range(start_a, sinogram_derived.shape[0], stride_a):
            for x in range(start_x, volume_shape[0], stride_x):
                for y in range(start_y, volume_shape[1], stride_y):
                    # convert image pixel to world coordinates
                    point1 = vox2world_cuda(x, volume_spacing[0], volume_origin[0])
                    point0 = vox2world_cuda(y, volume_spacing[1], volume_origin[1])
                    point2 = 1.
                    p = projection_matrices[a, :, :]
                    u = p[0, 0] * point0 + p[0, 1] * point1 + p[0, 2] * point2
                    v = p[1, 0] * point0 + p[1, 1] * point1 + p[1, 2] * point2
                    sino_derived = sinogram_derived[a, :]
                    grad_proj_wrt_pos = interpolate1d_cuda(sino_derived, u / v)

                    vol_error = volume_error[x, y]

                    grad_pos_wrt_00 = point0 / v
                    grad_pos_wrt_01 = point1 / v
                    grad_pos_wrt_02 = point2 / v
                    grad_pos_wrt_10 = - point0 * u / (v**2)
                    grad_pos_wrt_11 = - point1 * u / (v**2)
                    grad_pos_wrt_12 = - point2 * u / (v**2)

                    grad_proj_wrt_00 = grad_proj_wrt_pos * grad_pos_wrt_00
                    grad_proj_wrt_01 = grad_proj_wrt_pos * grad_pos_wrt_01
                    grad_proj_wrt_02 = grad_proj_wrt_pos * grad_pos_wrt_02
                    grad_proj_wrt_10 = grad_proj_wrt_pos * grad_pos_wrt_10
                    grad_proj_wrt_11 = grad_proj_wrt_pos * grad_pos_wrt_11
                    grad_proj_wrt_12 = grad_proj_wrt_pos * grad_pos_wrt_12

                    cuda.atomic.add(proj_matrix_error, (a, 0, 0), grad_proj_wrt_00 * vol_error)
                    cuda.atomic.add(proj_matrix_error, (a, 0, 1), grad_proj_wrt_01 * vol_error)
                    cuda.atomic.add(proj_matrix_error, (a, 0, 2), grad_proj_wrt_02 * vol_error)
                    cuda.atomic.add(proj_matrix_error, (a, 1, 0), grad_proj_wrt_10 * vol_error)
                    cuda.atomic.add(proj_matrix_error, (a, 1, 1), grad_proj_wrt_11 * vol_error)
                    cuda.atomic.add(proj_matrix_error, (a, 1, 2), grad_proj_wrt_12 * vol_error)
