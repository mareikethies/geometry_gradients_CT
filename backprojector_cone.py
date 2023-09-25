import torch
from numba import cuda
from helper import vox2world_cuda
from torch.autograd import Function
from helper import interpolate2d_cuda
from torch.autograd.function import once_differentiable


class DifferentiableConeBeamBackprojector(Function):

    @staticmethod
    def forward(ctx, sinogram, projection_matrices, geometry):
        ''' Performs 2d fan beam backprojection, saves detector sampling positions for backward. Geometry
        object does not need gradient.

        :param ctx: context to save variables for backward
        :param sinogram: Sinogram tensor, shape [num_angles x detector_size_x x detector_size_y]
        :param projection_matrices: Matrix tensor, shape [num_angles x 3 x 4]
        :param geometry: Collector for all geometry parameters that have no derivative implemented
        :return:
        '''
        projection_matrices = projection_matrices.detach()

        reco = DifferentiableConeBeamBackprojector.call_forward_kernel(sinogram, projection_matrices, geometry)

        ctx.save_for_backward(sinogram, projection_matrices)
        ctx.geometry = geometry

        return reco

    @staticmethod
    def call_forward_kernel(sinogram, projection_matrices, geometry):
        blocks_per_grid = (32, 32, 32)
        threads_per_block = (4, 4, 4)
        gpu = torch.device('cuda')

        reco = cuda.as_cuda_array(torch.zeros(geometry.volume_shape, device=gpu))
        DifferentiableConeBeamBackprojector.forward_loop[blocks_per_grid, threads_per_block](sinogram,
                                                                                             projection_matrices,
                                                                                             reco,
                                                                                             geometry.volume_shape,
                                                                                             geometry.volume_spacing,
                                                                                             geometry.volume_origin)
        reco = torch.as_tensor(reco, device=gpu)

        return reco

    @staticmethod
    @cuda.jit
    def forward_loop(sinogram, projection_matrices, reco, volume_shape, volume_spacing,
                     volume_origin):
        start_x, start_y, start_z = cuda.grid(3)
        stride_x, stride_y, stride_z = cuda.gridsize(3)

        for x in range(start_x, volume_shape[0], stride_x):
            for y in range(start_y, volume_shape[1], stride_y):
                for z in range(start_z, volume_shape[2], stride_z):
                    for a in range(sinogram.shape[0]):
                        # convert image pixel to world coordinates
                        point2 = vox2world_cuda(x, volume_spacing[0], volume_origin[0])
                        point1 = vox2world_cuda(y, volume_spacing[1], volume_origin[1])
                        point0 = vox2world_cuda(z, volume_spacing[2], volume_origin[2])
                        # forward project using projection matrix
                        p = projection_matrices[a, :, :]
                        u = p[0, 0] * point0 + p[0, 1] * point1 + p[0, 2] * point2 + p[0, 3]
                        v = p[1, 0] * point0 + p[1, 1] * point1 + p[1, 2] * point2 + p[1, 3]
                        w = p[2, 0] * point0 + p[2, 1] * point1 + p[2, 2] * point2 + p[2, 3]
                        # sample sinogram at position (a, u/w, v/w)
                        sino = sinogram[a, :, :]
                        # add to reco image
                        cuda.atomic.add(reco, (x, y, z), interpolate2d_cuda(sino, v / w, u / w))

    @staticmethod
    @once_differentiable
    def backward(ctx, volume_error):
        sinogram, projection_matrices = ctx.saved_tensors
        geometry = ctx.geometry

        # cannot do full matrix multiplication of volume error and jacobi matrix because of memory constraints
        # instead multiply parts of Jacobi (tmp) and volume error on-the-fly in kernel and sum up for each angle
        # immediately
        proj_matrix_error = torch.zeros((sinogram.shape[0], 3, 4), device='cuda')
        sinogram_derived_x, sinogram_derived_y = torch.gradient(sinogram, dim=(2, 1))

        DifferentiableConeBeamBackprojector.call_backward_kernel(geometry.volume_shape,
                                                                 geometry.volume_spacing,
                                                                 geometry.volume_origin,
                                                                 projection_matrices,
                                                                 sinogram_derived_x,
                                                                 sinogram_derived_y,
                                                                 proj_matrix_error,
                                                                 volume_error)

        return None, proj_matrix_error, None

    @staticmethod
    def call_backward_kernel(volume_shape, volume_spacing, volume_origin, projection_matrices, sinogram_derived_x,
                             sinogram_derived_y, proj_matrix_error, volume_error):
        blocks_per_grid = (32, 32, 32)
        threads_per_block = (4, 4, 4)

        # kernel runs only over volume dimensions now; loop over angles needs to be separate to perform the summation
        # immediately in each iteration; this saves a lot of memory
        for a in range(sinogram_derived_x.shape[0]):
            DifferentiableConeBeamBackprojector.backward_loop[blocks_per_grid, threads_per_block](volume_shape,
                                                                                                  volume_spacing,
                                                                                                  volume_origin,
                                                                                                  projection_matrices[a, :, :],
                                                                                                  sinogram_derived_x[a, :, :],
                                                                                                  sinogram_derived_y[a, :, :],
                                                                                                  volume_error,
                                                                                                  proj_matrix_error,
                                                                                                  a)

    @staticmethod
    @cuda.jit
    def backward_loop(volume_shape, volume_spacing, volume_origin, projection_matrices, sinogram_derived_x,
                      sinogram_derived_y, volume_error, proj_matrix_error, angle_index):
        start_x, start_y, start_z = cuda.grid(3)
        stride_x, stride_y, stride_z = cuda.gridsize(3)

        for x in range(start_x, volume_shape[0], stride_x):
            for y in range(start_y, volume_shape[1], stride_y):
                for z in range(start_z, volume_shape[2], stride_z):
                    # convert image pixel to world coordinates
                    point2 = vox2world_cuda(x, volume_spacing[0], volume_origin[0])
                    point1 = vox2world_cuda(y, volume_spacing[1], volume_origin[1])
                    point0 = vox2world_cuda(z, volume_spacing[2], volume_origin[2])
                    # forward project using projection matrix
                    u = projection_matrices[0, 0] * point0 + projection_matrices[0, 1] * point1 + projection_matrices[0, 2] * point2 + projection_matrices[0, 3]
                    v = projection_matrices[1, 0] * point0 + projection_matrices[1, 1] * point1 + projection_matrices[1, 2] * point2 + projection_matrices[1, 3]
                    w = projection_matrices[2, 0] * point0 + projection_matrices[2, 1] * point1 + projection_matrices[2, 2] * point2 + projection_matrices[2, 3]

                    grad_proj_wrt_pos_x = interpolate2d_cuda(sinogram_derived_x, v / w, u / w)
                    grad_proj_wrt_pos_y = interpolate2d_cuda(sinogram_derived_y, v / w, u / w)

                    vol_error = volume_error[x, y, z]

                    cuda.atomic.add(proj_matrix_error, (angle_index, 0, 0), grad_proj_wrt_pos_x * point0 / w * vol_error)
                    cuda.atomic.add(proj_matrix_error, (angle_index, 0, 1), grad_proj_wrt_pos_x * point1 / w * vol_error)
                    cuda.atomic.add(proj_matrix_error, (angle_index, 0, 2), grad_proj_wrt_pos_x * point2 / w * vol_error)
                    cuda.atomic.add(proj_matrix_error, (angle_index, 0, 3), grad_proj_wrt_pos_x / w * vol_error)
                    cuda.atomic.add(proj_matrix_error, (angle_index, 1, 0), grad_proj_wrt_pos_y * point0 / w * vol_error)
                    cuda.atomic.add(proj_matrix_error, (angle_index, 1, 1), grad_proj_wrt_pos_y * point1 / w * vol_error)
                    cuda.atomic.add(proj_matrix_error, (angle_index, 1, 2), grad_proj_wrt_pos_y * point2 / w * vol_error)
                    cuda.atomic.add(proj_matrix_error, (angle_index, 1, 3), grad_proj_wrt_pos_y / w * vol_error)
                    cuda.atomic.add(proj_matrix_error, (angle_index, 2, 0), -(grad_proj_wrt_pos_x * point0 * u + grad_proj_wrt_pos_y * point0 * v) / (w**2) * vol_error)
                    cuda.atomic.add(proj_matrix_error, (angle_index, 2, 1), -(grad_proj_wrt_pos_x * point1 * u + grad_proj_wrt_pos_y * point1 * v) / (w**2) * vol_error)
                    cuda.atomic.add(proj_matrix_error, (angle_index, 2, 2), -(grad_proj_wrt_pos_x * point2 * u + grad_proj_wrt_pos_y * point2 * v) / (w**2) * vol_error)
                    cuda.atomic.add(proj_matrix_error, (angle_index, 2, 3), -(grad_proj_wrt_pos_x * u + grad_proj_wrt_pos_y * v) / (w**2) * vol_error)

