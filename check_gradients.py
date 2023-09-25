import torch
import numpy as np
from geometry import Geometry
from backprojector_fan import DifferentiableFanBeamBackprojector
from backprojector_cone import DifferentiableConeBeamBackprojector
from matplotlib import pyplot as plt
from copy import deepcopy
from helper import params_2_proj_matrix
import seaborn as sns
import matplotlib
from tqdm import tqdm
from torch.autograd import gradcheck


def check_gradients_fan(plot=True):
    geometry = Geometry((128, 128), (-63.5, -63.5), (1., 1.), -255., 2.)

    angles = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    dsd = 2000 * np.ones_like(angles)
    dsi = 1000 * np.ones_like(angles)
    tx = np.zeros_like(angles)
    ty = np.zeros_like(angles)

    proj_matrices, _, _ = params_2_proj_matrix(angles, dsd, dsi, tx, ty, geometry.detector_spacing,
                                               -geometry.detector_origin / geometry.detector_spacing)

    proj_matrices = torch.from_numpy(proj_matrices)

    sino = np.load('data/sinogram_filtered.npy')
    sino = torch.from_numpy(sino.copy())
    phantom = torch.from_numpy(np.load('data/image.npy'))
    mini = 0
    maxi = 1

    device = torch.device('cuda')
    proj_matrices = proj_matrices.to(device)
    sino = sino.to(device)
    backprojector = DifferentiableFanBeamBackprojector.apply

    # important: turn on gradients after transferring tensors to device
    proj_matrices.requires_grad = True

    reco = backprojector(sino, proj_matrices, geometry)

    loss = torch.mean(reco)
    loss.backward()

    analytical_gradients_matrices = proj_matrices.grad.cpu().numpy()

    with torch.no_grad():

        h_00 = 0.025
        numerical_gradients_00 = []
        for i in range(len(angles)):
            proj_matrices_ = deepcopy(proj_matrices.detach())
            proj_matrices_[i, 0, 0] = proj_matrices_[i, 0, 0] + h_00
            reco_ = backprojector(sino, proj_matrices_, geometry)
            loss_ = torch.mean(reco_)

            delta_loss = loss_ - loss
            numerical_gradients_00.append(delta_loss / h_00)
        numerical_gradients_00 = np.asarray([g.to('cpu') for g in numerical_gradients_00])

        h_01 = 0.01
        numerical_gradients_01 = []
        for i in range(len(angles)):
            proj_matrices_ = deepcopy(proj_matrices.detach())
            proj_matrices_[i, 0, 1] = proj_matrices_[i, 0, 1] + h_01
            reco_ = backprojector(sino, proj_matrices_, geometry)
            loss_ = torch.mean(reco_)

            delta_loss = loss_ - loss
            numerical_gradients_01.append(delta_loss / h_01)
        numerical_gradients_01 = np.asarray([g.to('cpu') for g in numerical_gradients_01])

        h_02 = 1.5
        numerical_gradients_02 = []
        for i in range(len(angles)):
            proj_matrices_ = deepcopy(proj_matrices.detach())
            proj_matrices_[i, 0, 2] = proj_matrices_[i, 0, 2] + h_02
            reco_ = backprojector(sino, proj_matrices_, geometry)
            loss_ = torch.mean(reco_)

            delta_loss = loss_ - loss
            numerical_gradients_02.append(delta_loss / h_02)
        numerical_gradients_02 = np.asarray([g.to('cpu') for g in numerical_gradients_02])

        h_10 = 0.00025
        numerical_gradients_10 = []
        for i in range(len(angles)):
            proj_matrices_ = deepcopy(proj_matrices.detach())
            proj_matrices_[i, 1, 0] = proj_matrices_[i, 1, 0] + h_10
            reco_ = backprojector(sino, proj_matrices_, geometry)
            loss_ = torch.mean(reco_)

            delta_loss = loss_ - loss
            numerical_gradients_10.append(delta_loss / h_10)
        numerical_gradients_10 = np.asarray([g.to('cpu') for g in numerical_gradients_10])

        h_11 = 0.00025
        numerical_gradients_11 = []
        for i in range(len(angles)):
            proj_matrices_ = deepcopy(proj_matrices.detach())
            proj_matrices_[i, 1, 1] = proj_matrices_[i, 1, 1] + h_11
            reco_ = backprojector(sino, proj_matrices_, geometry)
            loss_ = torch.mean(reco_)

            delta_loss = loss_ - loss
            numerical_gradients_11.append(delta_loss / h_11)
        numerical_gradients_11 = np.asarray([g.to('cpu') for g in numerical_gradients_11])

        h_12 = 0.01
        numerical_gradients_12 = []
        for i in range(len(angles)):
            proj_matrices_ = deepcopy(proj_matrices.detach())
            proj_matrices_[i, 1, 2] = proj_matrices_[i, 1, 2] + h_12
            reco_ = backprojector(sino, proj_matrices_, geometry)
            loss_ = torch.mean(reco_)

            delta_loss = loss_ - loss
            numerical_gradients_12.append(delta_loss / h_12)
        numerical_gradients_12 = np.asarray([g.to('cpu') for g in numerical_gradients_12])

    if plot:

        plt.figure()
        plt.subplot(131)
        plt.imshow(sino.cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.title('sinogram')
        plt.subplot(132)
        plt.imshow(phantom.cpu().numpy(), cmap='gray', vmin=mini, vmax=maxi)
        plt.axis('off')
        plt.title('phantom')
        plt.subplot(133)
        plt.imshow(reco.detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.title('reconstruction')

        sns.set_style('darkgrid')
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'
        matplotlib.rcParams.update({'font.size': 16})

        x = np.linspace(0, 360, 300, endpoint=False)

        plt.figure(figsize=(12.0, 6.0))

        plt.subplot(2, 3, 1)
        plt.plot(x, analytical_gradients_matrices[:, 0, 0], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_00, '--', label=f'numerical', c='coral')
        plt.legend()
        plt.title(f'$h={h_00}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.subplot(2, 3, 2)
        plt.plot(x, analytical_gradients_matrices[:, 0, 1], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_01, '--', label=f'numerical, h={h_01}', c='coral')
        plt.title(f'$h={h_01}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.subplot(2, 3, 3)
        plt.plot(x, analytical_gradients_matrices[:, 0, 2], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_02, '--', label=f'numerical, h={h_02}', c='coral')
        plt.title(f'$h={h_02}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.subplot(2, 3, 4)
        plt.plot(x, analytical_gradients_matrices[:, 1, 0], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_10, '--', label=f'numerical, h={h_10}', c='coral')
        plt.xlabel('Projection index')
        plt.title(f'$h={h_10}$')
        plt.xlim([0, 360])
        # plt.ylim([-0.55, 0.55])
        plt.tight_layout()

        plt.subplot(2, 3, 5)
        plt.plot(x, analytical_gradients_matrices[:, 1, 1], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_11, '--', label=f'numerical, h={h_11}', c='coral')
        plt.xlabel('Projection index')
        plt.title(f'$h={h_11}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.subplot(2, 3, 6)
        plt.plot(x, analytical_gradients_matrices[:, 1, 2], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_12, '--', label=f'numerical, h={h_12}', c='coral')
        plt.xlabel('Projection index')
        plt.title(f'$h={h_12}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.show()


def check_gradients_cone(plot=True):
    geometry = Geometry((128, 128, 128), (-63.5, -63.5, -63.5), (1., 1., 1.), (-255., -255.), (2., 2.))

    proj_matrices = np.load('data/projection_matrices.npy')
    proj_matrices = proj_matrices / proj_matrices[:, 2, 3][:, np.newaxis, np.newaxis]
    proj_matrices = proj_matrices.astype(np.float64)
    proj_matrices = torch.from_numpy(proj_matrices)

    sino = np.load('data/sinogram_filtered_cone.npy')
    sino = sino.astype(np.float64)
    sino = torch.from_numpy(sino.copy())

    mask = torch.rand((128, 128, 128))
    mask = mask > 0.75

    device = torch.device('cuda')
    proj_matrices = proj_matrices.to(device)
    sino = sino.to(device)
    mask = mask.to(device)
    backprojector = DifferentiableConeBeamBackprojector.apply

    # important: turn on gradients after transferring tensors to device
    proj_matrices.requires_grad = True

    reco = backprojector(sino, proj_matrices, geometry)

    print('Computing analytical gradients.')

    loss = torch.sum(reco * mask)
    loss.backward()

    analytical_gradients = proj_matrices.grad.cpu().numpy()

    with torch.no_grad():
        print('Computing numerical gradients 1/12.')
        h_00 = 0.001
        numerical_gradients_00 = get_numerical_gradient(backprojector, sino, proj_matrices, geometry, loss, h_00, 0, 0, mask)

        print('Computing numerical gradients 2/12.')
        h_01 = 0.001
        numerical_gradients_01 = get_numerical_gradient(backprojector, sino, proj_matrices, geometry, loss, h_01, 0, 1, mask)

        print('Computing numerical gradients 3/12.')
        h_02 = 0.001
        numerical_gradients_02 = get_numerical_gradient(backprojector, sino, proj_matrices, geometry, loss, h_02, 0, 2, mask)

        print('Computing numerical gradients 4/12.')
        h_03 = 1
        numerical_gradients_03 = get_numerical_gradient(backprojector, sino, proj_matrices, geometry, loss, h_03, 0, 3, mask)

        print('Computing numerical gradients 5/12.')
        h_10 = 0.005
        numerical_gradients_10 = get_numerical_gradient(backprojector, sino, proj_matrices, geometry, loss, h_10, 1, 0, mask)

        print('Computing numerical gradients 6/12.')
        h_11 = 0.005
        numerical_gradients_11 = get_numerical_gradient(backprojector, sino, proj_matrices, geometry, loss, h_11, 1, 1, mask)

        print('Computing numerical gradients 7/12.')
        h_12 = 0.001
        numerical_gradients_12 = get_numerical_gradient(backprojector, sino, proj_matrices, geometry, loss, h_12, 1, 2, mask)

        print('Computing numerical gradients 8/12.')
        h_13 = 1
        numerical_gradients_13 = get_numerical_gradient(backprojector, sino, proj_matrices, geometry, loss, h_13, 1, 3, mask)

        print('Computing numerical gradients 9/12.')
        h_20 = 0.0001
        numerical_gradients_20 = get_numerical_gradient(backprojector, sino, proj_matrices, geometry, loss, h_20, 2, 0, mask)

        print('Computing numerical gradients 10/12.')
        h_21 = 0.0001
        numerical_gradients_21 = get_numerical_gradient(backprojector, sino, proj_matrices, geometry, loss, h_21, 2, 1, mask)

        print('Computing numerical gradients 11/12.')
        h_22 = 0.0001
        numerical_gradients_22 = get_numerical_gradient(backprojector, sino, proj_matrices, geometry, loss, h_22, 2, 2, mask)

        print('Computing numerical gradients 12/12.')
        h_23 = 0.001
        numerical_gradients_23 = get_numerical_gradient(backprojector, sino, proj_matrices, geometry, loss, h_23, 2, 3, mask)

    if plot:
        sns.set_style('darkgrid')
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'
        matplotlib.rcParams.update({'font.size': 16})

        x = np.linspace(0, 360, 300, endpoint=False)

        plt.figure(figsize=(12.0, 6.0))

        plt.subplot(3, 4, 1)
        plt.plot(x, analytical_gradients[:, 0, 0], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_00, '--', label=f'numerical', c='coral')
        # plt.legend()
        plt.title(f'$h={h_00}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.subplot(3, 4, 2)
        plt.plot(x, analytical_gradients[:, 0, 1], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_01, '--', label=f'numerical, h={h_01}', c='coral')
        plt.title(f'$h={h_01}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.subplot(3, 4, 3)
        plt.plot(x, analytical_gradients[:, 0, 2], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_02, '--', label=f'numerical, h={h_02}', c='coral')
        plt.title(f'$h={h_02}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.subplot(3, 4, 4)
        plt.plot(x, analytical_gradients[:, 0, 3], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_03, '--', label=f'numerical, h={h_03}', c='coral')
        plt.xlabel('Projection index')
        plt.title(f'$h={h_03}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.subplot(3, 4, 5)
        plt.plot(x, analytical_gradients[:, 1, 0], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_10, '--', label=f'numerical, h={h_10}', c='coral')
        plt.xlabel('Projection index')
        plt.title(f'$h={h_10}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.subplot(3, 4, 6)
        plt.plot(x, analytical_gradients[:, 1, 1], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_11, '--', label=f'numerical, h={h_11}', c='coral')
        plt.xlabel('Projection index')
        plt.title(f'$h={h_11}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.subplot(3, 4, 7)
        plt.plot(x, analytical_gradients[:, 1, 2], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_12, '--', label=f'numerical, h={h_12}', c='coral')
        plt.xlabel('Projection index')
        plt.title(f'$h={h_12}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.subplot(3, 4, 8)
        plt.plot(x, analytical_gradients[:, 1, 3], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_13, '--', label=f'numerical, h={h_13}', c='coral')
        plt.xlabel('Projection index')
        plt.title(f'$h={h_13}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.subplot(3, 4, 9)
        plt.plot(x, analytical_gradients[:, 2, 0], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_20, '--', label=f'numerical, h={h_20}', c='coral')
        plt.xlabel('Projection index')
        plt.title(f'$h={h_20}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.subplot(3, 4, 10)
        plt.plot(x, analytical_gradients[:, 2, 1], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_21, '--', label=f'numerical, h={h_21}', c='coral')
        plt.xlabel('Projection index')
        plt.title(f'$h={h_21}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.subplot(3, 4, 11)
        plt.plot(x, analytical_gradients[:, 2, 2], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_22, '--', label=f'numerical, h={h_22}', c='coral')
        plt.xlabel('Projection index')
        plt.title(f'$h={h_22}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.subplot(3, 4, 12)
        plt.plot(x, analytical_gradients[:, 2, 3], label='analytical', c='navy')
        plt.plot(x, numerical_gradients_23, '--', label=f'numerical, h={h_23}', c='coral')
        plt.xlabel('Projection index')
        plt.title(f'$h={h_23}$')
        plt.xlim([0, 360])
        plt.tight_layout()

        plt.show()


def get_numerical_gradient(backprojector, sino, proj_matrices, geometry, loss, h, ind_0, ind_1, mask):
    numerical_gradients = []
    for i in tqdm(range(proj_matrices.shape[0])):
        proj_matrices_ = deepcopy(proj_matrices.detach())
        proj_matrices_[i, ind_0, ind_1] = proj_matrices_[i, ind_0, ind_1] + h
        reco_ = backprojector(sino, proj_matrices_, geometry)
        loss_ = torch.sum(reco_ * mask)

        delta_loss = loss_ - loss
        numerical_gradients.append(delta_loss / h)
    numerical_gradients = np.asarray([g.to('cpu') for g in numerical_gradients])
    return numerical_gradients


def pytorch_gradcheck_fan():
    geometry = Geometry((128, 128), (-63.5, -63.5), (1., 1.), -255., 2.)

    angles = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    dsd = 2000 * np.ones_like(angles)
    dsi = 1000 * np.ones_like(angles)
    tx = np.zeros_like(angles)
    ty = np.zeros_like(angles)

    proj_matrices, _, _ = params_2_proj_matrix(angles, dsd, dsi, tx, ty, geometry.detector_spacing,
                                               -geometry.detector_origin / geometry.detector_spacing)

    proj_matrices = torch.from_numpy(proj_matrices.astype(np.float64))

    sino = np.load('data/sinogram_filtered.npy')
    sino = torch.from_numpy(sino.copy().astype(np.float64))

    device = torch.device('cuda')
    proj_matrices = proj_matrices.to(device)
    sino = sino.to(device)
    backprojector = DifferentiableFanBeamBackprojector.apply

    # important: turn on gradients after transferring tensors to specific device
    proj_matrices.requires_grad = True

    input = (sino, proj_matrices, geometry)
    test = gradcheck(backprojector, input, eps=5e-5, rtol=0.4, atol=0, fast_mode=True, nondet_tol=1e-1)
    print(f'Pytorch gradient check for fan-beam backprojector returned: {test}')


def pytorch_gradcheck_cone():
    geometry = Geometry((128, 128, 128), (-63.5, -63.5, -63.5), (1., 1., 1.), (-255., -255.), (2., 2.))

    proj_matrices = np.load('data/projection_matrices.npy')
    proj_matrices = proj_matrices / proj_matrices[:, 2, 3][:, np.newaxis, np.newaxis]
    proj_matrices = proj_matrices.astype(np.float64)
    proj_matrices = torch.from_numpy(proj_matrices)

    sino = np.load('data/sinogram_filtered_cone.npy')
    sino = sino.astype(np.float64)
    sino = torch.from_numpy(sino.copy())

    device = torch.device('cuda')
    proj_matrices = proj_matrices.to(device)
    sino = sino.to(device)
    backprojector = DifferentiableConeBeamBackprojector.apply

    # important: turn on gradients after transferring tensors to specific device
    proj_matrices.requires_grad = True

    input = (sino, proj_matrices, geometry)
    test = gradcheck(backprojector, input, eps=1e-5, rtol=1e-5, atol=1e-3, fast_mode=True, nondet_tol=1e-1)
    print(f'Pytorch gradient check for cone-beam backprojector returned: {test}')


if __name__ == '__main__':
    check_gradients_fan()
    check_gradients_cone()
    # pytorch_gradcheck_cone()
    # pytorch_gradcheck_fan()
