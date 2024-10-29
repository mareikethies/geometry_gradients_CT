.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://opensource.org/licenses/Apache-2.0

.. image:: https://img.shields.io/badge/DOI-10.1088/1361-6560/acf90e.svg
    :target: https://doi.org/10.1088/1361-6560/acf90e

.. image:: https://img.shields.io/badge/arXiv-2212.02177-b31b1b.svg
    :target: https://arxiv.org/abs/2212.02177

Analytical Differentiation for Fan-Beam and Cone-Beam CT Geometry
=================================================================

This repository contains the code for computed tomography (CT) reconstruction in fan-beam and cone-beam geometry which
is differentiable with respect to its acquisition geometry. This works by computing the analytical partial derivatives
of the reconstructed image to the entries of the projection matrices. The backprojector inherits from PyTorch's
``torch.autograd.Function`` to automatically assign the analytical derivative to the differentiable graph in a deep
learning sense.

The code in this repository is at the core of the results presented in our paper **Gradient-based geometry learning for
fan-beam CT reconstruction** which has been published in `Physics in Medicine & Biology <https://doi.org/10.1088/1361-6560/acf90e>`_ 
and focuses on fan-beam geometry. Our follow-up paper **A gradient-based approach to fast and accurate head motion compensation 
in cone-beam CT** extends the concepts to cone-beam geometry and has been published in `IEEE Transactions on Medical
Imaging <https://doi.org/10.1109/TMI.2024.3474250>`_.

Usage
~~~~~~
An instance of the differentiable fan-beam backprojector can be obtained via:

.. code-block:: python

    from backprojector_fan import DifferentiableFanBeamBackprojector
    backprojector = DifferentiableFanBeamBackprojector.apply
    reconstruction = backprojector(sinogram, projection_matrices, geometry)
Similarly, for the cone-beam backprojector:

.. code-block:: python

    from backprojector_cone import DifferentiableConeBeamBackprojector
    backprojector = DifferentiableConeBeamBackprojector.apply
    reconstruction = backprojector(sinogram, projection_matrices, geometry)
These backprojectors can be incorporated into any PyTorch differentiable graph and gradients for the projection
matrices can be obtained.

Example optimization
~~~~~~~~~~~~~~~~~~~~
The script ``example.py`` contains code for a PyTorch-based optimization loop which updates the translational components
of the projection matrices for fan-beam geometry based on a supervised MSE-loss in image domain. It produces the
following output showing how the reconstructed image improves and the target function is minimized:

.. image:: optimization.gif

Gradient check
~~~~~~~~~~~~~~~~~~
The script ``check_gradients.py`` implements a comparison between analytical gradients and their
numerical approximation to demonstrate the correctness of all computations. It contains four functions:

* ``check_gradients_fan()``: plots analytical and numerical gradients for each projection matrix entry for visual comparison in fan-beam geometry
* ``check_gradients_cone()``: plots analytical and numerical gradients for each projection matrix entry for visual comparison in cone-beam geometry
* ``pytorch_gradcheck_fan()``: runs the `PyTorch gradient check <https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html>`_ for fan-beam geometry
* ``pytorch_gradcheck_cone()``: runs the `PyTorch gradient check <https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html>`_ for cone-beam geometry

Technical details
~~~~~~~~~~~~~~~~~
The code uses `numba cuda <https://numba.pydata.org/numba-doc/dev/cuda/index.html>`_ to parallelize the backprojection
operation and gradient computations on GPU. Therefore an
NVIDIA GPU and a running CUDA installation is required. So far, the code has only been tested on Linux operating system.
We provide some example fan-beam and cone-beam data of the Shepp-Logan phantom in the ``./data`` sub-folder. The
geometry settings and conventions of this repository are compatible with those in
`PyroNN <https://github.com/csyben/PYRO-NN>`_.

Citation
~~~~~~~~
If you use this code for your research, please cite our paper:

.. code-block::

    @article{10.1088/1361-6560/acf90e,
        author={Thies, Mareike and Wagner, Fabian and Maul, Noah and Folle, Lukas and Meier, Manuela and Rohleder, Maximilian and Schneider, Linda-Sophie and Pfaff, Laura and Gu, Mingxuan and Utz, Jonas and Denzinger, Felix and Manhart, Michael Thomas and Maier, Andreas},
        title={Gradient-based geometry learning for fan-beam CT reconstruction},
        journal={Physics in Medicine & Biology},
        url={http://iopscience.iop.org/article/10.1088/1361-6560/acf90e},
        year={2023}
    }

or

.. code-block::

    @article{10.1109/TMI.2024.3474250,
        author={Thies, Mareike and Wagner, Fabian and Maul, Noah and Yu, Haijun and Goldmann, Manuela and Schneider, Linda-Sophie and Gu, Mingxuan and Mei, Siyuan and Folle, Lukas and Preuhs, Alexander and Manhart, Michael and Maier, Andreas},
        journal={IEEE Transactions on Medical Imaging}, 
        title={A gradient-based approach to fast and accurate head motion compensation in cone-beam CT}, 
        year={2024},
        doi={10.1109/TMI.2024.3474250}
    }


If you have any questions about this repository or the paper, feel free to reach out
(`mareike.thies@fau.de <mareike.thies@fau.de>`_).
