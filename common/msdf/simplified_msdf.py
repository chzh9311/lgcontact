"""
Simplified version of MSDF generation. We dont care about the fine-grained surface details
MSDF centres are obtained by farest point sampling on the mesh surface.
MSDF scales are fixed due to the scale-awareness in real application.
"""

import numpy as np
import torch
import trimesh
from .utils.mesh import farthest_point_sampling
from .utils.msdf import calculate_msdf_value

def compute_sdf_gradient_on_grid(sdf_values, kernel_size, scale):
    """
    Compute SDF gradient via central finite differences on a regular grid.

    Args:
        sdf_values: torch.Tensor (n_samples, kernel_size^3), SDF values on the grid.
        kernel_size: int, the grid resolution per axis.
        scale: float, the half-extent of each grid cell's domain.

    Returns:
        gradients: numpy array (n_samples, kernel_size^3, 3), the SDF gradient at each grid point.
    """
    n_samples = sdf_values.shape[0]
    K = kernel_size
    # Reshape to (n_samples, K, K, K)
    vol = sdf_values.view(n_samples, K, K, K).float()

    # Spacing between adjacent grid nodes: total extent is 2*scale, divided into (K-1) intervals
    h = 2.0 * scale / (K - 1)

    # Central differences along each axis, forward/backward at boundaries
    grad_x = torch.zeros_like(vol)
    grad_y = torch.zeros_like(vol)
    grad_z = torch.zeros_like(vol)

    # Interior: central differences
    if K > 2:
        grad_x[:, 1:-1, :, :] = (vol[:, 2:, :, :] - vol[:, :-2, :, :]) / (2 * h)
        grad_y[:, :, 1:-1, :] = (vol[:, :, 2:, :] - vol[:, :, :-2, :]) / (2 * h)
        grad_z[:, :, :, 1:-1] = (vol[:, :, :, 2:] - vol[:, :, :, :-2]) / (2 * h)

    # Boundaries: forward/backward differences
    grad_x[:, 0, :, :] = (vol[:, 1, :, :] - vol[:, 0, :, :]) / h
    grad_x[:, -1, :, :] = (vol[:, -1, :, :] - vol[:, -2, :, :]) / h
    grad_y[:, :, 0, :] = (vol[:, :, 1, :] - vol[:, :, 0, :]) / h
    grad_y[:, :, -1, :] = (vol[:, :, -1, :] - vol[:, :, -2, :]) / h
    grad_z[:, :, :, 0] = (vol[:, :, :, 1] - vol[:, :, :, 0]) / h
    grad_z[:, :, :, -1] = (vol[:, :, :, -1] - vol[:, :, :, -2]) / h

    # Stack to (n_samples, K, K, K, 3) then reshape to (n_samples, K^3, 3)
    gradients = torch.stack([grad_x, grad_y, grad_z], dim=-1)
    return gradients.reshape(n_samples, K ** 3, 3).numpy()


def mesh2msdf(mesh, n_samples, kernel_size, scale, return_gradient=False):
    sampled_points = farthest_point_sampling(mesh.vertices, n_samples)
    sampled_points = torch.from_numpy(sampled_points)

    # scales = torch.ones(n_samples).float() * scale
    # grid_points = get_grid_points(sampled_points, scale, kernel_size).detach().cpu().numpy().reshape(-1, 3)
    initial_msdf_values = calculate_msdf_value(scale, sampled_points, mesh, kernel_size)

    if not return_gradient:
        return np.concatenate((initial_msdf_values.view(-1, kernel_size ** 3), sampled_points), axis=1)

    # Compute SDF gradient via finite differences on the grid
    sdf_gradients = compute_sdf_gradient_on_grid(initial_msdf_values, kernel_size, scale)
    # sdf_gradients: (n_samples, kernel_size^3, 3)
    return np.concatenate((initial_msdf_values.view(-1, kernel_size ** 3), sampled_points), axis=1), sdf_gradients