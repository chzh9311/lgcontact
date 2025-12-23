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

def mesh2msdf(mesh, n_samples, kernel_size, scale):
    sampled_points = farthest_point_sampling(mesh.vertices, n_samples)
    sampled_points = torch.from_numpy(sampled_points)

    # scales = torch.ones(n_samples).float() * scale
    # grid_points = get_grid_points(sampled_points, scale, kernel_size).detach().cpu().numpy().reshape(-1, 3)
    initial_msdf_values = calculate_msdf_value(scale, sampled_points, mesh, kernel_size)

    return np.concatenate((initial_msdf_values.view(-1, kernel_size ** 3), sampled_points), axis=1)