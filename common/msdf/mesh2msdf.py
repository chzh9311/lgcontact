import torch
from scipy.spatial import cKDTree
from pysdf import SDF
from .utils.mesh import farthest_point_sampling
from .optimize_msdf import MSDFOptimizer, calculate_scale
from .utils.msdf import calculate_msdf_value, get_grid_points

def mesh2msdf(mesh, num_grids, kernel_size):
    """
    Optimize the mesh for MSDF generation.
    :param input_path: path to the input mesh
    :param output_path: path to the output msdf representation
    :param visualize: whether to visualize the mesh
    :param resolution: resolution of the grid for marching cubes
    :param with_optimization: whether to optimize the msdf values after initialization
    """
    # w_vertices, w_faces = pcu.make_mesh_watertight(mesh.vertices, mesh.faces)
    # mesh = trimesh.Trimesh(vertices=w_vertices, faces=w_faces)
    # if visualize:
    #     visualize_mesh(mesh, name="watertight_mesh")

    # sdf_gt = SDF(mesh.vertices, mesh.faces)

    sampled_points = farthest_point_sampling(mesh.vertices, num_grids)
    sampled_points = torch.from_numpy(sampled_points)
    init_scale = calculate_scale(sampled_points)
    # logger.info(f'Initial scale: {init_scale}')

    scales = torch.ones(num_grids).float() * init_scale
    grid_points = get_grid_points(sampled_points, init_scale, kernel_size).detach().cpu().numpy().reshape(-1, 3)
    initial_msdf_values = calculate_msdf_value(init_scale, sampled_points, mesh, kernel_size)

    msdf_values = initial_msdf_values.view(-1, kernel_size ** 3)
    optimizer = MSDFOptimizer(msdf_values.view(-1, kernel_size, kernel_size, kernel_size).clone(),
                                sampled_points.clone(), scales.clone(), mesh, kernel_size)
    optimizer.optimize()
    optimized_msdf_values = optimizer.Vi.detach().cpu()
    optimized_scales = optimizer.scales.detach().cpu()
    optimized_centers = optimizer.centers.detach().cpu()
    n_grids = optimized_msdf_values.size(0)

    concatenated = torch.cat([optimized_msdf_values.view(n_grids, -1),
                              optimized_centers.view(n_grids, 3),
                              optimized_scales[:, None]], dim=1)
    return concatenated