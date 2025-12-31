from itertools import product

import numpy as np
from tqdm import tqdm
import torch
import trimesh
from pysdf import SDF
from skimage.measure import marching_cubes

def get_grid(kernel_size, device='cpu') -> torch.Tensor:
    """
    Get the cubic interpolant for the given kernel size in range [-1, 1].
    """
    interpolant = torch.zeros(kernel_size, kernel_size, kernel_size, 3, device=device)
    for i, j, k in product(range(kernel_size), repeat=3):
        coord = torch.tensor([i, j, k], device=device)
        val = 2 * coord - (kernel_size - 1)
        val = val / (kernel_size - 1)
        interpolant[i, j, k, :] = val
    return interpolant


def calculate_weights(X, centers, scales) -> torch.Tensor:
    """
    Calculate the weight for points X.
    :arg
    X: torch.Tensor, (N, 3), the points to calculate the weight.
    centers: torch.Tensor, (M, 3), the centers of the grid.
    scales: torch.Tensor, (M), the scales of the grid.
    :return
    weights: torch.Tensor, (N, M), the weights for each point and each grid.
    """
    distances = (X.view(-1, 1, 3) - centers.view(1, -1, 3)) / scales.view(1, -1, 1)
    max_norm = torch.norm(distances, p=torch.inf, dim=2)
    weights = torch.nn.functional.relu(1 - max_norm)  # N, M
    norm_weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
    return norm_weights


def find_closest_vertices_indices(x, kernel_size):
    # this code was taken from the hashgrid implementation
    # https://github.com/Ending2015a/hash-grid-encoding/blob/master/encoding.py#L109
    dim = x.shape[-1]
    n_neigs = 2 ** dim
    neigs = np.arange(n_neigs, dtype=np.int64).reshape((-1, 1))
    dims = np.arange(dim, dtype=np.int64).reshape((1, -1))
    bin_mask = torch.tensor(neigs & (1 << dims) == 0, dtype=bool).to(x.device)  # (neig, dim)

    bdims = len(x.shape[:-1])
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2 * (kernel_size - 1)
    xi = x.long()
    xi = torch.clamp(xi, 0, max=kernel_size - 2)
    xf = x - xi.float().detach()

    xi = xi.unsqueeze(dim=-2)  # (b..., 1, dim)
    xf = xf.unsqueeze(dim=-2)  # (b..., 1, dim)
    # to match the input batch shape
    bin_mask = bin_mask.reshape((1,) * bdims + bin_mask.shape)  # (1..., neig, dim)
    # get neighbors' indices and weights on each dim
    inds = torch.where(bin_mask, xi, xi + 1)  # (b..., neig, dim)
    ws = torch.where(bin_mask, 1 - xf, xf)  # (b...., neig, dim)
    # aggregate nehgibors' interp weights
    w = ws.prod(dim=-1, keepdim=True)  # (b..., neig, 1)
    return w, inds  # (b..., feat)


def get_values_at_indices(Vi: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Get the values from Vi at the given indices, optimized.
    :param Vi: matrix with values shape (M, k, k, k)
    :param indices: indices for Vi matrix shape (N, M, 8, 3)
    :return: values from Vi at indices shape (N, M, 8)
    """

    N, M, _, _ = indices.shape  # Extract dimensions
    m_range = torch.arange(M)[None, :, None].to(Vi.device)  # Shape: (1, M, 1) for broadcasting over M
    I, J, K = indices.to(Vi.device).unbind(-1)  # Each will have shape (N, M, 8)
    results = Vi[m_range, I, J, K]
    return results


def msdf_at_point_batched(X, centers, scales, Vi, kernel_size, batch_size: int = 4096 * 16) -> torch.Tensor:
    result = []
    for i in tqdm(range(0, X.size(0), batch_size)):
        range_end = min(i + batch_size, X.size(0))
        result.append(msdf_at_point(X[i:range_end], centers, scales, Vi, kernel_size))
    return torch.cat(result, dim=0)


def msdf_at_point(X, centers, scales, Vi, kernel_size) -> torch.Tensor:
    """
    Calculate the MSDF value at point X.
    :arg
    X: torch.Tensor, (N, 3), the points to calculate the weight.
    centers: torch.Tensor, (M, 3), the centers of the grid.
    scales: torch.Tensor, (M), the scales of the grid.
    Vi: torch.Tensor, (M, k, k, k), the SDF values grid.
    kernel_size: int, the kernel size used for the grid.
    :return
    msdf: torch.Tensor, (N), the SDF value at the points.
    """

    centered = (X.view(-1, 1, 3) - centers.view(1, -1, 3)) / scales.view(1, -1, 1)  # N, M, 3
    out_of_cube = torch.any(torch.abs(centered) > 1, dim=-1)
    out_of_cube = torch.all(out_of_cube, dim=-1)
    inside_cube = ~out_of_cube
    result = torch.zeros(X.size(0), device=X.device).float()
    if not torch.any(inside_cube):
        return result
    centered = centered[inside_cube]

    centered_flat = centered.view(-1, 3).float()  # N * M, 3
    weights, indices = find_closest_vertices_indices(centered_flat, kernel_size)
    indices = indices.view(centered.size(0), centers.size(0), 8, 3)
    weights = weights.view(centered.size(0), centers.size(0), 8)

    values_at_corners = get_values_at_indices(Vi, indices)
    weights = weights.to(values_at_corners.device)
    values = (values_at_corners * weights).sum(dim=-1)
    weights = calculate_weights(X[inside_cube], centers, scales)
    weighted_values = (values * weights).sum(dim=1).float()
    result[inside_cube] = weighted_values
    return result


def calculate_msdf_value(scale: float, points: torch.Tensor, mesh: trimesh.Trimesh, kernel_size) -> torch.Tensor:
    all_points = get_grid_points(points, scale, kernel_size)
    n_points, n_grids, _ = all_points.shape
    f = SDF(mesh.vertices, mesh.faces)
    all_points = all_points.detach().cpu().numpy()
    sdf_values = f(all_points.reshape(-1, 3)).reshape(n_points, n_grids)
    return -torch.tensor(sdf_values, device=points.device)


def sample_volume(bounds, resolution, bound_delata=0.1, device='cpu'):
    """
    Sample a 3D volume defined by 'bounds' at a given 'resolution'.
    bounds: (min, maxh)
    resolution: int, number of samples per dimension
    bound_delta: float, the delta to add to the bounds to avoid sampling on the boundary.
    device: torch device to use for the tensor
    Returns a tensor of shape (N, 3) of sampled points.
    """
    delta = (bounds[1] - bounds[0]) * bound_delata
    grid_x = torch.linspace(bounds[0] - delta, bounds[1] + delta, resolution)
    grid_y = torch.linspace(bounds[0] - delta, bounds[1] + delta, resolution)
    grid_z = torch.linspace(bounds[0] - delta, bounds[1] + delta, resolution)
    meshgrid = torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    points = torch.stack(meshgrid, dim=-1).reshape(-1, 3).to(device)
    return points


def get_grid_points(centers, scales, kernel_size):
    """
    Get the grid points for the given centers and scales.
    :param centers: Nx3 tensor of centers
    :param scales: N tensor of scales
    :param kernel_size: kernel size used
    :return: NxK*K*Kx3 tensor of grid points
    """
    grid = get_grid(kernel_size=kernel_size, device=centers.device)
    grid_flat = grid.reshape(-1, 3).to(centers.device)
    if isinstance(scales, torch.Tensor):
        scales = scales.view(-1, 1, 1)
    all_points = centers.reshape(centers.shape[0], 1, 3) + grid_flat.view(1, -1, 3) * scales
    return all_points


def reconstruct_mesh(Vi, scales, centers, kernel_size, resolution=16, threshold=0.0) -> trimesh.Trimesh:
    grid = sample_volume((-1, 1), resolution, device=centers.device)
    Vi = Vi.view(Vi.size(0), kernel_size, kernel_size, kernel_size)

    grid_sdf = msdf_at_point_batched(grid, centers, scales, Vi, kernel_size)
    grid_sdf = grid_sdf.detach().cpu().numpy().reshape(resolution, resolution, resolution)

    verts, faces, _, _ = marching_cubes(-grid_sdf, level=threshold)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh


def reconstruct_from_sdf(sdf, resolution):
    grid = sample_volume((-1, 1), resolution)
    grid_sdf = sdf(grid)
    grid_sdf = grid_sdf.reshape(resolution, resolution, resolution)

    verts, faces, _, _ = marching_cubes(grid_sdf, level=0.0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh


def calculate_contact_mask(point_cloud, hand_verts, grid_scale, device):
    """
    Calculate contact mask for a given point cloud.

    Args:
        point_cloud: torch tensor of shape (B, N, 3)
        hand_verts: torch tensor of shape (B, H, 3)
        grid_scale: float, scale of half the grid
        device: torch device to use
    Returns:
        mask: torch tensor of shape (B, N) indicating contact points
    """
    # Convert inputs to torch tensors on device if needed
    if not isinstance(point_cloud, torch.Tensor):
        point_cloud = torch.from_numpy(point_cloud).float().to(device)
    else:
        point_cloud = point_cloud.to(device)

    if not isinstance(hand_verts, torch.Tensor):
        hand_verts = torch.from_numpy(hand_verts).float().to(device)
    else:
        hand_verts = hand_verts.to(device)

    ## Determine grid contact using Chebyshev distance
    # point_cloud: (B, N, 3), hand_verts: (B, H, 3)
    cbdist = torch.max(torch.abs(point_cloud[:, :, None, :] - hand_verts[:, None, :, :]), dim=-1)[0]  # (B, N, H)
    obj_pt_mask = torch.min(cbdist, dim=-1)[0] < grid_scale  # (B, N)
    hand_vert_mask = torch.min(cbdist, dim=1)[0] < grid_scale  # (B, H)

    return obj_pt_mask, hand_vert_mask


def calc_local_grid_batch(contact_points, normalized_coords, obj_mesh, kernel_size, grid_scale, hand_verts, hand_cse, device):
    M = contact_points.shape[0]

    if M == 0:
        # No contact points for this sample
        local_grids = torch.empty(0, kernel_size, kernel_size, kernel_size,
                                                2 + hand_cse.shape[-1], device=device)

    else:
        # Scale and translate to world coordinates for each contact point
        # contact_points: (M, 3), normalized_coords: (kernel_size^3, 3)
        # Result: (M, kernel_size^3, 3)
        grid_points_flat = (contact_points[:, None, :] + normalized_coords[None, :, :] * grid_scale).reshape(-1, 3)

        # Calculate SDF values if obj_mesh is provided
        if obj_mesh is not None:
            objSDF = SDF(obj_mesh.vertices, obj_mesh.faces)
            grid_sdfs_np = objSDF(grid_points_flat.cpu().numpy())
            grid_sdfs = torch.from_numpy(grid_sdfs_np).float().to(device).reshape(M, kernel_size, kernel_size, kernel_size, 1)

        # Calculate distance to nearest hand vertex for this sample
        dist_mat = torch.norm(hand_verts[None, :, :] - grid_points_flat[:, None, :], dim=-1)  # (M*kernel_size^3, H)
        nn_dist, nn_idx = torch.min(dist_mat, dim=1)  # (M * kernel_size^3)

        grid_distance = nn_dist.reshape(M, kernel_size, kernel_size, kernel_size, 1)
        grid_hand_cse = hand_cse[nn_idx].reshape(M, kernel_size, kernel_size, kernel_size, -1)

        # Concatenate features based on whether obj_mesh is provided
        if obj_mesh is not None:
            local_grids = torch.cat([grid_sdfs, grid_distance, grid_hand_cse], dim=-1)  # (M, kernel_size, kernel_size, kernel_size, 1 + 1 + cse_dim)
        else:
            local_grids = torch.cat([grid_distance, grid_hand_cse], dim=-1)  # (M, kernel_size, kernel_size, kernel_size, 1 + cse_dim)

    return local_grids


def calc_local_grid(contact_point, normalized_coords, obj_mesh, kernel_size, grid_scale, hand_verts, hand_cse):
    """
    Calculate local grid for a single contact point using numpy on CPU.

    Args:
        contact_point: numpy array of shape (3,), single contact point
        normalized_coords: numpy array of shape (kernel_size^3, 3), normalized grid coordinates
        obj_mesh: trimesh.Trimesh, object mesh
        kernel_size: int, size of the cubic grid
        grid_scale: float, scale of the grid
        hand_verts: numpy array of shape (778, 3), hand vertices
        hand_cse: numpy array of shape (778, cse_dim), hand contact surface embeddings

    Returns:
        local_grid: numpy array of shape (kernel_size, kernel_size, kernel_size, 2 + cse_dim)
        verts_mask: numpy array of shape (n_verts), boolean mask indicating valid vertices
    """
    # Scale and translate to world coordinates
    # contact_point: (3,), normalized_coords: (kernel_size^3, 3)
    # Result: (kernel_size^3, 3)
    grid_points_flat = contact_point[None, :] + normalized_coords * grid_scale

    ## Determine mask using Chebyshev distance
    verts_mask = np.max(np.abs(hand_verts[:, :]), axis=-1) < grid_scale

    # Calculate SDF values
    objSDF = SDF(obj_mesh.vertices, obj_mesh.faces)
    grid_sdfs_np = objSDF(grid_points_flat)
    grid_sdfs = grid_sdfs_np.reshape(kernel_size, kernel_size, kernel_size, 1)

    # Calculate distance to nearest hand vertex
    # grid_points_flat: (kernel_size^3, 3), hand_verts: (778, 3)
    dist_mat = np.linalg.norm(grid_points_flat[:, None, :] - hand_verts[None, :, :], axis=-1)  # (kernel_size^3, 778)
    nn_idx = np.argmin(dist_mat, axis=1)  # (kernel_size^3,)
    nn_dist = np.min(dist_mat, axis=1)  # (kernel_size^3,)

    grid_distance = nn_dist.reshape(kernel_size, kernel_size, kernel_size, 1)
    grid_hand_cse = hand_cse[nn_idx].reshape(kernel_size, kernel_size, kernel_size, -1)

    local_grid = np.concatenate([grid_sdfs, grid_distance, grid_hand_cse], axis=-1)  # (kernel_size, kernel_size, kernel_size, 2 + cse_dim)

    return local_grid, verts_mask


def msdf2mlcontact(obj_msdf, hand_verts, hand_cse, kernel_size, grid_scale):
    """
    calculate MLcontact from MSDF and hand vertices.

    obj_msdf: torch.Tensor, (B, N, R^3 + 3), the MSDF volume.
    hand_verts: torch.Tensor, (B, H, 3), the hand vertices.
    hand_cse: torch.Tensor, (H, C), the hand contact surface embeddings.
    """
    centers = obj_msdf[:, :, kernel_size**3:]  # B, N, 3
    obj_pt_mask, hand_vert_mask = calculate_contact_mask(centers, hand_verts, grid_scale=grid_scale, device=obj_msdf.device)
    normalized_coords = get_grid(kernel_size=kernel_size, device=obj_msdf.device).reshape(-1, 3).float()
    # all_pts = centers[:, :, None, :] + normalized_coords[None, None, :, :] * grid_scale  # B, N, K^3, 3
    local_grid_dist = torch.zeros(
        obj_msdf.shape[0], obj_msdf.shape[1],
        kernel_size, kernel_size, kernel_size,
        device=obj_msdf.device
    )  # B, N, K, K, K, 1 + cse_dim
    local_grid_cse = torch.zeros(
        obj_msdf.shape[0], obj_msdf.shape[1],
        kernel_size, kernel_size, kernel_size, hand_cse.shape[-1], dtype=torch.float32,
        device=obj_msdf.device
    )  # B, N, K, K, K, 1 + cse_dim

    for b in range(obj_msdf.shape[0]):
        local_grid = calc_local_grid_batch(
            contact_points=centers[b][obj_pt_mask[b]],
            normalized_coords=normalized_coords,
            obj_mesh=None,  # not needed here
            kernel_size=kernel_size,
            grid_scale=grid_scale,
            hand_verts=hand_verts[b],
            hand_cse=hand_cse,
            device=obj_msdf.device
        )  # M, K, K, K, 1 + cse_dim
        local_grid_dist[b][obj_pt_mask[b]] = local_grid[..., 0].float()
        local_grid_cse[b][obj_pt_mask[b]] = local_grid[..., 1:].float()
    
    return local_grid_dist, local_grid_cse, obj_pt_mask, hand_vert_mask, normalized_coords