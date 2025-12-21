import os
import os.path as osp
import pickle
import importlib
import numpy as np
from copy import deepcopy
import trimesh
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch3d.transforms import axis_angle_to_matrix
from scipy.spatial.transform import Rotation as R
from pysdf import SDF
from lightning import LightningDataModule
from tqdm import tqdm

from common.utils.geometry import sdf_to_contact


class LocalGridDataset(Dataset):
    """
    A dataset class for loading local grid data for hand-object interactions.
    """
    def __init__(self, cfg):
        super().__init__()
        self.grid_scale = cfg.msdf.scale
        self.kernel_size = cfg.msdf.kernel_size


class LocalGridDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.data
        self.data_dir = cfg.data.dataset_path
        self.preprocessed_dir = cfg.data.get('preprocessed_dir', 'data/preprocessed')
        self.hand_cse = torch.load(cfg.data.get('hand_cse_path', 'data/misc/hand_cse.pt')).detach().cpu().numpy()
        self.train_batch_size = cfg.train.batch_size
        self.val_batch_size = cfg.val.batch_size
        self.test_batch_size = cfg.test.batch_size
        self.module = importlib.import_module(f'common.dataset_utils.{cfg.data.dataset_name}_dataset')
        self.dataset_class = getattr(self.module, cfg.data.dataset_name.upper() + 'LocalGridDataset')
    

    def prepare_data(self):
        """
        dump local training data to disk
        """
        for split in ['train']:
            os.makedirs(osp.join(self.preprocessed_dir, 'grab', split), exist_ok=True)
            self.base_dataset = getattr(self.module, self.cfg.dataset_name.upper() + 'Dataset')(self.cfg, split, load_msdf=False)
            loader = DataLoader(self.base_dataset, batch_size=64, shuffle=False, num_workers=8)
            masks = []
            for sample in tqdm(loader, total=len(loader), desc="Preparing local grid data..."):
                obj_rot = sample['objRot']
                obj_t = sample['objTrans']
                objR = axis_angle_to_matrix(obj_rot)
                if self.cfg.dataset_name == 'grab':
                    objR = objR.transpose(-1, -2)
                obj_name = sample['objName']
                # simp_obj_mesh = self.base_dataset.simp_obj_mesh[obj_name]
                # Apply rotation and translation to mesh vertices
                # obj_verts = simp_obj_mesh['verts'] @ objR.T + obj_t
                # obj_faces = simp_obj_mesh['faces']
                # obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces, process=False)
                hand_verts = sample['handVerts']
                # result_grid, mask = calculate_local_grid(sample['objSamplePts'], obj_mesh, self.cfg.msdf.scale, self.cfg.msdf.kernel_size,
                #                                 hand_verts, self.hand_cse, device='cuda:0')
                # result_grid = result_grid.detach().cpu().numpy()
                mask = calculate_contact_mask(sample['objSamplePts'], hand_verts, self.cfg.msdf.scale, device='cuda:0')
                mask = mask.detach().cpu().numpy()

                # Clear GPU cache to prevent memory leaks
                torch.cuda.empty_cache()

                # Save and immediately clear the results to free memory
                # result['local_grids'].append(result_grid)
                masks.append(mask)
                # if cnt > next_save_cnt:
                #     concatenated_grids = np.concatenate(result['local_grids'], axis=0)
                #     concatenated_masks = np.concatenate(result['mask'], axis=0)
                #     with open(osp.join(self.preprocessed_dir, 'grab', split, f'{cnt + 2 - save_gap:08d}-{cnt:08d}_local_grid.pkl'), 'wb') as f:
                #         pickle.dump({'local_grids': concatenated_grids, 'mask': concatenated_masks}, f)
                #     print(f"Saved local grid for subject-object pair: {prev_sbj}, {prev_obj_name}.")

            # concatenated_grids = np.concatenate(result['local_grids'], axis=0)
            # concatenated_masks = np.concatenate(result['mask'], axis=0)
            all_mask = np.concatenate(masks, axis=0)
            np.save(osp.join(self.preprocessed_dir, 'grab', split, f'local_grid_masks_{self.cfg.msdf.scale*1000:.1f}mm.npy'), all_mask)
            # Clean up

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_set = self.data_class(self.cfg, 'train')
            self.val_set = self.data_class(self.cfg, 'val')
        elif stage == 'validate':
            self.val_set = self.data_class(self.cfg, 'val')
        elif stage == 'test':
            self.test_set = self.data_class(self.cfg, 'test')


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
    mask = torch.min(cbdist, dim=-1)[0] < grid_scale  # (B, N)

    return mask


def calculate_local_grid_from_mask(point_cloud, mask, obj_meshes, grid_scale, kernel_size, hand_verts, hand_cse, device):
    """
    Calculate local grid for contact points specified by mask.

    Args:
        point_cloud: torch tensor of shape (B, N, 3)
        mask: torch tensor of shape (B, N) indicating contact points
        obj_meshes: list of B trimesh meshes of the objects
        grid_scale: float, scale of half the grid
        kernel_size: int, the size of the cubic grid
        hand_verts: torch tensor of shape (B, H, 3)
        hand_cse: torch tensor of shape (H, cse_dim)
        device: torch device to use
    Returns:
        local_grids: list of B tensors, each of shape (M_i, kernel_size, kernel_size, kernel_size, 1 + 1 + cse_dim)
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

    if not isinstance(hand_cse, torch.Tensor):
        hand_cse = torch.from_numpy(hand_cse).float().to(device)
    else:
        hand_cse = hand_cse.to(device)

    B = point_cloud.shape[0]
    local_grids_batch = []

    ## Sample a kernel_size^3 3D grid centered at each contact point
    # Create grid in normalized space [-1, 1]
    indices = torch.tensor([(i, j, k) for i in range(kernel_size)
                            for j in range(kernel_size)
                            for k in range(kernel_size)], dtype=torch.float32, device=device)  # (kernel_size^3, 3)

    # Convert to coordinates in [-1, 1] range (xyz order)
    normalized_coords = 2 * indices - (kernel_size - 1)
    normalized_coords = normalized_coords / (kernel_size - 1)  # (kernel_size^3, 3)

    for b in range(B):
        contact_points = point_cloud[b][mask[b]]  # (M, 3)
        M = contact_points.shape[0]

        if M == 0:
            # No contact points for this sample
            local_grids_batch.append(torch.empty(0, kernel_size, kernel_size, kernel_size,
                                                 2 + hand_cse.shape[-1], device=device))
            continue

        # Scale and translate to world coordinates for each contact point
        # contact_points: (M, 3), normalized_coords: (kernel_size^3, 3)
        # Result: (M, kernel_size^3, 3)
        grid_points_flat = (contact_points[:, None, :] + normalized_coords[None, :, :] * grid_scale).reshape(-1, 3)

        # Calculate SDF values (need to convert to numpy for pysdf)
        objSDF = SDF(obj_meshes[b].vertices, obj_meshes[b].faces)
        grid_sdfs_np = objSDF(grid_points_flat.cpu().numpy())
        grid_sdfs = torch.from_numpy(grid_sdfs_np).float().to(device).reshape(M, kernel_size, kernel_size, kernel_size, 1)

        # Calculate distance to nearest hand vertex for this sample
        dist_mat = torch.norm(hand_verts[b][None, :, :] - grid_points_flat[:, None, :], dim=-1)  # (M*kernel_size^3, H)
        nn_dist, nn_idx = torch.min(dist_mat, dim=1)  # (M * kernel_size^3)

        grid_distance = nn_dist.reshape(M, kernel_size, kernel_size, kernel_size, 1)
        grid_hand_cse = hand_cse[nn_idx].reshape(M, kernel_size, kernel_size, kernel_size, -1)

        local_grids = torch.cat([grid_sdfs, grid_distance, grid_hand_cse], dim=-1)  # (M, kernel_size, kernel_size, kernel_size, 1 + 1 + cse_dim)
        local_grids_batch.append(local_grids)
    
    local_grids_batch = torch.cat(local_grids_batch, dim=0)

    return local_grids_batch


def calculate_local_grid(point_cloud, obj_mesh, grid_scale, kernel_size, hand_verts, hand_cse, device):
    """
    Calculate local grid for a given point cloud.

    Args:
        point_cloud: torch tensor of shape (N, 3)
        obj_mesh: trimesh mesh of the object
        grid_scale: float, scale of half the grid
        kernel_size: int, the size of the cubic grid
        hand_verts: torch tensor of shape (H, 3)
        hand_cse: torch tensor of shape (H, cse_dim)
        device: torch device to use
    Returns:
        local_grids: torch tensor of shape (M, kernel_size, kernel_size, kernel_size, 1 + 1 + cse_dim)
        mask: torch tensor of shape (N,) indicating contact points
    """
    mask = calculate_contact_mask(point_cloud, hand_verts, grid_scale, device)
    local_grids = calculate_local_grid_from_mask(point_cloud, mask, obj_mesh, grid_scale, kernel_size, hand_verts, hand_cse, device)
    return local_grids, mask
