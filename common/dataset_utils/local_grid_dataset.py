import os
import os.path as osp
import pickle
import importlib
import numpy as np
from copy import deepcopy
import trimesh
import torch
from torch.utils.data import Dataset
from pytorch3d.transforms import axis_angle_to_matrix
from scipy.spatial.transform import Rotation as R
from pysdf import SDF
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.utils.geometry import sdf_to_contact


class LocalGridDataset(Dataset):
    """
    A dataset class for loading local grid data for hand-object interactions.
    """
    def __init__(self, cfg):
        super().__init__()
        self.grid_scale = cfg.grid_scale
        self.kernel_size = cfg.kernel_size


class LocalGridDataModule(LightningDataModule):
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.cfg = cfg.data
        self.data_dir = cfg.data.dataset_path
        self.preprocessed_dir = cfg.data.get('preprocessed_dir', 'data/preprocessed')
        self.hand_cse = torch.load(cfg.data.get('hand_cse_path', 'data/misc/hand_cse.pt')).detach().cpu().numpy()
        self.train_batch_size = cfg.train.batch_size
        self.val_batch_size = cfg.val.batch_size
        self.test_batch_size = cfg.test.batch_size
        module = importlib.import_module(f'common.dataset_utils.{cfg.data.dataset_name}_dataset')
        self.base_dataset = getattr(module, cfg.data.dataset_name.upper() + 'Dataset')(cfg.data, split, load_msdf=False)
        self.dataset_class = getattr(module, cfg.data.dataset_name.upper() + 'LocalGridDataset')
    

    def prepare_data(self):
        """
        dump local training data to disk
        """
        for sample in tqdm(self.base_dataset, total=len(self.base_dataset), desc="Preparing local grid data..."):
            obj_rot = sample['objRot'].detach().cpu().numpy()
            obj_t = sample['objTrans'].detach().cpu().numpy()
            objR = R.from_rotvec(obj_rot).as_matrix()
            if self.cfg.dataset_name == 'grab':
                objR = objR.T
            obj_name = sample['objName']
            simp_obj_mesh = self.base_dataset.simp_obj_mesh[obj_name].copy()
            # Apply rotation and translation to mesh vertices
            obj_verts = simp_obj_mesh['verts'] @ objR.T + obj_t
            obj_faces = simp_obj_mesh['faces']
            obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces, process=False)
            hand_verts = sample['handVerts']
            result_grid = calculate_local_grid(sample['objSamplePts'], obj_mesh, self.cfg.msdf.scale, self.cfg.msdf.kernel_size, hand_verts, self.hand_cse)


    def setup(self, stage: str):
        if stage == 'fit':
            self.train_set = self.data_class(self.cfg, 'train')
            self.val_set = self.data_class(self.cfg, 'val')
        elif stage == 'validate':
            self.val_set = self.data_class(self.cfg, 'val')
        elif stage == 'test':
            self.test_set = self.data_class(self.cfg, 'test')


def calculate_local_grid(point_cloud, obj_mesh, grid_scale, kernel_size, hand_verts, hand_cse, device):
    """
    Calculate local grid for a given point cloud.

    Args:
        point_cloud: numpy array of shape (N, 3)
        obj_mesh: trimesh mesh of the object
        grid_scale: float, scale of half the grid
        kernel_size: int, the size of the cubic grid
        hand_verts: numpy array of shape (H, 3)
        hand_cse: numpy array of shape (H, cse_dim)
    Returns:
        local_grids: numpy array of shape (N, kernel_size**3, 1 + 1 + cse_dim)
    """
    ## Determine grid contact using Chebyshev distance
    cbdist = np.max(np.abs(point_cloud[:, None, :] - hand_verts[None, :, :]), axis=-1)  # (N, H) # Chebyshev distance
    mask = np.min(cbdist, axis=1) < grid_scale
    contact_points = point_cloud[mask]  # (M, 3)

    ## Sample a kernel_size^3 3D grid centered at each contact point
    # Create grid in normalized space [-1, 1]
    indices = np.array([(i, j, k) for i in range(kernel_size)
                        for j in range(kernel_size)
                        for k in range(kernel_size)])  # (kernel_size^3, 3)

    # Convert to coordinates in [-1, 1] range (xyz order)
    normalized_coords = 2 * indices - (kernel_size - 1)
    normalized_coords = normalized_coords / (kernel_size - 1)  # (kernel_size^3, 3)

    # Scale and translate to world coordinates for each contact point
    # contact_points: (M, 3), normalized_coords: (kernel_size^3, 3)
    # Result: (M, kernel_size^3, 3)
    grid_points_flat = (contact_points[:, None, :] + normalized_coords[None, :, :] * grid_scale).reshape(-1, 3)

    # Reshape to (M, kernel_size, kernel_size, kernel_size, 3)
    M = contact_points.shape[0]
    objSDF = SDF(obj_mesh.vertices, obj_mesh.faces)
    grid_sdfs = objSDF(grid_points_flat).reshape(M, kernel_size, kernel_size, kernel_size, 1)
    dist_mat = np.linalg.norm(hand_verts[None, :, :] - grid_points_flat[:, None, :], axis=-1)
    nn_dist, nn_idx = np.min(dist_mat, axis=1), np.argmin(dist_mat, axis=1)  # (M * kernel_size^3)
    # grid_contacts = sdf_to_contact(nn_dist / contact_th).reshape(M, kernel_size, kernel_size, kernel_size, 1)
    grid_distance = nn_dist.reshape(M, kernel_size, kernel_size, kernel_size, 1)
    grid_hand_cse = hand_cse[nn_idx].reshape(M, kernel_size, kernel_size, kernel_size, -1)

    local_grids = np.concatenate([grid_sdfs, grid_distance, grid_hand_cse], axis=-1)  # (M, kernel_size, kernel_size, kernel_size, 1 + 1 + cse_dim)

    return local_grids, mask
