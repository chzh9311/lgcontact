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
import h5py

from common.utils.geometry import sdf_to_contact
from common.msdf.utils.msdf import calc_local_grid_1pt


class LocalGridDataset(Dataset):
    """
    A dataset class for loading local grid data for hand-object interactions.
    """
    def __init__(self, cfg, split):
        super().__init__()
        self.grid_scale = cfg.msdf.scale
        self.kernel_size = cfg.msdf.kernel_size
        self.dataset_name = cfg.dataset_name
        self.dataset_path = cfg.dataset_path
        self.mano_root = cfg.get('mano_root', 'data/mano')
        self.preprocessed_dir = cfg.get('preprocessed_dir', 'data/preprocessed')
        self.downsample_rate = cfg.downsample_rate[split]
        self.split = split
        mask_file = osp.join(cfg.preprocessed_dir, self.dataset_name, split, f'local_grid_masks_{self.grid_scale*1000:.1f}mm_every{self.downsample_rate}.npy')
        if not osp.exists(mask_file):
            raise FileNotFoundError(f"Mask file not found: {mask_file}. Please run LocalGridDataModule.prepare_data() first.")
        self.masks = np.load(mask_file)
        hand_cse_sd = torch.load(cfg.get('hand_cse_path', 'data/misc/hand_cse.ckpt'), weights_only=True)['state_dict']
        self.hand_cse = hand_cse_sd['embedding_tensor'].detach().cpu().numpy()
        self.mano_faces = hand_cse_sd['cano_faces'].detach().cpu().numpy()
        self.obj_info = {}
        all_samples = np.stack(np.meshgrid(np.arange(self.masks.shape[0]), np.arange(self.masks.shape[1]), indexing='ij'), axis=-1)
        self.idx2sample = all_samples[self.masks, :].reshape(-1, 2)
        self.close_mano_faces = np.load(osp.join('data', 'misc', 'closed_mano_r_faces.npy'), allow_pickle=True)
        ## Downsample data
        # self.idx2sample = self.idx2sample[::cfg.downsample_rate[split], :]
        coords = torch.tensor([(i, j, k) for i in range(self.kernel_size)
                                    for j in range(self.kernel_size)
                                    for k in range(self.kernel_size)], dtype=torch.float32) 
        normalized_coords = 2 * coords - (self.kernel_size - 1)
        self.normalized_coords = (normalized_coords / (self.kernel_size - 1))

        self._load_data()
        if self.downsample_rate > 1:
            self.rh_data = self.sample_frames(self.rh_data)
            self.object_data = self.sample_frames(self.object_data)
            self.frame_names = self.sample_frames(self.frame_names)

    def sample_frames(self, data):
        if type(data) is dict:
            for k, v in data.items():
                data[k] = v[::self.downsample_rate]
        else:
            data = data[::self.downsample_rate]

        return data

    def _load_data(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def __len__(self):
        return self.idx2sample.shape[0]
    
    def __getitem__(self, idx):
        sample_idx, point_idx = self.idx2sample[idx].tolist()
        fname_path = self.frame_names[sample_idx].split('/')
        sbj_id = fname_path[2]
        obj_name = fname_path[3].split('_')[0]
        obj_sample_pt = self.obj_info[obj_name]['samples'][point_idx] # 3
        obj_sample_normal = self.obj_info[obj_name]['sample_normals'][point_idx] # 3
        obj_rot = self.object_data['global_orient'][sample_idx]
        obj_trans = self.object_data['transl'][sample_idx]
        
        objR = axis_angle_to_matrix(obj_rot).detach().cpu().numpy()
        if self.dataset_name == 'grab':
            objR = objR.T
        objt = obj_trans.detach().cpu().numpy()
        obj_sample_pt = (objR @ obj_sample_pt.reshape(3, 1)).reshape(3) + objt
        obj_sample_normal = (objR @ obj_sample_normal.reshape(3, 1)).reshape(3)

        obj_mesh = self.simp_obj_mesh[obj_name]
        obj_verts = (objR @ obj_mesh['verts'].T).T + objt
        obj_faces = obj_mesh['faces']
        obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces, process=False)

        hdata = self.rh_data
        handV, handJ, part_T = self.rh_models[sbj_id](
            th_pose_coeffs=torch.cat([hdata['global_orient'][sample_idx], hdata['fullpose'][sample_idx].view(45)], dim=0)[None, :],
            th_trans=hdata['transl'][sample_idx][None, :])
        nhandV = handV[0].detach().cpu().numpy() - obj_sample_pt[np.newaxis, :]

        grid_file_path = osp.join(self.preprocessed_dir, self.dataset_name, self.split,
                                  f'local_grid_values_{self.grid_scale*1000:.1f}mm_every{self.downsample_rate}',
                                  f'{sample_idx:08d}_local_grid.h5')
        with h5py.File(grid_file_path, 'r') as grid_data_raw:
            verts_mask = grid_data_raw['verts_mask'][point_idx]
            grid_mask = grid_data_raw['grid_mask'][:]
            # Convert unmasked point_idx to masked array index
            masked_point_idx = np.sum(grid_mask[:point_idx]).item()
            # masked_point_idx = np.searchsorted(np.where(grid_mask)[0], point_idx)
            grid_data = grid_data_raw['grid_data'][masked_point_idx]
            ## TODO: do shape transformation for nn_face_idx and nn_point when saving data.
            nn_face_idx = grid_data_raw['nn_face_idx'][masked_point_idx].reshape(self.kernel_size**3)
            nn_point= grid_data_raw['nn_point'][masked_point_idx].reshape(self.kernel_size**3, 3)
            nn_point = nn_point - obj_sample_pt[np.newaxis, :]

        nn_vert_idx = self.mano_faces[nn_face_idx] # N,  3
        face_verts = nhandV[nn_vert_idx]  # (N, 3, 3)
        face_cse = self.hand_cse[nn_vert_idx]  # (M * K^3, 3, cse_dim)
        w = np.linalg.inv(face_verts.transpose((0, 2, 1))) @ nn_point[:, :, np.newaxis]  # (M * K^3, 3, 1)
        # Make sure weights are all positive and normalized
        w = np.clip(w, 0, 1)
        w = w / np.sum(w, axis=1, keepdims=True)  # (M * K^3, 3, 1)

        grid_hand_cse = np.sum(face_cse * w, axis=1).reshape(self.kernel_size, self.kernel_size, self.kernel_size, -1)  # (N, cse_dim)

        # hand_mesh = trimesh.Trimesh(vertices=nhandV, faces=self.close_mano_faces, process=False)
        # grid_data, verts_mask = calc_local_grid_1pt(self.normalized_coords.numpy(), obj_mesh, hand_mesh,
        #                                         self.kernel_size, self.grid_scale, self.hand_cse)
        ## contact data
        grid_data = torch.from_numpy(grid_data).float()
        grid_data[:, :, :, 1] = sdf_to_contact(grid_data[:, :, :, 1] / (self.grid_scale / (self.kernel_size-1)), None, method=2)
        grid_data[:, :, :, 0] = grid_data[:, :, :, 0] / self.grid_scale / np.sqrt(3)  # Normalize SDF
        grid_data = torch.cat([grid_data, torch.from_numpy(grid_hand_cse).float()], dim=-1).numpy()

        sample = {
                  'localGrid': grid_data,
                  'objSamplePt': obj_sample_pt,
                  'objSampleNormal': obj_sample_normal,
                  'objRot': obj_rot,
                  'objTrans': obj_trans,
                  'nHandVerts': nhandV,
                  'handVertMask': verts_mask,
                  'obj_name': obj_name,
                  'face_idx': nn_face_idx,
                  'cse_weights': w.squeeze(-1),
                }

        return sample

