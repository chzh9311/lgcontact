import os
import os.path as osp
import importlib
import numpy as np
import torch
import trimesh
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from tqdm import tqdm

from common.msdf.utils.msdf import calculate_contact_mask
from common.msdf.simplified_msdf import mesh2msdf

class LocalGridDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.data
        self.data_dir = cfg.data.dataset_path
        self.preprocessed_dir = cfg.data.get('preprocessed_dir', 'data/preprocessed')
        self.train_batch_size = cfg.train.batch_size
        self.val_batch_size = cfg.val.batch_size
        self.test_batch_size = cfg.test.batch_size
        self.module = importlib.import_module(f'common.dataset_utils.{cfg.data.dataset_name}_dataset')
        self.dataset_class = getattr(self.module, cfg.data.dataset_name.upper() + 'LocalGridDataset')

    def prepare_data(self):
        """
        dump local training data to disk
        """
        for split in ['train', 'val', 'test']:
            os.makedirs(osp.join(self.preprocessed_dir, 'grab', split), exist_ok=True)
            self.base_dataset = getattr(self.module, self.cfg.dataset_name.upper() + 'Dataset')(self.cfg, split, load_msdf=False)
            loader = DataLoader(self.base_dataset, batch_size=64, shuffle=False, num_workers=8)
            masks = []
            preprocessed_file = osp.join(self.preprocessed_dir, 'grab', split, f'local_grid_masks_{self.cfg.msdf.scale*1000:.1f}mm.npy')
            if not osp.exists(preprocessed_file):
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
                    mask, _ = calculate_contact_mask(sample['objSamplePts'], hand_verts, self.cfg.msdf.scale, device='cuda:0')
                    mask = mask.detach().cpu().numpy()

                    ## Also add samples from non-contact regions
                    contact_indices = np.where(mask)
                    non_contact_indices = np.where(~mask)
                    n_contact_samples = len(contact_indices[0])
                    n_non_contact_samples = int(n_contact_samples * self.cfg.non_contact_rate)
                    if n_non_contact_samples > 0 and len(non_contact_indices[0]) > 0:
                        n_non_contact_samples = min(n_non_contact_samples, len(non_contact_indices[0]))
                        selected_idx = np.random.choice(len(non_contact_indices[0]), n_non_contact_samples, replace=False)
                        selected_non_contact = tuple(idx[selected_idx] for idx in non_contact_indices)
                        mask[selected_non_contact] = True

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
                np.save(preprocessed_file, all_mask)

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_set = self.dataset_class(self.cfg, 'train')
            self.val_set = self.dataset_class(self.cfg, 'val')
        elif stage == 'validate':
            self.val_set = self.dataset_class(self.cfg, 'val')
        elif stage == 'test':
            self.test_set = self.dataset_class(self.cfg, 'test')

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=True, num_workers=self.cfg.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=False, num_workers=self.cfg.num_workers)


class HOIDatasetModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.data
        self.data_dir = cfg.data.dataset_path
        self.preprocessed_dir = cfg.data.get('preprocessed_dir', 'data/preprocessed')
        self.train_batch_size = cfg.train.batch_size
        self.val_batch_size = cfg.val.batch_size
        self.test_batch_size = cfg.test.batch_size
        module = importlib.import_module(f'common.dataset_utils.{cfg.data.dataset_name}_dataset')
        self.dataset_class = getattr(module, cfg.data.dataset_name.upper() + 'Dataset')

    def prepare_data(self):
        """
        Dump precalculated variables, mainly M-SDF
        """
        obj_info = self.dataset_class.load_mesh_info(self.data_dir)
        for k, v in obj_info.items():
            mesh = trimesh.Trimesh(v['verts'], v['faces'], process=False)
            msdf_path = osp.join(self.preprocessed_dir, self.cfg.dataset_name,
                                 f'msdf_{self.cfg.msdf.num_grids}_{self.cfg.msdf.kernel_size}_{int(self.cfg.msdf.scale*1000):02d}mm',
                                 f'{k}.npz')
            if not osp.exists(osp.dirname(msdf_path)):
                os.makedirs(osp.dirname(msdf_path))
            if not osp.exists(msdf_path):
                print(f'Preprocessing M-SDF for {k}...')
                ## M-SDF: K^3 + 3
                msdf = mesh2msdf(mesh, n_samples=self.cfg.msdf.num_grids, kernel_size=self.cfg.msdf.kernel_size, scale=self.cfg.msdf.scale)
                                 
                np.savez_compressed(msdf_path, msdf=msdf)
                print('result saved to ', msdf_path)
    
        # Simplified object meshes
        # simp_obj_mesh_path = osp.join('data', 'preprocessed', 'simplified_obj_mesh.pkl')
        # if not osp.exists(simp_obj_mesh_path):
        #     simp_obj_mesh = {}
        #     obj_info = np.load(osp.join(self.data_dir, 'obj_info.npy'), allow_pickle=True).item()
        #     for k, v in obj_info.items():
        #         simp_obj_mesh[k] = {'verts': [], 'faces': []}
        #         mesh = o3dmesh(v['verts'], v['faces'])
        #         mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=self.cfg.n_simp_faces)
        #         # mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
        #         simp_obj_mesh[k] = {'verts': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)}
        #     with open(simp_obj_mesh_path, 'wb') as f:
        #         pickle.dump(simp_obj_mesh, f)


    def setup(self, stage: str):
        if stage == 'fit':
            self.train_set = self.dataset_class(self.cfg, 'train')
            self.val_set = self.dataset_class(self.cfg, 'val')
        elif stage == 'validate':
            self.val_set = self.dataset_class(self.cfg, 'val')
        elif stage == 'test':
            self.test_set = self.dataset_class(self.cfg, 'test')

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=False, num_workers=self.cfg.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=False, num_workers=self.cfg.num_workers)

