import os
import os.path as osp
import importlib
import numpy as np
import torch
import trimesh
import h5py
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from tqdm import tqdm
from copy import deepcopy

from multiprocessing.pool import Pool

from common.msdf.utils.msdf import calculate_contact_mask, calc_local_grid_all_pts
from common.msdf.simplified_msdf import mesh2msdf
from common.manopth.manopth.manolayer import ManoLayer
from common.utils.vis import o3dmesh
import open3d as o3d

class LocalGridDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.data
        self.data_dir = cfg.data.dataset_path
        self.dataset_name = cfg.data.dataset_name
        self.preprocessed_dir = cfg.data.get('preprocessed_dir', 'data/preprocessed')
        self.train_batch_size = cfg.train.batch_size
        self.val_batch_size = cfg.val.batch_size
        self.test_batch_size = cfg.test.batch_size
        self.hand_cse = torch.load(cfg.data.get('hand_cse_path', 'data/misc/hand_cse.ckpt'))['state_dict']['embedding_tensor'].detach().cpu().numpy()
        self.mano_layer = ManoLayer(mano_root=cfg.data.mano_root, side='right', use_pca=False, ncomps=45, flat_hand_mean=True)
        self.module = importlib.import_module(f'common.dataset_utils.{cfg.data.dataset_name}_dataset')
        self.dataset_class = getattr(self.module, cfg.data.dataset_name.upper() + 'LocalGridDataset')
        self.mano_faces = self.mano_layer.th_faces.numpy()
        kernel_size = self.cfg.msdf.kernel_size
        coords = torch.tensor([(i, j, k) for i in range(kernel_size)
                                    for j in range(kernel_size)
                                    for k in range(kernel_size)], dtype=torch.float32) 
        normalized_coords = 2 * coords - (kernel_size - 1)
        self.normalized_coords = (normalized_coords / (kernel_size - 1))

    def prepare_data(self):
        """
        dump local training data to disk
        """
        data_cfg = deepcopy(self.cfg)
        for split in ['train', 'val', 'test']:
        # for split in ['val']:
            masks = []
            os.makedirs(osp.join(self.preprocessed_dir, self.dataset_name, split), exist_ok=True)
            self.base_dataset = getattr(self.module, self.cfg.dataset_name.upper() + 'Dataset')(data_cfg, split, load_msdf=False, test_gt=True)
            # loader = DataLoader(self.base_dataset, batch_size=64, shuffle=False, num_workers=8)
            # masks = []
            # if not osp.exists(preprocessed_file):
            #     for sample in tqdm(loader, total=len(loader), desc="Preparing local grid masks..."):
            #         obj_rot = sample['objRot']
            #         obj_t = sample['objTrans']
            #         objR = axis_angle_to_matrix(obj_rot)
            #         if self.cfg.dataset_name == 'grab':
            #             objR = objR.transpose(-1, -2)
            #         obj_name = sample['objName']
            #         # simp_obj_mesh = self.base_dataset.simp_obj_mesh[obj_name]
            #         # Apply rotation and translation to mesh vertices
            #         # obj_verts = simp_obj_mesh['verts'] @ objR.T + obj_t
            #         # obj_faces = simp_obj_mesh['faces']
            #         # obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces, process=False)
            #         hand_verts = sample['handVerts']
            #         # result_grid, mask = calculate_local_grid(sample['objSamplePts'], obj_mesh, self.cfg.msdf.scale, self.cfg.msdf.kernel_size,
            #         #                                 hand_verts, self.hand_cse, device='cuda:0')
            #         # result_grid = result_grid.detach().cpu().numpy()
            #         mask, _ = calculate_contact_mask(sample['objSamplePts'], hand_verts, self.cfg.msdf.scale, device='cuda:0')
            #         mask = mask.detach().cpu().numpy()

            #         ## Also add samples from non-contact regions
            #         # contact_indices = np.where(mask)
            #         # non_contact_indices = np.where(~mask)
            #         # n_contact_samples = len(contact_indices[0])
            #         # n_non_contact_samples = int(n_contact_samples * self.cfg.non_contact_rate)
            #         # if n_non_contact_samples > 0 and len(non_contact_indices[0]) > 0:
            #         #     n_non_contact_samples = min(n_non_contact_samples, len(non_contact_indices[0]))
            #         #     selected_idx = np.random.choice(len(non_contact_indices[0]), n_non_contact_samples, replace=False)
            #         #     selected_non_contact = tuple(idx[selected_idx] for idx in non_contact_indices)
            #         #     mask[selected_non_contact] = True

            #         # Clear GPU cache to prevent memory leaks
            #         torch.cuda.empty_cache()

                    # Save and immediately clear the results to free memory
                    # result['local_grids'].append(result_grid)

            loader = DataLoader(self.base_dataset, batch_size=20, shuffle=False, num_workers=8)
            local_grid_dir = osp.join(self.preprocessed_dir, self.dataset_name, split, f'local_grid_values_{self.cfg.msdf.scale*1000:.1f}mm_every{self.cfg.downsample_rate[split]}')
            if not osp.exists(local_grid_dir):
                os.makedirs(local_grid_dir, exist_ok=True)
                pool = Pool(processes=20)
                for idx, sample in tqdm(enumerate(loader), total=len(loader), desc="Preparing local grid data..."):
                    if split == 'train' and idx < 449:
                        continue
                    obj_samples = sample['objSamplePts'].numpy() # (B, N, 3)
                    hand_verts = sample['handVerts'].numpy() # (B, H, 3)

                    obj_rot = sample['objRot']
                    obj_t = sample['objTrans']
                    objR = axis_angle_to_matrix(obj_rot)
                    if self.dataset_name == 'grab':
                        objR = objR.transpose(-1, -2)
                    obj_names = sample['objName']
                    objR = objR.numpy()
                    obj_meshes = []
                    for i, obj_name in enumerate(obj_names):
                        mesh_dict = self.base_dataset.simp_obj_mesh[obj_name]
                        obj_verts = (objR[i] @ mesh_dict['verts'].T).T + obj_t[i].numpy()
                        obj_faces = mesh_dict['faces']
                        obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces, process=False)
                        obj_meshes.append(obj_mesh)

                    param_list = [{'contact_points': obj_samples[i], 'normalized_coords': self.normalized_coords.numpy(),
                                'obj_mesh': obj_meshes[i], 'hand_mesh': trimesh.Trimesh(vertices=hand_verts[i], faces=self.mano_faces, process=False),
                                'kernel_size': self.cfg.msdf.kernel_size, 'grid_scale': self.cfg.msdf.scale, 'hand_cse': self.hand_cse,
                                'idx': idx * loader.batch_size + i, 'save_dir': local_grid_dir,
                                } for i in range(len(obj_names))]
                    pool.map(self.calc_and_save_local_grid, param_list)
                    # for param in tqdm(param_list):
                    #     print(f"processing sample idx: {param['idx']}")
                    #     self.calc_and_save_local_grid(param)

                pool.close()
                pool.join()

            preprocessed_mask_file = osp.join(self.preprocessed_dir, self.dataset_name, split, f'local_grid_masks_{self.cfg.msdf.scale*1000:.1f}mm_every{self.cfg.downsample_rate[split]}.npy')
            if not osp.exists(preprocessed_mask_file):
                for i in tqdm(range(len(os.listdir(local_grid_dir))), desc="Collecting local grid masks..."):
                    f = osp.join(local_grid_dir, f'{i:08d}_local_grid.h5')
                    with h5py.File(f, 'r') as data:
                        mask = data['grid_mask'][:]
                        masks.append(mask)
                    # if cnt > next_save_cnt:
                    #     concatenated_grids = np.concatenate(result['local_grids'], axis=0)
                    #     concatenated_masks = np.concatenate(result['mask'], axis=0)
                    #     with open(osp.join(self.preprocessed_dir, 'grab', split, f'{cnt + 2 - save_gap:08d}-{cnt:08d}_local_grid.pkl'), 'wb') as f:
                    #         pickle.dump({'local_grids': concatenated_grids, 'mask': concatenated_masks}, f)
                    #     print(f"Saved local grid for subject-object pair: {prev_sbj}, {prev_obj_name}.")

                    # concatenated_grids = np.concatenate(result['local_grids'], axis=0)
                    # concatenated_masks = np.concatenate(result['mask'], axis=0)
                all_mask = np.stack(masks, axis=0)
                np.save(preprocessed_mask_file, all_mask)
                # grid_data, verts_mask, grid_mask, nn_face_idx, nn_point = calc_local_grid_all_pts(obj_samples, self.normalized_coords.numpy(), obj_mesh, hand_mesh,
                #                                         self.cfg.msdf.kernel_size, self.cfg.msdf.scale, self.hand_cse)
                # pool.apply_async(np.savez_compressed, args=(save_path,), kwds={'grid_data': grid_data, 'verts_mask': verts_mask})
    
    @staticmethod
    def calc_and_save_local_grid(param):
        contact_points = param['contact_points']
        normalized_coords = param['normalized_coords']
        obj_mesh = param['obj_mesh']
        hand_mesh = param['hand_mesh']
        kernel_size = param['kernel_size']
        grid_scale = param['grid_scale']
        hand_cse = param['hand_cse']
        idx = param['idx']
        local_grid_dir = param['save_dir']
        grid_data, verts_mask, grid_mask, _, nn_face_idx, nn_point = calc_local_grid_all_pts(contact_points, normalized_coords, obj_mesh, hand_mesh,
                                                        kernel_size, grid_scale, hand_cse)
        save_path = osp.join(local_grid_dir, f'{idx:08d}_local_grid.h5')
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('grid_data', data=grid_data, compression='gzip', compression_opts=4)
            f.create_dataset('verts_mask', data=verts_mask, compression='gzip', compression_opts=4)
            f.create_dataset('grid_mask', data=grid_mask, compression='gzip', compression_opts=4)
            f.create_dataset('nn_face_idx', data=nn_face_idx, compression='gzip', compression_opts=4)
            f.create_dataset('nn_point', data=nn_point, compression='gzip', compression_opts=4)
        # return grid_mask
        print('Saved local grid data to ', save_path)

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


class HOIDatasetModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.data
        self.data_dir = cfg.data.dataset_path
        self.preprocessed_dir = cfg.data.get('preprocessed_dir', 'data/preprocessed')
        self.train_batch_size = cfg.train.batch_size
        self.val_batch_size = cfg.val.batch_size
        self.test_batch_size = cfg.test.batch_size
        self.dataset_name = cfg.data.dataset_name
        self.test_gt = cfg.get('test_gt', False)
        self.module = importlib.import_module(f'common.dataset_utils.{cfg.data.dataset_name}_dataset')
        self.dataset_class = getattr(self.module, cfg.data.dataset_name.upper() + 'Dataset')
        self.mano_layer = ManoLayer(mano_root=cfg.data.mano_root, side='right', use_pca=False, ncomps=45, flat_hand_mean=True)
        self.mano_faces = self.mano_layer.th_faces.numpy()
        kernel_size = self.cfg.msdf.kernel_size
        coords = torch.tensor([(i, j, k) for i in range(kernel_size)
                                    for j in range(kernel_size)
                                    for k in range(kernel_size)], dtype=torch.float32)
        normalized_coords = 2 * coords - (kernel_size - 1)
        self.normalized_coords = (normalized_coords / (kernel_size - 1))

    def prepare_data(self):
        """
        Dump precalculated variables, mainly M-SDF
        """
        obj_info = self.dataset_class.load_mesh_info(self.data_dir)
        for k, v in obj_info.items():
            mesh = trimesh.Trimesh(v['verts'], v['faces'], process=False)
            if 'msdf' in self.cfg:
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
                
            if hasattr(self.cfg, 'n_obj_samples'):
                sample_path = osp.join(self.preprocessed_dir, self.cfg.dataset_name,
                                    f'obj_samples_{self.cfg.n_obj_samples}', f'{k}.npz')
                if not osp.exists(osp.dirname(sample_path)):
                    os.makedirs(osp.dirname(sample_path))
                if not osp.exists(sample_path):
                    print(f'Preprocessing object surface samples for {k}...')
                    sample_pts, fid = mesh.sample(self.cfg.n_obj_samples, return_index=True)
                    sample_normals = mesh.face_normals[fid]
                    np.savez_compressed(sample_path, samples=sample_pts, sample_normals=sample_normals)
                    print('result saved to ', sample_path)
            
        ## Preprocess the grid contact data of the hand.
        data_cfg = deepcopy(self.cfg)
        for split in ['train', 'val', 'test']:
            os.makedirs(osp.join(self.preprocessed_dir, self.dataset_name, split), exist_ok=True)
            self.base_dataset = getattr(self.module, self.dataset_name.upper() + 'Dataset')(data_cfg, split, load_msdf=True, test_gt=True)
            dataset_file = osp.join(self.preprocessed_dir, self.dataset_name, split, f'hand_grid_contact_{self.cfg.msdf.num_grids}_{self.cfg.msdf.kernel_size}_{int(self.cfg.msdf.scale*1000):02d}mm_every{self.cfg.downsample_rate[split]}.h5')
            loader = DataLoader(self.base_dataset, batch_size=20, shuffle=False, num_workers=8)
            if not osp.exists(dataset_file):
                pool = Pool(processes=20)
                total_samples = len(self.base_dataset)
                kernel_size = self.cfg.msdf.kernel_size
                n_grids = self.cfg.msdf.num_grids

                # Create HDF5 file with resizable datasets
                with h5py.File(dataset_file, 'w') as f:
                    # Create resizable datasets - we'll resize as we write
                    f.create_dataset('local_grid', shape=(0, n_grids, kernel_size, kernel_size, kernel_size, 1),
                                     maxshape=(total_samples, n_grids, kernel_size, kernel_size, kernel_size, 1),
                                     dtype='float32', compression='gzip', compression_opts=4, chunks=(1, n_grids, kernel_size, kernel_size, kernel_size, 1))
                    f.create_dataset('ho_dist', shape=(0, n_grids, 778),
                                     maxshape=(total_samples, n_grids, 778),
                                     dtype='float32', compression='gzip', compression_opts=4, chunks=(1, n_grids, 778))
                    f.create_dataset('nn_face_idx', shape=(0, n_grids, kernel_size, kernel_size, kernel_size),
                                     maxshape=(total_samples, n_grids, kernel_size, kernel_size, kernel_size),
                                     dtype='int32', compression='gzip', compression_opts=4, chunks=(1, n_grids, kernel_size, kernel_size, kernel_size))
                    f.create_dataset('nn_point', shape=(0, n_grids, kernel_size, kernel_size, kernel_size, 3),
                                     maxshape=(total_samples, n_grids, kernel_size, kernel_size, kernel_size, 3),
                                     dtype='float32', compression='gzip', compression_opts=4, chunks=(1, n_grids, kernel_size, kernel_size, kernel_size, 3))

                    write_idx = 0
                    for idx, sample in tqdm(enumerate(loader), total=len(loader), desc="Preparing hand grid contact data..."):
                        centers_np = sample['objMsdf'][..., -3:].numpy()  # (B, n_grids, 3)
                        hand_verts_np = sample['handVerts'].numpy()
                        batch_size_actual = centers_np.shape[0]

                        # Build object transform matrix
                        objR = axis_angle_to_matrix(sample['objRot']).numpy()  # (B, 3, 3)
                        objTrans_np = sample['objTrans'].numpy()  # (B, 3)

                        objT = np.eye(4)[None, ...].repeat(batch_size_actual, axis=0)
                        objT[:, :3, :3] = objR.transpose(0, 2, 1) if self.dataset_name == 'grab' else objR
                        objT[:, :3, 3] = objTrans_np

                        # Compute inverse transform
                        objT_inv = np.eye(4)[None, ...].repeat(batch_size_actual, axis=0)
                        objT_inv[:, :3, :3] = objT[:, :3, :3].transpose(0, 2, 1)  # R^T
                        objT_inv[:, :3, 3] = -(objT_inv[:, :3, :3] @ objT[:, :3, 3:4]).squeeze(-1)  # -R^T * t

                        # Transform hand vertices
                        homo_hand_verts = np.concatenate([hand_verts_np, np.ones((*hand_verts_np.shape[:-1], 1))], axis=-1)
                        hand_verts_np = (objT_inv[:, None, :, :] @ homo_hand_verts[..., None])[:, :, :3, 0]

                        # hand_mesh = o3dmesh(hand_verts_np[0], self.mano_faces)
                        # contact_pts = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(centers_np[0]))
                        # o3d.visualization.draw_geometries([hand_mesh, contact_pts])

                        # Prepare parameters for parallel processing
                        param_list = [{
                            'contact_points': centers_np[b],
                            'normalized_coords': self.normalized_coords.numpy(),
                            'hand_verts': hand_verts_np[b],
                            'mano_faces': self.mano_faces,
                            'kernel_size': self.cfg.msdf.kernel_size,
                            'grid_scale': self.cfg.msdf.scale,
                        } for b in range(batch_size_actual)]

                        # Run in parallel
                        results = pool.map(self._calc_hand_grid_contact, param_list)

                        # Write results incrementally
                        for result in results:
                            local_grid, ho_dist, nn_face_idx, nn_point = result
                            # Resize datasets
                            f['local_grid'].resize(write_idx + 1, axis=0)
                            f['ho_dist'].resize(write_idx + 1, axis=0)
                            f['nn_face_idx'].resize(write_idx + 1, axis=0)
                            f['nn_point'].resize(write_idx + 1, axis=0)
                            # Write data
                            f['local_grid'][write_idx] = local_grid
                            f['ho_dist'][write_idx] = ho_dist
                            f['nn_face_idx'][write_idx] = nn_face_idx
                            f['nn_point'][write_idx] = nn_point
                            write_idx += 1

                pool.close()
                pool.join()
                print(f'Saved hand grid contact data to {dataset_file}')
    
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

    @staticmethod
    def _calc_hand_grid_contact(param):
        """
        Worker function for parallel computation of hand grid contact.
        Returns local_grid, ho_dist, nn_face_idx, nn_point for a single sample.
        """
        contact_points = param['contact_points']
        normalized_coords = param['normalized_coords']
        hand_verts = param['hand_verts']
        mano_faces = param['mano_faces']
        kernel_size = param['kernel_size']
        grid_scale = param['grid_scale']

        hand_mesh = trimesh.Trimesh(vertices=hand_verts, faces=mano_faces, process=False)
        local_grid, verts_mask, grid_mask, ho_dist, nn_face_idx, nn_point = calc_local_grid_all_pts(
            contact_points=contact_points,
            normalized_coords=normalized_coords,
            obj_mesh=None,
            hand_mesh=hand_mesh,
            kernel_size=kernel_size,
            grid_scale=grid_scale,
        )

        n_grids = contact_points.shape[0]
        # Reconstruct full arrays with zeros for non-contact grids
        full_local_grid = np.zeros((n_grids, kernel_size, kernel_size, kernel_size, 1), dtype=np.float32)
        full_nn_face_idx = np.zeros((n_grids, kernel_size, kernel_size, kernel_size), dtype=np.int32)
        full_nn_point = np.zeros((n_grids, kernel_size, kernel_size, kernel_size, 3), dtype=np.float32)

        # Fill in values for valid grids
        full_local_grid[grid_mask] = local_grid
        full_nn_face_idx[grid_mask] = nn_face_idx
        full_nn_point[grid_mask] = nn_point

        return full_local_grid, ho_dist.astype(np.float32), full_nn_face_idx, full_nn_point

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_set = self.dataset_class(self.cfg, 'train', load_msdf=True, load_grid_contact=True)
            self.val_set = self.dataset_class(self.cfg, 'val', load_msdf=True, load_grid_contact=True)
        elif stage == 'validate':
            self.val_set = self.dataset_class(self.cfg, 'val', load_msdf=True, load_grid_contact=True)
        elif stage == 'test':
            self.test_set = self.dataset_class(self.cfg, 'test', load_msdf=True, load_grid_contact=self.test_gt, test_gt=self.test_gt)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=False, num_workers=self.cfg.num_workers)

    def test_dataloader(self):
        batch_size = self.test_batch_size if self.test_gt else 1
        ## load only one object at a time for sample generation
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=self.cfg.num_workers) 

