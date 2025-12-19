import pickle
import os
import os.path as osp
import numpy as np
from numpy.dtypes import StringDType
import torch
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import trimesh
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
import mcubes
import open3d as o3d

from utils.geometry import mesh_to_sdf
from utils.vis import o3dmesh

class ObjSDFDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = cfg.data.dataset_name
        self.data_dir = cfg.data.dataset_path
        self.train_batch_size = cfg.train.batch_size
        self.val_batch_size = cfg.val.batch_size
        self.test_batch_size = cfg.test.batch_size
        self.num_obj_samples = cfg.data.num_obj_samples
        self.num_sdf_samples = cfg.data.num_sdf_samples
        self.object_list = cfg.data.object_list
        self.th_sdf = cfg.data.th_sdf

    def prepare_data(self):
        for split in ['train', 'val', 'test']:
            preprocess_path = osp.join('data', 'preprocessed', 'objects', 'ycb_'+split+'.pkl')
            ## Sample near the object surfaces
            if not osp.exists(preprocess_path):
                result = {'obj_name': [], 'sdf': [], 'samplePts': [], 'sampleNormals': [], 'com': []}
                for obj_name in tqdm(self.object_list[split]):
                    # mesh = trimesh.load(os.path.join(self.data_dir,  obj_name+'.ply'))
                    mesh = load_mesh(self.dataset_name, self.data_dir, obj_name)
                    sample_pts, idxs = trimesh.sample.sample_surface(mesh, self.num_obj_samples)
                    sample_normals = mesh.face_normals[idxs]
                    com = np.mean(sample_pts, axis=0)
                    sample_pts = sample_pts - com
                    mesh.apply_translation(-com)

                    ## Sample SDFs from surroundings
                    BBox3D = np.array([[np.min(mesh.vertices[:, 0]) - self.th_sdf, np.max(mesh.vertices[:, 0]) + self.th_sdf],
                                      [np.min(mesh.vertices[:, 1]) - self.th_sdf, np.max(mesh.vertices[:, 1]) + self.th_sdf],
                                      [np.min(mesh.vertices[:, 2]) - self.th_sdf, np.max(mesh.vertices[:, 2]) + self.th_sdf]])
                    ## uniformly sample points inside this 3D bbox
                    valid_sdfs = []
                    valid_cnt = 0
                    for i in range(10):
                        samples = (BBox3D[:, 1] - BBox3D[:, 0]).reshape(1, 3) * np.random.rand(int(2 * self.num_sdf_samples), 3) + BBox3D[:, 0].reshape(1, 3)
                        sdfs = mesh_to_sdf(mesh, samples)
                        valid_mask = np.abs(sdfs) < self.th_sdf
                        sdfs = sdfs[valid_mask]
                        valid_cnt += np.sum(valid_mask)
                        valid_sdfs.append(np.concatenate((samples[valid_mask], sdfs.reshape(-1, 1)), axis=-1))
                        if valid_cnt >= self.num_sdf_samples:
                            break

                    if valid_cnt < self.num_sdf_samples:
                        print(f'WARNING: sampled {obj_name} 10 times found only {valid_cnt} points, while {self.num_obj_samples} points are needed.')
                        continue
                    valid_sdfs = np.concatenate(valid_sdfs, axis=0)[:self.num_sdf_samples]

                    result['obj_name'].append(obj_name)
                    result['com'].append(com)
                    result['sdf'].append(valid_sdfs)
                    result['samplePts'].append(sample_pts)
                    result['sampleNormals'].append(sample_normals)

                for k, v in result.items():
                    if k == 'obj_name':
                        result[k] = np.array(v, dtype=StringDType())
                    else:
                        result[k] = np.stack(v, axis=0)

                with open(preprocess_path, 'wb') as f:
                    pickle.dump(result, f)

            # else:
            #     with open(preprocess_path, 'rb') as f:
            #         res = pickle.load(f)
                #
                # for obj_name, v_dict in res.items():
                #     sdf = v_dict['sdf'].reshape(self.sdf_resolution, self.sdf_resolution, self.sdf_resolution)
                #     vertices, faces = mcubes.marching_cubes(sdf, 0)
                #     mesh = o3dmesh(vertices, faces)
                #     o3d.visualization.draw_geometries([mesh])


    def setup(self, stage: str):
        if stage == 'fit':
            self.train_set = ObjSDFDataset(self.cfg, 'train')
            self.val_set = ObjSDFDataset(self.cfg, 'val')
        elif stage == 'validate':
            self.val_set = ObjSDFDataset(self.cfg, 'val')
        elif stage == 'test':
            self.test_set = ObjSDFDataset(self.cfg, 'test')


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=False, num_workers=self.cfg.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=False, num_workers=self.cfg.num_workers)


def load_mesh(dataset_name, data_dir, obj_name):
    if dataset_name == 'ycb':
        obj_path = osp.join(data_dir, 'simple_ycb_models', obj_name, 'textured_simple_2000.obj')
        return trimesh.load(obj_path, process=False)


class ObjSDFDataset(Dataset):
    def __init__(self, cfg, split):
        self.data_dir = cfg.data.dataset_path
        with open(osp.join(self.data_dir, split+'.pkl'), 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data['obj_name'])

    def __getitem__(self, idx):
        sample = {}
        for k, v in self.data:
            sample[k] = v[idx]

        return sample