import numpy as np
from copy import deepcopy
import trimesh
import torch
import os.path as osp
from torch.utils.data import Dataset

from common.utils.geometry import transform_obj
from common.utils.vis import o3dmesh_from_trimesh, o3dmesh
from pytorch3d.transforms import axis_angle_to_matrix

def get_kine_parent(idx):
    if idx in [1, 4, 7, 10, 13]:
        return 0
    else:
        return idx - 1

kinetree = {'Index': [0, 1, 2, 3, 17], 'Middle': [0, 4, 5, 6, 18], 'Little': [0, 10, 11, 12, 20],
          'Ring': [0, 7, 8, 9, 19], 'Thumb': [0, 13, 14, 15, 16]}
    

class BaseHOIDataset(Dataset):
    def __init__(self, cfg, split, load_msdf=False, test_gt=False):
        super(BaseHOIDataset, self).__init__()
        self.data_dir = cfg.dataset_path
        self.preprocessed_dir = cfg.get('preprocessed_dir', 'data/preprocessed')
        self.downsample_rate = cfg.downsample_rate[split]
        self.split = split
        # self.n_obj_samples = cfg.object_sample
        self.mano_root = cfg.mano_root
        self.test_gt = test_gt
        self.msdf_scale = cfg.msdf.scale
        self.hand_sides = None
        self.lh_data = None
        if load_msdf:
            self.msdf_path = osp.join(self.preprocessed_dir, cfg.dataset_name,
                                    f'msdf_{cfg.msdf.num_grids}_{cfg.msdf.kernel_size}_{int(cfg.msdf.scale*1000):02d}mm')
        else:
            self.msdf_path = None

        self._load_data()

        if self.downsample_rate > 1:
            self.rh_data = self.sample_frames(self.rh_data)
            self.object_data = self.sample_frames(self.object_data)
            self.frame_names = self.sample_frames(self.frame_names)
            if self.hand_sides is not None:
                self.hand_sides = self.sample_frames(self.hand_sides)
                self.lh_data = self.sample_frames(self.lh_data)

        if self.split == 'test' and not self.test_gt:
            self.num_samples = cfg.test_samples

        ## Calculate partitions

    def _load_data(self):
        self.obj_info = None
        self.rh_data = None
        self.object_data = None
        self.frame_names = None


    def sample_frames(self, data):
        if type(data) is dict:
            for k, v in data.items():
                data[k] = v[::self.downsample_rate]
        else:
            data = data[::self.downsample_rate]

        return data

    def __len__(self):
        # return 900
        if self.split == 'test' and not self.test_gt:
            return len(self.test_objects) * self.num_samples
        else:
            return len(self.frame_names)

    def __getitem__(self, idx):
        if self.split in ['train', 'val'] or self.test_gt:
            fname_path = self.frame_names[idx].split('/')
            sbj_id = fname_path[2]
            obj_name = fname_path[3].split('_')[0]
            obj_sample_pts = self.obj_info[obj_name]['samples'] # n_sample x 3
            # obj_sample_normals = self.obj_info[obj_name]['sample_normals'] # n_sample x 3
            obj_rot = self.object_data['global_orient'][idx]
            obj_trans = self.object_data['transl'][idx]

            ## Transform the vertices:
            objR = axis_angle_to_matrix(obj_rot).detach().cpu().numpy()
            objt = obj_trans.detach().cpu().numpy()
            obj_sample_pts = obj_sample_pts @ objR + objt
            # obj_sample_normals = obj_sample_normals @ objR

            samples = {}
            if self.hand_sides is not None:
                hand_side = self.hand_sides[idx]
                if hand_side == 'left':
                    # change the hand
                    hmodels = self.lh_models
                    hdata = self.lh_data
                else:
                    hmodels = self.rh_models
                    hdata = self.rh_data
            else:
                hand_side = 'right'
                hmodels = self.rh_models
                hdata = self.rh_data

            sample = {
                'frameName': '/'.join(fname_path[2:]),
                'objName': obj_name,
                'sbjId': sbj_id,
                'objSamplePts': obj_sample_pts,
                # 'objSampleNormals': obj_sample_normals,
                'objTrans': obj_trans,
                'objRot': obj_rot,
                'handSide': hand_side,
            }

            if self.msdf_path is not None:
                obj_msdf = self.obj_info[obj_name]['msdf'].copy() ## K^3 + 3
                obj_msdf[:, :-3] = obj_msdf[:, :-3] / self.msdf_scale / np.sqrt(3) # Normalize SDF
                sample['objMsdf'] = obj_msdf

            # hand_out = hmodels[sbj_id](
            #     global_orient=hdata['global_orient'][idx:idx+1],
            #     hand_pose=hdata['fullpose'][idx:idx+1],
            #     transl=hdata['transl'][idx:idx+1])
            # handV, handJ = hand_out.vertices[0].detach().cpu().numpy(), hand_out.joints[0].detach().cpu().numpy()
            handV, handJ, part_T = hmodels[sbj_id](
                th_pose_coeffs=torch.cat([hdata['global_orient'], hdata['fullpose'].view(-1, 45)], dim=1)[idx:idx+1],
                th_trans=hdata['transl'][idx:idx+1])
            handV, handJ, part_T = handV[0].detach().cpu().numpy(), handJ[0].detach().cpu().numpy(), part_T[0].detach().cpu().numpy()
            # model = ManoLayer(mano_root=self.mano_root, flat_hand_mean=True)
            hand_part_ids = torch.argmax(hmodels[sbj_id].th_weights, dim=-1).detach().cpu().numpy()

            handN = trimesh.Trimesh(handV, hmodels[sbj_id].th_faces).vertex_normals
            sample.update({
                'handRot': hdata['global_orient'][idx],
                'handPose': hdata['fullpose'][idx],
                'handTrans': hdata['transl'][idx],
                'handVerts': handV,
                'handJoints': handJ,
                'handPartT': part_T,
                'handPartIds': hand_part_ids,
                'handNormals': np.stack(handN, axis=0),
            })
        else:
            obj_name = self.test_objects[int(idx // self.num_samples)]
            # obj_sample_pts = self.obj_info[obj_name]['samples'] # n_sample x 3
            # obj_sample_normals = self.obj_info[obj_name]['sample_normals'] # n_sample x 3
            ## For testing, return object with random rotations. The rng is set to make sure
            ## each idx generates fixed rotation.
            # obj_rot = R.identity()
            sample = {
                'objName': obj_name,
                # 'objSamplePts': obj_sample_pts,
                # 'objSampleNormals': obj_sample_normals,
                # 'objRot': obj_rot.as_rotvec(),
            }

        return sample

    def get_obj_mesh(self, obj_name, obj_rot, obj_trans):
        obj_verts = self.obj_info[obj_name]['verts']
        obj_faces = self.obj_info[obj_name]['faces']
        obj_verts = transform_obj(obj_verts, obj_rot.cpu().numpy(), obj_trans.cpu().numpy())
        return obj_verts, obj_faces

    def get_obj_hulls(self, obj_name, obj_rot, obj_trans):
        obj_hull = deepcopy(self.obj_hulls[obj_name])
        for i, h in enumerate(obj_hull):
            obj_hull[i].vertices = transform_obj(h.vertices, obj_rot.cpu().numpy(), obj_trans.cpu().numpy())

        return obj_hull


def canonical_hand_parts(mano_layer, num_samples, batch_size, betas=None, device='cpu'):
    cano_verts, cano_joints, cano_part_T = mano_layer(th_pose_coeffs=torch.zeros(batch_size, 48).to(device),
                                                      th_betas=betas, th_trans=torch.zeros(batch_size, 3).to(device))
    cano_verts, cano_joints = cano_verts.detach().cpu().numpy(), cano_joints.detach().cpu().numpy()
    hand_part_ids = torch.argmax(mano_layer.th_weights, dim=-1).detach().cpu().numpy()
    faces = mano_layer.th_faces.detach().cpu().numpy()
    faces_in_part = []
    for part_idx in range(16):
        part_vert_ids = np.where(hand_part_ids == part_idx)[0]
        n_faces = np.sum(np.sum(part_vert_ids.reshape(1, -1) == faces.reshape(-1, 1),
                                      axis=1).reshape(-1, 3), axis=-1)
        faces_in_part.append(np.where(n_faces > 0))

    all_part_sample_pts = []
    all_part_sample_normals = []
    for b in range(batch_size):
        cano_mesh = trimesh.Trimesh(cano_verts[b], faces)
        part_meshes = []
        part_sample_pts = []
        part_sample_normals = []
        for part_idx in range(16):
            part_mesh = cano_mesh.submesh(faces_in_part[part_idx], append=True)
            part_meshes.append(part_mesh)
            if num_samples > 0:
                # part_samples, part_sampled_ids = trimesh.sample.sample_surface(part_mesh, count=num_samples)
                mesh = o3dmesh_from_trimesh(part_mesh)
                pc = mesh.sample_points_poisson_disk(num_samples, init_factor=8)
                part_samples = np.asarray(pc.points)
                part_sample_pts.append(part_samples - cano_joints[b, part_idx].reshape(1, 3))
                part_sample_normals.append(np.asarray(pc.normals))
                # part_sample_normals.append(part_mesh.face_normals[part_sampled_ids])
            else:
                ## Do not sample
                part_samples = part_mesh.vertices
                part_sample_pts.append(part_samples - cano_joints[b, part_idx].reshape(1, 3))
                part_sample_normals = part_mesh.vertex_normals

        if num_samples > 0:
            part_sample_pts = np.stack(part_sample_pts, axis=0)
            part_sample_normals = np.stack(part_sample_normals, axis=0)
        else:
            max_len = max([part_sample_pts[i].shape[0] for i in range(16)])
            part_sample_pts = np.stack([np.pad(p, pad_width=((0, max_len-p.shape[0]), (0, 0))) for p in part_sample_pts], axis=0)
            part_sample_normals = np.stack([np.pad(p, pad_width=((0, max_len-p.shape[0]), (0, 0))) for p in part_sample_normals], axis=0)

        all_part_sample_pts.append(part_sample_pts)
        all_part_sample_normals.append(part_sample_normals)

    all_part_sample_pts = np.stack(all_part_sample_pts, axis=0)
    all_part_sample_normals = np.stack(all_part_sample_normals, axis=0)

    return all_part_sample_pts, all_part_sample_normals, cano_joints
