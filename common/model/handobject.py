import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from copy import copy, deepcopy
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d, axis_angle_to_quaternion
import open3d as o3d
import open3d.visualization as vis
from matplotlib import pyplot as plt

from common.manopth.manopth.manolayer import ManoLayer
from common.utils.vis import o3dmesh_from_trimesh, o3d_arrow, geom_to_img, extract_masked_mesh_components
from common.utils.misc import linear_normalize
from common.utils.geometry import (
        flip_x_axis,
        transform_mesh,
        sdf_to_contact,
        calculate_contact_capsule
    )

from common.msdf.utils.msdf import msdf2mlcontact, get_grid

class HandObject:
    def __init__(self, cfg, device, mano_layer=None, normalize=True):
        self.device = device
        self.cfg = cfg
        if mano_layer is None:
            self.mano_layer = ManoLayer(mano_root=cfg.mano_root, side='right', use_pca=False, flat_hand_mean=True)
        else:
            self.mano_layer = mano_layer

        self.closed_hand_faces = np.load("data/misc/closed_mano_r_faces.npy")
        self.hand_cse = torch.load(cfg.get('hand_cse_path', 'data/misc/hand_cse.ckpt'), weights_only=True)['state_dict']['embedding_tensor'].detach().to(self.device)
        self.hand_part_ids = torch.argmax(self.mano_layer.th_weights, dim=-1).detach().cpu().numpy()
        self.hand_verts = None
        self.hand_pose = None
        self.hand_root_rot = None
        self.hand_trans = None
        self.obj_verts = None
        self.contact_map = None
        self.part_map = None
        self.obj_normals = None
        self.hand_joints = None
        self.hand_sides = None
        self.hand_models = []
        self.obj_models = []
        self.obj_hulls = []
        self.obj_names = []
        self.obj_rot = None
        self.obj_trans = None
        self.normalize = normalize
        self.obj_com = None
        self.contact_unit = cfg.contact_unit
        self.normalized_coords = get_grid(kernel_size=self.cfg.msdf.kernel_size, device=self.device).reshape(-1, 3).float()

    def __copy__(self):
        new_ho = HandObject(cfg=self.cfg, device=self.device, normalize=self.normalize)
        new_ho.obj_names = self.obj_names
        new_ho.mano_layer = self.mano_layer
        new_ho.hand_part_ids = copy(self.hand_part_ids)
        new_ho.hand_sides = copy(self.hand_sides)
        new_ho.obj_verts = copy(self.obj_verts)
        new_ho.hand_verts = copy(self.hand_verts)
        new_ho.hand_root_rot = copy(self.hand_root_rot)
        new_ho.hand_pose = copy(self.hand_pose)
        new_ho.hand_trans = copy(self.hand_trans)
        new_ho.contact_map = copy(self.contact_map)
        new_ho.part_map = copy(self.part_map)
        new_ho.obj_normals = copy(self.obj_normals)
        new_ho.hand_joints = copy(self.hand_joints)
        new_ho.hand_models = deepcopy(self.hand_models)
        new_ho.obj_models = deepcopy(self.obj_models)
        new_ho.obj_hulls = deepcopy(self.obj_hulls)
        new_ho.obj_rot = copy(self.obj_rot)
        new_ho.obj_trans = copy(self.obj_trans)
        new_ho.obj_com = copy(self.obj_com)
        # new_ho.batch_size = self.batch_size
        return new_ho

    def load_from_batch(self, batch, obj_templates=None, obj_hulls=None, pool=None):
        """
        Load the sampled vertices of objects from batched data. Used for training & testing.
        Force labels are also loaded.
        Used in training and validating
        """
        # self.obj_verts = batch['objSamplePts'].clone()
        # self.obj_normals = batch['objSampleNormals'].clone()
        self.obj_rot = batch['objRot'].clone().to(self.device).float()
        self.obj_trans = batch['objTrans'].clone().to(self.device).float()
        self.obj_names = batch['objName']
        self.hand_pose = batch['handPose'].clone().to(self.device).float()
        self.hand_root_rot = batch['handRot'].clone().to(self.device).float()
        self.hand_trans = batch['handTrans'].clone().to(self.device).float()
        self.hand_verts = batch['handVerts'].clone().to(self.device).float()
        self.hand_normals = batch['handNormals'].clone().to(self.device).float()
        self.hand_joints = batch['handJoints'].clone().to(self.device).float()
        self.obj_verts = batch['objSamplePts'].clone().to(self.device).float()
        self.obj_normals = batch['objSampleNormals'].clone().to(self.device).float()
        self.cano_joints = batch['canoJoints'].clone().to(self.device).float()
        self.batch_size = self.obj_rot.shape[0]
        # handV, handJ = batch['handVerts'].clone().cpu().numpy(), batch['handJoints'][:, :16].clone().cpu().numpy()
        # self.hand_models = [trimesh.Trimesh(handV[i], self.closed_hand_faces.copy()) for i in range(handV.shape[0])]

        # hand_frames = batch['handPartT'].clone()

        # Do transformations
        objT = torch.eye(4).to(self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        objR = axis_angle_to_matrix(self.obj_rot)
        objT[:, :3, :3] = objR
        objT[:, :3, 3] = self.obj_trans

        # Load object templates
        if obj_templates is not None:
            self.obj_models = []
            for b in range(self.batch_size):
                obj_mesh = copy(obj_templates[b])
                self.obj_models.append(obj_mesh)

        # Load object hulls
        if obj_hulls is not None:
            self.obj_hulls = []
            for b in range(self.batch_size):
                ohs = []
                for h in obj_hulls[b]:
                    h0 = copy(h)
                    ohs.append(h0)
                self.obj_hulls.append(ohs)

        if self.normalize:
            # Apply inverse transformation objT^-1 to hand related attributes
            # Compute inverse transformation
            objT_inv = torch.eye(4).to(self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
            objT_inv[:, :3, :3] = objT[:, :3, :3].transpose(-1, -2)  # R^T
            objT_inv[:, :3, 3] = -(objT_inv[:, :3, :3] @ objT[:, :3, 3:4]).squeeze(-1)  # -R^T * t

            # Transform hand vertices
            homo_hand_verts = F.pad(self.hand_verts, (0, 1), 'constant', 1)
            self.hand_verts = (objT_inv.unsqueeze(1) @ homo_hand_verts.unsqueeze(-1))[:, :, :3, 0]

            # Transform hand joints
            homo_hand_joints = F.pad(self.hand_joints, (0, 1), 'constant', 1)
            self.hand_joints = (objT_inv.unsqueeze(1) @ homo_hand_joints.unsqueeze(-1))[:, :, :3, 0]

            root_j = self.cano_joints[:, 0]
            ## Transform hand rotation & translation
            self.hand_root_rot = matrix_to_axis_angle(objT_inv[:, :3, :3] @ axis_angle_to_matrix(self.hand_root_rot))
            self.hand_trans = (objT_inv[:, :3, :3] @ (self.hand_trans + root_j).unsqueeze(-1)).squeeze(-1) + objT[:, :3, 3] - root_j
            # self.hand_trans = (objT_inv[:, :3, :3] @ self.hand_trans[:, :, None]).squeeze(-1) + objT_inv[:, :3, 3]

            # Compute and transform hand normals
            # self.hand_normals = torch.stack([
            #     torch.as_tensor(self.hand_models[i].vertex_normals, dtype=torch.float32, device=self.device)
            #     for i in range(len(self.hand_models))
            # ], dim=0)
            self.hand_normals = (objT_inv[:, :3, :3].unsqueeze(1) @ self.hand_normals.unsqueeze(-1)).squeeze(-1)

            # Transform hand models
            # for i in range(len(self.hand_models)):
            #     objT_inv_np = objT_inv[i].detach().cpu().numpy()
            #     self.hand_models[i] = transform_mesh(self.hand_models[i], objT_inv_np)

            # Transform hand frames with inverse transformation
            # hand_frames = (objT_inv.unsqueeze(1) @ hand_frames).to(self.device)

            # Apply inverse transformation objT^-1 to object vertices and normals
            # (they are already transformed when loaded from batch)
            homo_obj_verts = F.pad(self.obj_verts, (0, 1), 'constant', 1)
            self.obj_verts = (objT_inv.unsqueeze(1) @ homo_obj_verts.unsqueeze(-1))[:, :, :3, 0]

            # Transform object normals with inverse transformation
            self.obj_normals = (objT_inv[:, :3, :3].unsqueeze(1) @ self.obj_normals.unsqueeze(-1)).squeeze(-1)
        else:
            # When normalize=False, object vertices and normals are already transformed
            # Only transform object models and hulls which are loaded from canonical space

            # Transform object models
            if obj_templates is not None:
                for i in range(len(self.obj_models)):
                    objT_np = objT[i].detach().cpu().numpy()
                    self.obj_models[i] = transform_mesh(self.obj_models[i], objT_np)

            # Transform object hulls
            if obj_hulls is not None:
                for i in range(len(self.obj_hulls)):
                    objT_np = objT[i].detach().cpu().numpy()
                    for j in range(len(self.obj_hulls[i])):
                        self.obj_hulls[i][j] = transform_mesh(self.obj_hulls[i][j], objT_np)
            self.obj_com = self.obj_verts.mean(dim=1, keepdim=True)
            self.obj_verts = self.obj_verts - self.obj_com
            self.hand_verts = self.hand_verts - self.obj_com
            self.hand_joints = self.hand_joints - self.obj_com
        
        ## Compute Local grid contact representation
        if self.contact_unit == 'grid':
            self.obj_msdf = batch['objMsdf'].clone().to(self.device).float()
            # ml_dist, ml_cse, mask, hand_vert_mask, ho_dist, normalized_coords = msdf2mlcontact(self.obj_msdf, self.hand_verts,
            #                                                                           self.hand_cse, self.cfg.msdf.kernel_size, self.cfg.msdf.scale,
            #                                                                           self.mano_layer.th_faces, pool=pool)

            # ml_dist[mask] = sdf_to_contact(ml_dist[mask] / (self.cfg.msdf.scale / (self.cfg.msdf.kernel_size-1)), None, method=2)
            ## Do the mapping: 0 -> -1; 1 -> 0; infty -> 1: contact = 1 - 2/(dist + 1)
            # ho_dist = ho_dist / self.cfg.msdf.scale # For grid-level contact
            # self.n_ho_dist = 1 - 2 / (ho_dist + 1)
            # # ml_contact[mask, :, :, :, 0] = sdf_to_contact(ml_contact[mask, :, :, :, 0] / (self.cfg.msdf.scale / self.cfg.msdf.num_grids), None, method=0)
            # self.ml_contact = torch.cat([ml_dist.unsqueeze(-1), ml_cse], dim=-1)
            # self.normalized_coords = normalized_coords
            # self.obj_pt_mask = mask
            # self.hand_vert_mask = hand_vert_mask

            self.n_ho_dist = batch['nHoDist'].clone().to(self.device).float()
            self.ml_contact = torch.cat([batch['localGridContact'], batch['localGridCSE']], dim=-1).to(self.device).float()
            self.obj_pt_mask = batch['objPtMask'].clone().to(self.device).bool()
            self.hand_vert_mask = batch['handVertMask'].clone().to(self.device).bool()

            ## For profiling
            # self.normalized_coords = get_grid(kernel_size=self.cfg.msdf.kernel_size, device=self.device).reshape(-1, 3).float()
            # self.n_ho_dist = torch.randn(self.batch_size, self.cfg.msdf.num_grids).to(self.device).float()
            # self.ml_contact = torch.randn(self.batch_size, self.cfg.msdf.num_grids, self.cfg.msdf.kernel_size, self.cfg.msdf.kernel_size, self.cfg.msdf.kernel_size,
            #                               1 + self.hand_cse.shape[1]).to(self.device).float()
            # self.obj_pt_mask = torch.randint(0, 2, (self.batch_size, self.cfg.msdf.num_grids)).bool().to(self.device)
            # self.hand_vert_mask = torch.randint(0, 2, (self.batch_size, self.cfg.msdf.num_grids, 778)).bool().to(self.device)

        elif self.contact_unit == 'point':
        ## Calculate contacts
            obj_cmap, _, nn_idx = calculate_contact_capsule(self.hand_verts, self.hand_normals,
                                                            self.obj_verts, self.obj_normals)
            self.contact_map = obj_cmap.to(self.device).squeeze(-1)
            self.pmap = torch.as_tensor(self.hand_part_ids).to(self.device)[nn_idx].squeeze(-1)
            self.part_map = F.one_hot(self.pmap, 16).float()
            self.obj2hand_nn_idx = nn_idx.to(self.device)


    def load_from_batch_obj_only(self, batch, obj_template=None, obj_hulls=None):
        """
        Load only object-related data from batched data. Used for testing.
        No hand-related variables are loaded - those will be generated later.
        The batch size is 1: loading only one object each time.
        """
        self.obj_names = batch['objName']
        self.obj_verts = batch['objSamplePts'].clone().to(self.device).float()
        self.obj_normals = batch['objSampleNormals'].clone().to(self.device).float()

        # Load object templates
        if obj_template is not None:
            self.obj_models = []
            obj_mesh = copy(obj_template)
            self.obj_models.append(obj_mesh)

        # Load object hulls
        if obj_hulls is not None:
            self.obj_hulls = []
            ohs = []
            for h in obj_hulls:
                h0 = copy(h)
                ohs.append(h0)
            self.obj_hulls.append(ohs)

        # Center object at origin
        # self.obj_com = self.obj_verts.mean(dim=1, keepdim=True)
        # self.obj_verts = self.obj_verts - self.obj_com
        self.obj_msdf = batch['objMsdf'].clone().to(self.device).float()

    def get_99_dim_mano_params(self):
        """
        The mano parameters are: 
        3D translation (3) + 6D root rotation (6) + 6D hand pose (90) = 99D.
        """
        translation = self.hand_trans
        rot6d = matrix_to_rotation_6d(axis_angle_to_matrix(self.hand_root_rot))
        pose6d = matrix_to_rotation_6d(axis_angle_to_matrix(self.hand_pose.view(-1, 15, 3))).view(-1, 90)
        return torch.cat([translation, rot6d, pose6d], dim=-1)


    def _load_templates(self, idx, obj_templates, obj_hull=None):
        if self.hand_verts is not None:
            hand_mesh = trimesh.Trimesh(self.hand_verts[idx].detach().cpu().numpy(), self.closed_hand_faces.copy())
        else:
            hand_mesh = None

        self.hand_models = []
        self.obj_models = []
        for i in range(len(obj_templates)):
            if not self.normalize:
                objR = axis_angle_to_matrix(self.obj_rot[i]).detach().cpu().numpy()
                objt = self.obj_trans[i].detach().cpu().numpy()
                T = np.eye(4)
                T[:3, :3] = objR
                T[:3, 3] = objt
                obj_mesh = copy(obj_templates[i])
                # if self.hand_sides[idx] == 'left':
                #     flip_x_axis(obj_mesh)
                obj_mesh.apply_transform(T)
                obj_mesh.apply_translation(-self.obj_com[i, 0].detach().cpu().numpy())  # Move to zero-centered
                self.obj_models.append(obj_mesh)
            else:
                obj_mesh = copy(obj_templates[i])
                self.obj_models.append(obj_mesh)
                
        obj_mesh = self.obj_models[idx]

        if obj_hull is not None:
            ohs = []
            for h in obj_hull:
                h0 = copy(h)
                # if self.hand_sides[idx] == 'left':
                #     flip_x_axis(h0)
                # h0.apply_transform(T)
                ohs.append(h0)
        else:
            ohs = None

        return hand_mesh, obj_mesh, ohs

    def get_vis_geoms(self, idx=0, draw_maps=False, draw_hand=True, draw_obj=True, **kwargs):
        if 'obj_templates' in kwargs:
            hand_mesh, obj_mesh, _ = self._load_templates(idx=idx, obj_templates=kwargs['obj_templates'])
        else:
            hand_mesh = self.hand_models[idx]
            obj_mesh = self.obj_models[idx]

        # Downsample object mesh for visualization if it has too many faces
        if obj_mesh is not None and len(obj_mesh.faces) > 6000:
            obj_mesh = obj_mesh.simplify_quadric_decimation(face_count=6000)

        default_mat = vis.rendering.MaterialRecord()
        default_mat.shader = 'defaultLit'
        hand_mat = vis.rendering.MaterialRecord()
        hand_mat.shader = "defaultLitTransparency"
        hand_mat.base_color = [0.8, 0.7, 0.5, 0.8]
        obj_mat = vis.rendering.MaterialRecord()
        obj_mat.shader = "defaultLitTransparency"
        obj_mat.base_color = [0.5, 0.5, 0.5, 0.9]
        obj_mat1 = vis.rendering.MaterialRecord()
        obj_mat1.shader = "defaultLitTransparency"
        obj_mat1.base_color = [0.5, 0.9, 0.4, 0.7]
        heat_cmap = plt.colormaps['inferno']
        part_cmap = plt.colormaps['hsv']

        vis_geoms = []

        hand = o3dmesh_from_trimesh(hand_mesh, (0.8, 0.7, 0.5))
        hand_vcolor = part_cmap(self.hand_part_ids / 16)[:, :3] * 0.5 + 0.25
        hand.vertex_colors = o3d.utility.Vector3dVector(hand_vcolor)
        obj0 = o3dmesh_from_trimesh(obj_mesh, (0.5, 0.5, 0.5))
        if draw_hand:
            vis_geoms.append({'name': 'hand', 'geometry': hand, 'material': hand_mat})

        if draw_obj:
            vis_geoms.append({'name': 'object', 'geometry': obj0, 'material': obj_mat})

        if draw_maps:
            offset = np.array([0.0, 0, 0])
            comesh = copy(obj_mesh)
            comesh.apply_translation(offset)
            ## contact upscale
            dists = np.linalg.norm(self.obj_verts[idx].view(1, -1, 3).detach().cpu().numpy()
                                   - obj_mesh.vertices.reshape(-1, 1, 3), axis=-1)
            nn_idx = np.argmin(dists, axis=1)
            up_contact = self.contact_map[idx, nn_idx]
            up_contact = linear_normalize(up_contact, 0, 1).detach().cpu().numpy()
            vertex_colors = heat_cmap(up_contact)[:, :3]
            vertex_colors[up_contact <= 0.1] = 0.1
            comesh = o3dmesh_from_trimesh(comesh, (0.5, 0.5, 0.5))
            comesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            pomesh = copy(obj_mesh)
            pomesh.apply_translation(-offset)
            pomesh = o3dmesh_from_trimesh(pomesh)
            part_confs, part_ids = torch.max(self.part_map, dim=-1)
            up_part_ids = part_ids[idx, nn_idx].detach().cpu().numpy() / 16
            vertex_colors = part_cmap(up_part_ids)[:, :3]
            vertex_colors *= up_contact.reshape(-1, 1) # Regularize by the contact maps
            pomesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            vis_geoms.extend([{'name': 'obj_contacts', 'geometry': comesh, 'material': default_mat},
                              {'name': 'obj_parts', 'geometry': pomesh, 'material': default_mat}])

        return vis_geoms

    def vis_frame(self, idx=0, **kwargs):
        vis_geoms = self.get_vis_geoms(idx, **kwargs)
        o3d.visualization.draw(vis_geoms, show_skybox=False, lookat=[0, 1, 0], eye=[0, -1, 0], up=[0, 0, 1])
    
    def get_masked_recon_hand_meshes(self, recon_hand_verts, vis_idx=0):
        """
        get o3d meshes of reconstructed hand vertices. Only masked vertices are shown.
        :param recon_hand_verts: (B, 778, 3) reconstructed hand vertices
        :param vis_idx: int, index of the hand to visualize
        """
        hand_verts = recon_hand_verts[vis_idx].detach().cpu().numpy()
        hand_mask = self.hand_vert_mask[vis_idx].detach().cpu().numpy()
        geometries = extract_masked_mesh_components(hand_verts, self.closed_hand_faces, hand_mask,
                                                    self.hand_part_ids, create_geometries=True)
        return geometries
    

    def vis_recon_hand_with_object(self, recon_hand_verts, vis_idx=0, w=800, h=600):
        hand_geoms = self.get_masked_recon_hand_meshes(recon_hand_verts, vis_idx=vis_idx)
        obj_geom = o3dmesh_from_trimesh(self.obj_models[vis_idx], (0.5, 0.5, 0.5))
        img = geom_to_img(hand_geoms + [obj_geom], w=w, h=h, scale=0.6)
        return img


    def get_ho_features(self, rot_dim=3):
        """
        Obtain data directly used for training.
        The random rotations only affects the output of this function, not modifying the physics representation.
        """

        if self.hand_verts is not None:
            if rot_dim == 3:
                hand_pose = self.hand_pose
                root_rot = self.hand_root_rot
            elif rot_dim == 4:
                hand_pose = axis_angle_to_quaternion(self.hand_pose.view(-1, 15, 3)).view(-1, 60)
                root_rot = axis_angle_to_quaternion(self.hand_root_rot)
            elif rot_dim == 6:
                hand_pose = matrix_to_rotation_6d(axis_angle_to_matrix(self.hand_pose.view(-1, 15, 3))).view(-1, 90)
                root_rot = matrix_to_rotation_6d(axis_angle_to_matrix(self.hand_root_rot))
            elif rot_dim == 9:
                hand_pose = axis_angle_to_matrix(self.hand_pose.view(-1, 15, 3)).view(-1, 135)
                root_rot = axis_angle_to_matrix(self.hand_root_rot).view(-1, 9)
            else:
                raise NotImplementedError

            hand_params = {'handVerts': self.hand_verts, 'handPose': hand_pose,
                           'handTrans': self.hand_trans, 'handRot': root_rot}
        else:
            hand_params = None

        return self.obj_verts, self.obj_normals, hand_params

    def vis_img(self, idx:int, h:int=600, w:int=800, draw_maps=False, **kwargs) -> np.ndarray:
        """
        Visualize the hand-object as an image.
        pts: N x 3
        returns an image array of h x w x 3
        """
        vis_geoms = self.get_vis_geoms(idx, draw_maps=draw_maps, **kwargs)

        scale = 0.5
        # Separate geometries into three groups
        hand_obj_geoms = [g for g in vis_geoms if g['name'] in ['hand', 'object']]
        img_hand_obj = geom_to_img(hand_obj_geoms, w, h, scale=scale)

        if draw_maps:
            contact_geoms = [g for g in vis_geoms if g['name'] == 'obj_contacts']
            part_geoms = [g for g in vis_geoms if g['name'] == 'obj_parts']

            # Render each group separately
            img_contacts = geom_to_img(contact_geoms, w, h, scale=scale)
            img_parts = geom_to_img(part_geoms, w, h, scale=scale)
            ret_img = np.concatenate([img_hand_obj, img_contacts, img_parts], axis=0)
        else:
            ret_img = img_hand_obj

        # Concatenate vertically (along y axis)
        return ret_img


def recover_hand_verts_from_contact(handcse, gt_face_idx, grid_contact, grid_cse, grid_coords, mask_th=0.01):
    """
    :param grid_contact: (B, N) contact values
    :param grid_cse: (B, N, D) contact signature embeddings
    :param grid_coords: (B, N, 3)
    """
    targetWverts = handcse.emb2Wvert(grid_cse, gt_face_idx)
    # verts_mask = torch.sum(targetWverts, dim=1) > 0.01  # (B, 778)
    weight = (targetWverts * grid_contact.unsqueeze(-1)).transpose(-1, -2)  # (B, 778, K^3)
    verts_mask = torch.sum(weight, dim=-1) > mask_th  # (B, 778)
    weight[verts_mask] = weight[verts_mask] / torch.sum(weight[verts_mask], dim=-1, keepdim=True)
    pred_verts = weight @ grid_coords  # (B, 778, 3)
    return pred_verts, verts_mask