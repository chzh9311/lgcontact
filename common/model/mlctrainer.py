import os
import torch
import torch.nn as nn
import lightning as L
import trimesh
import open3d as o3d
import numpy as np

from common.manopth.manopth.manolayer import ManoLayer
from common.model.pose_optimizer import optimize_pose_wrt_local_grids
from common.model.handobject import HandObject
from common.model.hand_cse.hand_cse import HandCSE
from common.utils.vis import o3dmesh_from_trimesh

class MLCTrainer(L.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.mano_layer = ManoLayer(mano_root=cfg.data.mano_root, side='right',
                                    use_pca=False, ncomps=45, flat_hand_mean=True)
        cse_mat = torch.load(cfg.data.hand_cse_path)

        handF = self.mano_layer.th_faces
        # Initialize model and load state
        self.handcse = HandCSE(n_verts=778, emb_dim=cse_mat.shape[1], cano_faces=handF.cpu().numpy()).to(self.device)
        self.handcse.load_state_dict({'embedding_tensor': cse_mat.to(self.device),
                                      'cano_faces': handF.long().to(self.device)})
        self.handcse.eval()

    def training_step(self, batch, batch_idx):
        loss_dict = self.model.compute_loss(batch)
        total_loss = sum(loss_dict.values())
        self.log_dict(loss_dict, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        loss_dict = self.model.compute_loss(batch)
        total_loss = sum(loss_dict.values())
        self.log_dict({f'val_{k}': v for k, v in loss_dict.items()}, prog_bar=True)
        return total_loss
    
    def test_step(self, batch, batch_idx):
        ho_gt = HandObject(cfg = self.cfg.data, device=self.device, mano_layer=self.mano_layer)
        ho_gt.load_from_batch(batch)
        target_pts = ho_gt.normalized_coords[None, None, :, :] * self.cfg.msdf.scale + ho_gt.obj_msdf[:, :, None, self.cfg.msdf.kernel_size**3:]
        pred_ml_contact = self.model(ho_gt.obj_msdf, ho_gt.ml_contact)
        batch_size, n_centers = ho_gt.obj_msdf.shape[0], ho_gt.obj_msdf.shape[1]
        pred_contact, pred_target_emb = pred_ml_contact[..., 0], pred_ml_contact[..., 1:]
        pred_contact = pred_contact.view(batch_size, n_centers, -1) * ho_gt.obj_pt_mask.unsqueeze(-1).float()
        pred_targetWverts = self.handcse.emb2Wvert(pred_target_emb.view(batch_size, n_centers*self.cfg.msdf.kernel_size**3, -1))  # N_verts x N_verts

        ## For debugging
        # vis_idx = 0
        # obj_info = self.trainer.datamodule.test_set.obj_info[batch['objName'][vis_idx]]
        # obj_mesh = trimesh.Trimesh(vertices=obj_info['verts'], faces=obj_info['faces'], process=False)
        # hand_mesh = trimesh.Trimesh(vertices=ho_gt.hand_verts[vis_idx].cpu().numpy(), faces=ho_gt.hand_faces, process=False)
        # valid_pts = target_pts[vis_idx][pred_contact[vis_idx] > 0.5]
        # corr_mat = pred_targetWverts[vis_idx][pred_contact[vis_idx].flatten() > 0.5]
        # self.vis_relations(obj_mesh, hand_mesh, valid_pts.cpu().numpy(), corr_mat.cpu().numpy())
        # self.vis_contact(obj_mesh, hand_mesh, ho_gt.obj_msdf[vis_idx, :, -3:], ho_gt.obj_pt_mask[vis_idx])
        with torch.enable_grad():
            global_pose, mano_pose, mano_shape, mano_trans = optimize_pose_wrt_local_grids(
                                            self.mano_layer,
                                            target_pts.view(batch_size, -1, 3),
                                            pred_targetWverts,
                                            pred_contact.view(batch_size, -1),
                                            n_iter=self.cfg.pose_optimizer.n_opt_iter,
                                            lr=self.cfg.pose_optimizer.opt_lr
                                        )
        handV, handJ, _ = self.mano_layer(torch.cat([global_pose, mano_pose], dim=1),
                                          th_betas=mano_shape, th_trans=mano_trans)
        mpvpe = torch.mean(torch.norm(handV - ho_gt.hand_verts, dim=-1)) * 1000
        mpjpe = torch.mean(torch.norm(handJ - ho_gt.hand_joints, dim=-1)) * 1000
        print(f"MPVPE: {mpvpe.item():.2f} | MPJPE: {mpjpe.item():.2f}")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.training.lr)
        return optimizer
    
    @staticmethod
    def vis_contact(obj_mesh, hand_mesh, obj_pts, obj_pt_mask):
        hand_mesh = o3dmesh_from_trimesh(hand_mesh, color=[0.8, 0.7, 0.6])
        obj_mesh = o3dmesh_from_trimesh(obj_mesh, color=[0.7, 0.7, 0.7])

        # Convert obj_pts to numpy if it's a tensor
        if isinstance(obj_pts, torch.Tensor):
            obj_pts_np = obj_pts.cpu().numpy()
        else:
            obj_pts_np = obj_pts

        # Convert obj_pt_mask to numpy if it's a tensor
        if isinstance(obj_pt_mask, torch.Tensor):
            obj_pt_mask_np = obj_pt_mask.cpu().numpy()
        else:
            obj_pt_mask_np = obj_pt_mask

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_pts_np)

        # Color points: red for masked (True), blue for rest (False)
        colors = np.zeros((len(obj_pts_np), 3))
        colors[obj_pt_mask_np] = [1.0, 0.0, 0.0]  # Red for selected points
        colors[~obj_pt_mask_np] = [0.0, 0.0, 1.0]  # Blue for rest
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize
        o3d.visualization.draw_geometries([hand_mesh, obj_mesh, pcd])
    
    @staticmethod
    def vis_relations(obj_mesh, hand_mesh, contact_pts, relations):
        """
        Docstring for vis_relations

        :param obj_mesh: trimesh.Trimesh
        :param hand_mesh: trimesh.Trimesh
        :param contact_pts: N x 3
        :param relations: N x 778 the relation matrix between contact points and hand vertices
        """
        hand_mesh_o3d = o3dmesh_from_trimesh(hand_mesh, color=[0.8, 0.7, 0.6])
        obj_mesh = o3dmesh_from_trimesh(obj_mesh, color=[0.7, 0.7, 0.7])

        # Convert contact_pts to numpy if it's a tensor
        if isinstance(contact_pts, torch.Tensor):
            contact_pts_np = contact_pts.cpu().numpy()
        else:
            contact_pts_np = contact_pts

        # Convert relations to numpy if it's a tensor
        if isinstance(relations, torch.Tensor):
            relations_np = relations.cpu().numpy()
        else:
            relations_np = relations

        # Get hand vertices
        hand_verts = np.asarray(hand_mesh.vertices)

        # Create point cloud for contact points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(contact_pts_np)
        pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red contact points

        # Find the hand vertex with maximum weight for each contact point
        max_vert_indices = np.argmax(relations_np, axis=1)  # N indices

        # Create line set to connect contact points to their related hand vertices
        lines = [[i, len(contact_pts_np) + max_vert_indices[i]] for i in range(len(contact_pts_np))]

        # Combine contact points and hand vertices for line set
        all_points = np.vstack([contact_pts_np, hand_verts])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(all_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([1.0, 0.0, 0.0])  # Red lines

        # Visualize
        o3d.visualization.draw_geometries([hand_mesh_o3d, obj_mesh, pcd, line_set])


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, msdf, ml_contact):
        return ml_contact