import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import trimesh
import open3d as o3d
import numpy as np

from common.manopth.manopth.manolayer import ManoLayer
from common.model.pose_optimizer import optimize_pose_wrt_local_grids
from common.model.handobject import HandObject
from common.model.hand_cse.hand_cse import HandCSE
from common.utils.vis import o3dmesh_from_trimesh, visualize_local_grid_with_hand, geom_to_img
from common.msdf.utils.msdf import get_grid

class LGTrainer(L.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.lr = cfg.train.lr
        self.recon_weight = cfg.train.get('recon_weight', 100)
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
        total_loss = self.train_val_step(batch, batch_idx, stage='train')
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        total_loss = self.train_val_step(batch, batch_idx, stage='val')
        return total_loss
    
    def train_val_step(self, batch, batch_idx, stage):
        grid_sdf, gt_grid_contact = batch['localGrid'][..., 0], batch['localGrid'][..., 1:]
        loss, recon_grid_contact, perplexity = self.model(gt_grid_contact.permute(0, 4, 1, 2, 3))
        recon_loss = F.mse_loss(recon_grid_contact, gt_grid_contact.permute(0, 4, 1, 2, 3))
        loss_dict = {f'{stage}/embedding_loss': loss, f'{stage}/recon_loss': recon_loss, f'{stage}/perplexity': perplexity}
        # total_loss = sum(loss_dict.values())
        total_loss = loss + self.recon_weight * recon_loss
        self.log_dict(loss_dict, prog_bar=True)
        return total_loss
    
    def test_step(self, batch, batch_idx):
        grid_sdf, gt_grid_contact = batch['localGrid'][..., 0], batch['localGrid'][..., 1:]
        loss, recon_grid_contact, perplexity = self.model(gt_grid_contact.permute(0, 4, 1, 2, 3))
        recon_grid_contact = recon_grid_contact.permute(0, 2, 3, 4, 1)
        batch_size = grid_sdf.shape[0]
        def rec_error(pred, gt, mask):
            return torch.mean((torch.norm(pred - gt, dim=-1)) / mask.sum(dim=1, keepdim=True)) * 1000 # in mm

        grid_coords = get_grid(self.cfg.msdf.kernel_size, device=grid_sdf.device) * self.cfg.msdf.scale  # (K^3, 3)
        gt_rec_hand_verts, gt_rec_verts_mask = self.recover_hand_verts_from_contact(
            gt_grid_contact[..., 0].view(batch_size, -1), gt_grid_contact[..., 1:].view(batch_size, -1, gt_grid_contact.shape[-1]-1),
            grid_coords=grid_coords.view(-1, 3)
        )
        gt_geoms = self.visualize_grid_and_hand(
            grid_coords=grid_coords.view(-1, 3),
            grid_contact=gt_grid_contact[..., 0].view(batch_size, -1),
            pred_hand_verts=gt_rec_hand_verts,
            hand_faces=self.mano_layer.th_faces,
            pred_mask=gt_rec_verts_mask,
            gt_hand_verts=batch['nHandVerts'],
            gt_mask=gt_rec_verts_mask,
            batch_idx=0
        )

        gt_rec_error = rec_error(gt_rec_hand_verts, batch['nHandVerts'], gt_rec_verts_mask)

        pred_hand_verts, pred_verts_mask = self.recover_hand_verts_from_contact(
            recon_grid_contact[..., 0].reshape(batch_size, -1),
            recon_grid_contact[..., 1:].reshape(batch_size, -1, gt_grid_contact.shape[-1] - 1),
            grid_coords=grid_coords.view(-1, 3)
        )
        pred_geoms = self.visualize_grid_and_hand(
            grid_coords=grid_coords.view(-1, 3),
            grid_contact=recon_grid_contact[..., 0].view(batch_size, -1),
            pred_hand_verts=pred_hand_verts,
            hand_faces=self.mano_layer.th_faces,
            pred_mask=pred_verts_mask,
            gt_hand_verts=batch['nHandVerts'],
            gt_mask=gt_rec_verts_mask,
            batch_idx=0
        )
        all_geoms = gt_geoms + [g.translate((self.cfg.msdf.scale * 3, 0, 0)) for g in pred_geoms]
        o3d.visualization.draw_geometries(all_geoms, window_name='GT and Pred Local Grid Visualization')

        pred_rec_error = rec_error(pred_hand_verts, batch['nHandVerts'], pred_verts_mask)
        loss_dict = {'test/gt_rec_error': gt_rec_error,
                     'test/pred_rec_error': pred_rec_error}
        self.log_dict(loss_dict, prog_bar=True)

        ## GT visualization
        # vis_idx = 0
        # gt_geoms = visualize_local_grid_with_hand(
        #         batch['localGrid'][vis_idx].cpu().numpy(), hand_verts=batch['nHandVerts'][vis_idx].cpu().numpy(),
        #         hand_faces=self.mano_layer.th_faces.cpu().numpy(), hand_cse=self.handcse.embedding_tensor.detach().cpu().numpy(),
        #         kernel_size=self.cfg.msdf.kernel_size, grid_scale=self.cfg.msdf.scale
        #     )
        # o3d.visualization.draw_geometries(gt_geoms, window_name='GT Local Grid Visualization')

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
    
    @staticmethod
    def visualize_grid_and_hand(grid_coords, grid_contact, pred_hand_verts, hand_faces, pred_mask,
                                 batch_idx=0, gt_hand_verts=None, gt_mask=None):
        """
        Visualize grid points colored by contact likelihood and masked hand mesh.

        Args:
            grid_coords: (K^3, 3) grid coordinates in world space (same for all batches)
            grid_contact: (B, K^3) contact values between 0 and 1
            pred_hand_verts: (B, H, 3) predicted hand vertices
            hand_faces: (F, 3) hand face indices
            pred_mask: (B, H) boolean mask for predicted hand vertices
            batch_idx: which sample in the batch to visualize (default: 0)
            gt_hand_verts: (B, H, 3) optional ground truth hand vertices
            gt_mask: (B, H) optional boolean mask for GT hand vertices

        Returns:
            list: List of open3d geometries (point cloud for grid, mesh/points for hand)
        """
        # Convert tensors to numpy if needed
        if isinstance(grid_coords, torch.Tensor):
            grid_coords_np = grid_coords.cpu().numpy()
        else:
            grid_coords_np = grid_coords

        if isinstance(grid_contact, torch.Tensor):
            grid_contact_np = grid_contact[batch_idx].cpu().numpy()
        else:
            grid_contact_np = grid_contact[batch_idx]

        if isinstance(pred_hand_verts, torch.Tensor):
            pred_hand_verts_np = pred_hand_verts[batch_idx].cpu().numpy()
        else:
            pred_hand_verts_np = pred_hand_verts[batch_idx]

        if isinstance(hand_faces, torch.Tensor):
            hand_faces_np = hand_faces.cpu().numpy()
        else:
            hand_faces_np = hand_faces

        if isinstance(pred_mask, torch.Tensor):
            pred_mask_np = pred_mask[batch_idx].cpu().numpy()
        else:
            pred_mask_np = pred_mask[batch_idx]

        geometries = []

        # 1. Create grid point cloud with inferno colormap based on contact values
        grid_pcd = o3d.geometry.PointCloud()
        grid_pcd.points = o3d.utility.Vector3dVector(grid_coords_np)

        # Apply inferno colormap to contact values
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('inferno')
        grid_colors = np.array([cmap(val)[:3] for val in grid_contact_np])
        grid_pcd.colors = o3d.utility.Vector3dVector(grid_colors)
        geometries.append(grid_pcd)

        # 2. Create predicted masked hand mesh
        # Find faces where all vertices are masked
        face_mask = pred_mask_np[hand_faces_np].all(axis=1)  # (F,) boolean array
        masked_faces = hand_faces_np[face_mask]

        if len(masked_faces) > 0:
            # Create mesh with only masked faces
            pred_hand_mesh = o3d.geometry.TriangleMesh()
            pred_hand_mesh.vertices = o3d.utility.Vector3dVector(pred_hand_verts_np)
            pred_hand_mesh.triangles = o3d.utility.Vector3iVector(masked_faces)
            pred_hand_mesh.paint_uniform_color([0.8, 0.6, 0.4])  # Skin color for prediction
            pred_hand_mesh.compute_vertex_normals()
            geometries.append(pred_hand_mesh)

        # 3. Find isolated vertices (masked but not in any face) for prediction
        vertices_in_faces = np.unique(masked_faces.flatten()) if len(masked_faces) > 0 else np.array([])
        masked_vert_indices = np.where(pred_mask_np)[0]
        isolated_vert_indices = np.setdiff1d(masked_vert_indices, vertices_in_faces)

        if len(isolated_vert_indices) > 0:
            # Visualize isolated vertices as points
            isolated_pcd = o3d.geometry.PointCloud()
            isolated_pcd.points = o3d.utility.Vector3dVector(pred_hand_verts_np[isolated_vert_indices])
            isolated_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red for isolated points
            geometries.append(isolated_pcd)

        # 4. Visualize GT hand if provided
        if gt_hand_verts is not None and gt_mask is not None:
            # Convert GT tensors to numpy
            if isinstance(gt_hand_verts, torch.Tensor):
                gt_hand_verts_np = gt_hand_verts[batch_idx].cpu().numpy()
            else:
                gt_hand_verts_np = gt_hand_verts[batch_idx]

            if isinstance(gt_mask, torch.Tensor):
                gt_mask_np = gt_mask[batch_idx].cpu().numpy()
            else:
                gt_mask_np = gt_mask[batch_idx]

            # Find faces where all vertices are masked for GT
            gt_face_mask = gt_mask_np[hand_faces_np].all(axis=1)
            gt_masked_faces = hand_faces_np[gt_face_mask]

            if len(gt_masked_faces) > 0:
                # Create GT mesh with only masked faces
                gt_hand_mesh = o3d.geometry.TriangleMesh()
                gt_hand_mesh.vertices = o3d.utility.Vector3dVector(gt_hand_verts_np)
                gt_hand_mesh.triangles = o3d.utility.Vector3iVector(gt_masked_faces)
                gt_hand_mesh.paint_uniform_color([0.4, 0.8, 0.4])  # Green color for GT
                gt_hand_mesh.compute_vertex_normals()
                geometries.append(gt_hand_mesh)

            # Find isolated vertices for GT
            gt_vertices_in_faces = np.unique(gt_masked_faces.flatten()) if len(gt_masked_faces) > 0 else np.array([])
            gt_masked_vert_indices = np.where(gt_mask_np)[0]
            gt_isolated_vert_indices = np.setdiff1d(gt_masked_vert_indices, gt_vertices_in_faces)

            if len(gt_isolated_vert_indices) > 0:
                # Visualize isolated GT vertices as points
                gt_isolated_pcd = o3d.geometry.PointCloud()
                gt_isolated_pcd.points = o3d.utility.Vector3dVector(gt_hand_verts_np[gt_isolated_vert_indices])
                gt_isolated_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # Green for isolated GT points
                geometries.append(gt_isolated_pcd)

        return geometries

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
    
    def recover_hand_verts_from_contact(self, grid_contact, grid_cse, grid_coords):
        """
        :param grid_contact: (B, K^3) contact values
        :param grid_cse: (B, K^3, D) contact signature embeddings
        :param grid_coords: (K^3, 3)
        """
        targetWverts = self.handcse.emb2Wvert(grid_cse)
        # verts_mask = torch.sum(targetWverts, dim=1) > 0.01  # (B, 778)
        weight = (targetWverts * grid_contact.unsqueeze(-1)).transpose(-1, -2)  # (B, 778, K^3)
        verts_mask = torch.sum(weight, dim=-1) > 0.01  # (B, 778)
        weight[verts_mask] = weight[verts_mask] / torch.sum(weight[verts_mask], dim=-1, keepdim=True)
        pred_verts = weight @ grid_coords.unsqueeze(0)  # (B, 778, 3)
        return pred_verts, verts_mask