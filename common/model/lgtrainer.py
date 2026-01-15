import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import trimesh
import open3d as o3d
import numpy as np

from common.manopth.manopth.manolayer import ManoLayer
from common.model.losses import kl_div_normal, masked_rec_loss
from common.model.handobject import recover_hand_verts_from_contact
from common.model.handobject import HandObject, recover_hand_verts_from_contact
from common.model.hand_cse.hand_cse import HandCSE
from common.utils.vis import o3dmesh_from_trimesh, visualize_local_grid_with_hand, geom_to_img, extract_masked_mesh_components
from common.model.losses import masked_rec_loss
from common.msdf.utils.msdf import get_grid


class LGTrainer(L.LightningModule):
    """
    The Lightning trainer interface to train Local-grid based contact autoencoder.
    """
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.debug = cfg.get('debug', False)
        self.loss_weights = cfg.train.loss_weights
        self.lr = cfg.train.lr
        self.mano_layer = ManoLayer(mano_root=cfg.data.mano_root, side='right',
                                    use_pca=False, ncomps=45, flat_hand_mean=True)
        cse_ckpt = torch.load(cfg.data.hand_cse_path)

        handF = self.mano_layer.th_faces
        # Initialize model and load state
        self.handcse = HandCSE(n_verts=778, emb_dim=cse_ckpt['emb_dim'], cano_faces=handF.cpu().numpy()).to(self.device)
        self.handcse.load_state_dict(cse_ckpt['state_dict'])
        self.handcse.eval()
        for param in self.handcse.parameters():
            param.requires_grad = False
        self.grid_coords = get_grid(self.cfg.msdf.kernel_size) * self.cfg.msdf.scale  # (K^3, 3)

    def training_step(self, batch, batch_idx):
        total_loss = self.train_val_step(batch, batch_idx, stage='train')
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        total_loss = self.train_val_step(batch, batch_idx, stage='val')
        return total_loss
    
    def train_val_step(self, batch, batch_idx, stage):
        self.grid_coords = self.grid_coords.to(self.device)
        grid_sdf, gt_grid_contact = batch['localGrid'][..., 0], batch['localGrid'][..., 1:]

        recon_cgrid, posterior, obj_feat = self.model(gt_grid_contact.permute(0, 4, 1, 2, 3), grid_sdf.unsqueeze(1))
        recon_cgrid = recon_cgrid.permute(0, 2, 3, 4, 1)
        contact, contact_hat = gt_grid_contact[..., 0], recon_cgrid[..., 0]
        cse, cse_hat = gt_grid_contact[..., 1:], recon_cgrid[..., 1:]

        batch_size = grid_sdf.shape[0]
        pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
            self.handcse, batch['face_idx'],
            contact_hat.reshape(batch_size, -1),
            cse_hat.reshape(batch_size, -1, cse.shape[-1]),
            grid_coords=self.grid_coords.view(1, -1, 3).repeat(batch_size, 1, 1),
            mask_th = 0.02
        )
        loss_dict = self.loss_net(gt_grid_contact, recon_cgrid, posterior, pred_hand_verts,
                                  batch['nHandVerts'], batch['handVertMask'], gt_face_idx=batch['face_idx'],
                                  gt_w=batch['cse_weights'], proc=stage)
        # recon_loss = F.mse_loss(recon_grid_contact, gt_grid_contact.permute(0, 4, 1, 2, 3))
        # loss_dict = {f'{stage}/embedding_loss': loss, f'{stage}/recon_loss': recon_loss, f'{stage}/perplexity': perplexity}
        # total_loss = sum(loss_dict.values())
        # total_loss = loss + self.recon_weight * recon_loss

        # Log losses - for validation, compute epoch average; for training, log per step
        if stage == 'val':
            self.log_dict(loss_dict, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        else:
            self.log_dict(loss_dict, prog_bar=False, sync_dist=True)
        if batch_idx % self.cfg[stage].vis_every_n_batches == 0:
            pred_grid_contact = recon_cgrid
            gt_geoms = self.visualize_grid_and_hand(
                grid_coords=self.grid_coords.view(-1, 3),
                grid_contact=gt_grid_contact[..., 0].reshape(batch_size, -1),
                pred_hand_verts=batch['nHandVerts'],
                hand_faces=self.mano_layer.th_faces,
                pred_mask=batch['handVertMask'],
                batch_idx=0
            )
            pred_geoms = self.visualize_grid_and_hand(
                grid_coords=self.grid_coords.view(-1, 3),
                grid_contact=pred_grid_contact[..., 0].reshape(batch_size, -1),
                pred_hand_verts=pred_hand_verts,
                hand_faces=self.mano_layer.th_faces,
                pred_mask=batch['handVertMask'],
                gt_mask=batch['handVertMask'],
                gt_hand_verts=batch['nHandVerts'],
                batch_idx=0
            )
            if self.debug:
                all_geoms = gt_geoms + [g.translate((self.cfg.msdf.scale * 3, 0, 0)) for g in pred_geoms]
                o3d.visualization.draw_geometries(all_geoms, window_name=f'{stage} Local Grid Visualization')
            else:
                gt_img = geom_to_img(gt_geoms, w=400, h=400)
                pred_img = geom_to_img(pred_geoms, w=400, h=400)
                img = np.concatenate([gt_img, pred_img], axis=0)
                # Log image - works for both WandbLogger and TensorBoardLogger
                if hasattr(self.logger, 'experiment'):
                    if hasattr(self.logger.experiment, 'add_image'):
                        # TensorBoardLogger
                        global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.{stage}_dataloader()')) + batch_idx
                        self.logger.experiment.add_image(f'{stage}/local_grid', img, global_step, dataformats='HWC')
                    elif hasattr(self.logger.experiment, 'log'):
                        # WandbLogger
                        import wandb
                        self.logger.experiment.log({f'{stage}/local_grid': wandb.Image(img)},
                                                commit=False)
        return loss_dict[f'{stage}/total_loss']
    
    def test_step(self, batch, batch_idx):
        self.grid_coords = self.grid_coords.to(self.device)
        grid_sdf, gt_grid_contact = batch['localGrid'][..., 0], batch['localGrid'][..., 1:]
        recon_grid_contact, z_e, obj_feat = self.model(gt_grid_contact.permute(0, 4, 1, 2, 3), grid_sdf.unsqueeze(1))
        recon_grid_contact = recon_grid_contact.permute(0, 2, 3, 4, 1)
        batch_size = grid_sdf.shape[0]

        gt_rec_hand_verts, gt_rec_verts_mask = recover_hand_verts_from_contact(
            self.handcse,
            gt_grid_contact[..., 0].view(batch_size, -1), gt_grid_contact[..., 1:].view(batch_size, -1, gt_grid_contact.shape[-1]-1),
            grid_coords=self.grid_coords.view(1, -1, 3).repeat(batch_size, 1, 1),
            mask_th = 0.02
        )
        gt_geoms = self.visualize_grid_and_hand(
            grid_coords=self.grid_coords.view(-1, 3),
            grid_contact=gt_grid_contact[..., 0].view(batch_size, -1),
            pred_hand_verts=gt_rec_hand_verts,
            hand_faces=self.mano_layer.th_faces,
            pred_mask=batch['handVertMask'],
            # pred_mask=gt_rec_verts_mask,
            gt_hand_verts=batch['nHandVerts'],
            gt_mask=batch['handVertMask'],
            batch_idx=0
        )

        gt_rec_error = masked_rec_loss(gt_rec_hand_verts, batch['nHandVerts'], gt_rec_verts_mask)
        pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
            self.handcse,
            recon_grid_contact[..., 0].reshape(batch_size, -1),
            recon_grid_contact[..., 1:].reshape(batch_size, -1, gt_grid_contact.shape[-1] - 1),
            grid_coords=self.grid_coords.view(1, -1, 3).repeat(batch_size, 1, 1),
            mask_th=0.02
        )
        pred_geoms = self.visualize_grid_and_hand(
            grid_coords=self.grid_coords.view(-1, 3),
            grid_contact=recon_grid_contact[..., 0].view(batch_size, -1),
            pred_hand_verts=pred_hand_verts,
            hand_faces=self.mano_layer.th_faces,
            pred_mask=batch['handVertMask'],
            # pred_mask=pred_verts_mask,
            gt_hand_verts=batch['nHandVerts'],
            gt_mask=batch['handVertMask'],
            batch_idx=0
        )
        all_geoms = gt_geoms + [g.translate((self.cfg.msdf.scale * 3, 0, 0)) for g in pred_geoms]
        o3d.visualization.draw_geometries(all_geoms, window_name='GT and Pred Local Grid Visualization')

        pred_rec_error = masked_rec_loss(pred_hand_verts, batch['nHandVerts'], pred_verts_mask)
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
    
    def loss_net(self, x, x_hat, posterior, pred_hand_verts, gt_hand_verts, gt_verts_mask, gt_face_idx, gt_w, proc='train'):
        """
        Compute the loss for training the GRIDAE
        1. Reconstruction loss between x and x_hat
        2. Regularization loss on z_e (KL-divergence)
        """
        contact, contact_hat = x[..., 0], x_hat[..., 0]
        cse, cse_hat = x[..., 1:], x_hat[..., 1:]
        contact_diff = contact - contact_hat
        cse_diff = (cse - cse_hat) * contact[..., None] # weighing by contact likelihood
        cse_value_loss = F.mse_loss(cse_diff, torch.zeros_like(cse_diff))
        cse_rec_loss = self.handcse.cse_rec_loss(cse_hat.reshape(x.shape[0], -1, cse.shape[-1]), gt_face_idx, gt_w)
        contact_loss = F.mse_loss(contact_diff, torch.zeros_like(contact_diff))
        # kl_loss = kl_div_normal(z_e)
        kl_loss = posterior.kl().mean()
        rec_loss = masked_rec_loss(pred_hand_verts, gt_hand_verts, gt_verts_mask)
        if self.cfg.train.rec_loss_start_epoch is not None and self.current_epoch >= self.cfg.train.rec_loss_start_epoch:
            w_rec = self.loss_weights.w_rec
        else:
            w_rec = 0.0

        loss_dict = {
            f'{proc}/contact_loss': contact_loss,
            f'{proc}/cse_rec_loss': cse_rec_loss,
            f'{proc}/cse_value_loss': cse_value_loss,
            f'{proc}/kl_loss': kl_loss,
            f'{proc}/rec_loss': rec_loss,
            f'{proc}/total_loss': self.loss_weights.w_contact * contact_loss + self.loss_weights.w_cse_rec * cse_rec_loss\
                + self.loss_weights.w_cse_value * cse_value_loss + self.loss_weights.w_kl * kl_loss + w_rec * rec_loss
        }
        return loss_dict
    
    
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
            grid_coords_np = grid_coords.detach().cpu().numpy()
        else:
            grid_coords_np = grid_coords

        if isinstance(grid_contact, torch.Tensor):
            grid_contact_np = grid_contact[batch_idx].detach().cpu().numpy()
        else:
            grid_contact_np = grid_contact[batch_idx]

        if isinstance(pred_hand_verts, torch.Tensor):
            pred_hand_verts_np = pred_hand_verts[batch_idx].detach().cpu().numpy()
        else:
            pred_hand_verts_np = pred_hand_verts[batch_idx]

        if isinstance(hand_faces, torch.Tensor):
            hand_faces_np = hand_faces.detach().cpu().numpy()
        else:
            hand_faces_np = hand_faces

        if isinstance(pred_mask, torch.Tensor):
            pred_mask_np = pred_mask[batch_idx].detach().cpu().numpy()
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

        # 2. Create predicted masked hand mesh and isolated vertices
        pred_geometries = extract_masked_mesh_components(
            pred_hand_verts_np, hand_faces_np, pred_mask_np,
            create_geometries=True,
            mesh_color=[0.8, 0.6, 0.4],  # Skin color for prediction
            isolated_color=[1.0, 0.0, 0.0]  # Red for isolated points
        )
        geometries.extend(pred_geometries)

        # 4. Visualize GT hand if provided
        if gt_hand_verts is not None and gt_mask is not None:
            # Convert GT tensors to numpy
            if isinstance(gt_hand_verts, torch.Tensor):
                gt_hand_verts_np = gt_hand_verts[batch_idx].detach().cpu().numpy()
            else:
                gt_hand_verts_np = gt_hand_verts[batch_idx]

            if isinstance(gt_mask, torch.Tensor):
                gt_mask_np = gt_mask[batch_idx].detach().cpu().numpy()
            else:
                gt_mask_np = gt_mask[batch_idx]

            # Create GT masked hand mesh and isolated vertices
            gt_geometries = extract_masked_mesh_components(
                gt_hand_verts_np, hand_faces_np, gt_mask_np,
                create_geometries=True,
                mesh_color=[0.4, 0.8, 0.4],  # Green color for GT
                isolated_color=[0.0, 1.0, 0.0]  # Green for isolated GT points
            )
            geometries.extend(gt_geometries)

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