import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import open3d as o3d
import trimesh
from copy import copy
import numpy as np
# from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import Pool
import wandb
from matplotlib import pyplot as plt
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

from .lgcdifftrainer import LGCDiffTrainer
from common.model.handobject import HandObject
from common.utils.vis import o3dmesh, o3dmesh_from_trimesh, geom_to_img
from einops import rearrange


class GraspDiffTrainer(LGCDiffTrainer):
    """
    The Lightning trainer interface to train Local-grid based contact autoencoder.
    """
    def __init__(self, grid_ae, hand_ae, model, diffusion, cfg):
        super(GraspDiffTrainer, self).__init__(grid_ae=grid_ae, model=model, diffusion=diffusion, cfg=cfg)
        self.hand_ae = hand_ae
        if cfg.run_phase == 'train':
            self._load_pretrained_weights(cfg.hand_ae.get('pretrained_weight', None), target_prefix='hand_ae')
    
    def on_fit_start(self):
        # Set grid_ae BatchNorm layers to eval mode to prevent running stats drift
        super(GraspDiffTrainer, self).on_fit_start()
        bn_count = 0
        for module in self.hand_ae.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.eval()
                # Disable running stats updates during forward pass
                module.track_running_stats = False
                bn_count += 1

        print(f"Set {bn_count} BatchNorm layers to eval mode (prevents running stats drift)")
    
    def train_val_step(self, batch, batch_idx, stage):
        self.grid_coords = self.grid_coords.view(-1, 3).to(self.device)
        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer)
        handobject.load_from_batch(batch, pool=self.pool)
        lg_contact = handobject.ml_contact
        batch_size, n_grids = lg_contact.shape[:2]
        obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, 1, self.msdf_k, self.msdf_k, self.msdf_k)
        # n_ho_dist = handobject.n_ho_dist

        obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:] # B x 3
        ## First process all grids separately using GRIDAE
        flat_lg_contact = rearrange(lg_contact, 'b n k1 k2 k3 c -> b n (k1 k2 k3) c')
        lg_contact = rearrange(lg_contact, 'b n k1 k2 k3 c -> (b n) c k1 k2 k3')
        posterior, obj_feat, multi_scale_obj_cond = self.grid_ae.encode(lg_contact, obj_msdf)
        # z = torch.cat([n_ho_dist.unsqueeze(-1), posterior.sample().view(batch_size, n_grids, -1)], dim=-1) # n_dim + 1
        gt_contact_latent = posterior.sample().view(batch_size, n_grids, -1) # n_dim
        obj_pc = torch.cat([obj_msdf_center, obj_feat.view(batch_size, n_grids, -1)], dim=-1)
        gt_hand_latent = self.hand_ae.hand2latent(handobject.hand_verts, handobject.hand_root_rot, handobject.hand_trans) # B x hand_latent_dim

        ## Debug the hand data
        # sbj_id = batch['sbjId'][0]
        # hmodel = self.trainer.datamodule.train_set.rh_models[sbj_id].to(self.device)
        # full_pose = torch.cat([handobject.hand_root_rot, handobject.hand_pose], dim=1)
        # handV, handJ, _ = hmodel(full_pose, th_trans=handobject.hand_trans)
        # hmodel.to('cpu')
        # print(torch.norm(handV[0] - handobject.hand_verts[0], dim=-1).max())

        ## Pad the shape params with 0
        # betas = torch.zeros(batch_size, 10, device=self.device)
        # mano_param_vec = handobject.get_99_dim_mano_params()  # B x 99
        # mano_param_vec = torch.cat([mano_param_vec, betas], dim=-1) # B x 109
        cat_latent = torch.cat([gt_hand_latent, gt_contact_latent.view(batch_size, -1)], dim=-1) # B x (n_grids*n_dim + hand_latent_dim)

        input_data = {'x': cat_latent, 'obj_pc': obj_pc.permute(0, 2, 1)}

        # Check for NaN in input data
        for key, value in input_data.items():
            if torch.isnan(value).any():
                raise ValueError(f"NaN detected in input_data['{key}'] at batch_idx {batch_idx}")

        # grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords.view(-1, 3)[None, None, :, :]  # B x N x K^3 x 3
        losses = self.diffusion.training_losses(self.model, input_data, grid_ae=self.grid_ae, ms_obj_cond=multi_scale_obj_cond,
                                              hand_cse=self.hand_cse, msdf_k=self.msdf_k, grid_scale=self.msdf_scale,
                                              gt_lg_contact=flat_lg_contact,
                                              adj_pt_indices=batch['adjPointIndices'],
                                              adj_pt_distances=batch['adjPointDistances'])
        total_loss = sum([losses[k] * self.loss_weights[k] for k in losses.keys()])
        losses['total_loss'] = total_loss
        # grid_loss = F.mse_loss(err[:, :, 0], torch.zeros_like(err[:, :, 0]), reduction='mean')  # only compute loss on n_ho_dist dimension
        # diff_loss = F.mse_loss(err[:, :, 1:], torch.zeros_like(err[:, :, 1:]), reduction='none')
        # obj_pt_mask = input_data['x'][:, :, 0:1] < 0
        # diff_loss = torch.sum(diff_loss * obj_pt_mask) / (torch.sum(obj_pt_mask) + 1e-6) / diff_loss.shape[2]

        loss_dict = {f'{stage}/{k}': v for k, v in losses.items()}
        if stage == 'val':
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        else:
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True)

        if batch_idx % self.cfg[stage].vis_every_n_batches == 0:
            condition = self.model.condition({'obj_pc': input_data['obj_pc']})
            samples = self.diffusion.p_sample_loop(self.model, input_data['x'].shape, condition, clip_denoised=False)
            # samples = self.diffusion.p_sample_loop(self.model, input_data['x'].shape, input_data['obj_pc'])
            vis_idx = 0
            simp_obj_mesh = getattr(self.trainer.datamodule, f'{stage}_set').simp_obj_mesh
            rot = batch['aug_rot'][vis_idx].cpu().numpy() if 'aug_rot' in batch else np.eye(3)
            obj_templates = [trimesh.Trimesh(simp_obj_mesh[name]['verts'] @ rot.T, simp_obj_mesh[name]['faces'])
                            for i, name in enumerate(batch['objName'])]
            # sample_latent = samples[:, 0] ## 1 + latent
            hand_latent, grid_latent = samples[:, :self.cfg.generator.unet.d_y], samples[:, self.cfg.generator.unet.d_y:].view(batch_size, n_grids, -1) ## latent
            # grid_contact, sample_latent = sample_x[:, :, 0], sample_x[:, :, 1:]
            # grid_contact = grid_contact < 0
            pred_hand_verts, pred_verts_mask, pred_grid_contact = self.reconstruct_from_latent(grid_latent, multi_scale_obj_cond, obj_msdf_center)

            gt_rec_hand_verts, gt_rec_verts_mask, gt_rec_grid_contact = self.reconstruct_from_latent(gt_contact_latent, multi_scale_obj_cond, obj_msdf_center)

            contact_img = self._visualize_contact_comparison(
                obj_msdf_center, pred_grid_contact, gt_rec_grid_contact, handobject, obj_templates, vis_idx)
            img = self._visualize_hand_comparison(
                pred_hand_verts, pred_verts_mask, gt_rec_hand_verts, gt_rec_verts_mask,
                handobject, obj_templates, obj_msdf_center, rot, vis_idx)
            # plt.imsave(f'tmp/contact_vis_{batch_idx}.png', contact_img)
            # plt.imsave(f'tmp/rec_img_{batch_idx}.png', img)
            recon_handV, recon_handJ = self.hand_ae.latent2hand(hand_latent)
            full_hand_img = self.visualize_full_hand_comparison(recon_handV[vis_idx], handobject.hand_verts[vis_idx], obj_templates[vis_idx])

            if hasattr(self.logger, 'experiment'):
                if hasattr(self.logger.experiment, 'add_image'):
                    # TensorBoardLogger
                    global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.{stage}_dataloader()')) + batch_idx
                    self.logger.experiment.add_image(f'{stage}/GT_vs_GTRec_vs_sampled_contact', contact_img, global_step, dataformats='HWC')
                    self.logger.experiment.add_image(f'{stage}/GT_vs_GTRec_vs_sampled_hand', img, global_step, dataformats='HWC')
                    self.logger.experiment.add_image(f'{stage}/GT_vs_sampled_full_hand', full_hand_img, global_step, dataformats='HWC')
                elif hasattr(self.logger.experiment, 'log'):
                    # WandbLogger
                    import wandb
                    self.logger.experiment.log({f'{stage}/GT_vs_GTRec_vs_sampled_contact': wandb.Image(contact_img)}, step=self.global_step)
                    self.logger.experiment.log({f'{stage}/GT_vs_GTRec_vs_sampled_hand': wandb.Image(img)}, step=self.global_step)
                    self.logger.experiment.log({f'{stage}/GT_vs_sampled_full_hand': wandb.Image(full_hand_img)}, step=self.global_step)
        return total_loss
    
    def visualize_full_hand_comparison(self, pred_handV, gt_handV, obj_template):
        hand_faces = self.mano_layer.th_faces.detach().cpu().numpy()
        pred_handV = pred_handV.detach().cpu().numpy()
        gt_handV = gt_handV.detach().cpu().numpy()
        pred_mesh = o3dmesh(pred_handV, hand_faces, color=[0.8, 0.7, 0.6])
        gt_mesh = o3dmesh(gt_handV, hand_faces, color=[0.2, 0.4, 0.8])
        obj_mesh = o3dmesh_from_trimesh(obj_template, color=[0.7, 0.7, 0.7])
        pred_img = geom_to_img([pred_mesh, obj_mesh], w=400, h=400, scale=0.5, half_range=0.12)
        gt_img = geom_to_img([gt_mesh, obj_mesh], w=400, h=400, scale=0.5, half_range=0.12)
        return np.concatenate([gt_img, pred_img], axis=0)