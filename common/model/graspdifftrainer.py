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

from .lgcdifftrainer import LGCDiffTrainer
from common.model.handobject import HandObject
from einops import rearrange


class GraspDiffTrainer(LGCDiffTrainer):
    """
    The Lightning trainer interface to train Local-grid based contact autoencoder.
    """
    def __init__(self, grid_ae, model, diffusion, cfg):
        super(GraspDiffTrainer, self).__init__(grid_ae=grid_ae, model=model, diffusion=diffusion, cfg=cfg)
    
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
        z = posterior.sample().view(batch_size, n_grids, -1) # n_dim
        obj_pc = torch.cat([obj_msdf_center, obj_feat.view(batch_size, n_grids, -1), z], dim=-1)

        ## Pad the shape params with 0
        betas = torch.zeros(batch_size, 10, device=self.device)
        mano_param_vec = handobject.get_99_dim_mano_params()  # B x 99
        mano_param_vec = torch.cat([mano_param_vec, betas], dim=-1) # B x 109

        input_data = {'x': mano_param_vec, 'obj_pc': obj_pc.permute(0, 2, 1)}

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