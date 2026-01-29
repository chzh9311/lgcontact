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

from common.manopth.manopth.manolayer import ManoLayer
from common.model.handobject import recover_hand_verts_from_contact
from common.model.pose_optimizer import optimize_pose_wrt_local_grids
from common.model.handobject import HandObject, recover_hand_verts_from_contact
from common.model.hand_cse.hand_cse import HandCSE
from common.utils.vis import visualize_recon_hand_w_object, visualize_grid_contact
from common.msdf.utils.msdf import get_grid
from common.evaluation.eval_fns import calculate_metrics, calc_diversity


class LGCDiffTrainer(L.LightningModule):
    """
    The Lightning trainer interface to train Local-grid based contact autoencoder.
    """
    def __init__(self, grid_ae, model, diffusion, cfg):
        super().__init__()
        self.grid_ae = grid_ae
        self.model = model
        self.diffusion = diffusion
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.debug = cfg.get('debug', False)
        self.msdf_k = cfg.msdf.kernel_size
        self.lr = cfg.train.lr

        ## Load and freeze autoencoder pretrained weights
        if cfg.run_phase == 'train':
            self._load_ae_pretrained_weights(cfg.generator.get('ae_pretrained_weight', None))
            self._freeze_pretrained_weights()

        self.mano_layer = ManoLayer(mano_root=cfg.data.mano_root, side='right',
                                    use_pca=True, ncomps=cfg.pose_optimizer.ncomps, flat_hand_mean=True)
        self.closed_mano_faces = np.load(osp.join('data', 'misc', 'closed_mano_r_faces.npy'))
        cse_ckpt = torch.load(cfg.data.hand_cse_path)

        handF = self.mano_layer.th_faces
        # Initialize model and load state
        self.cse_dim = cse_ckpt['emb_dim']
        self.hand_cse = HandCSE(n_verts=778, emb_dim=self.cse_dim, cano_faces=handF.cpu().numpy()).to(self.device)
        self.hand_cse.load_state_dict(cse_ckpt['state_dict'])
        self.hand_cse.eval()
        self.grid_coords = get_grid(self.cfg.msdf.kernel_size) * self.cfg.msdf.scale  # (K^3, 3)
        self.pool = None

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                batch[key] = value.float()
        return batch

    def _load_ae_pretrained_weights(self, checkpoint_path):
        """
        Load pretrained autoencoder weights from a Lightning checkpoint.
        Only loads weights for layers that exist in both the checkpoint and current model.

        Args:
            checkpoint_path: Path to the Lightning checkpoint file
        """
        import os

        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint file not found at {checkpoint_path}. Skipping weight initialization.")
            return

        print(f"Loading pretrained autoencoder weights from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract state dict from Lightning checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Filter state dict to only include grid_ae parameters
        # Lightning saves with 'model.' prefix, we need to extract grid_ae weights
        ae_state_dict = {}
        for key, value in state_dict.items():
            # Look for keys like 'model.obj_encoder.xxx' or 'model.encoder.xxx' or 'model.decoder.xxx'
            if key.startswith('model.'):
                # Remove 'model.' prefix to get the actual model key
                model_key = key[6:]  # Remove 'model.'
                # Add 'grid_ae.' prefix to match our model structure
                new_key = f'grid_ae.{model_key}'
                ae_state_dict[new_key] = value

        # Get current model state dict
        current_state_dict = self.state_dict()

        # Filter to only load weights that exist in current model (handle extra layers)
        filtered_state_dict = {}
        for key, value in ae_state_dict.items():
            if key in current_state_dict:
                # Check if shapes match
                if current_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"Warning: Shape mismatch for {key}. "
                          f"Checkpoint: {value.shape}, Current: {current_state_dict[key].shape}. Skipping.")
            else:
                print(f"Info: {key} in checkpoint but not in current model. Skipping.")

        # Load the filtered state dict
        missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)

        # Track which keys were successfully loaded
        self.pretrained_keys = list(filtered_state_dict.keys())

        # Count only missing keys in grid_ae
        missing_grid_ae_keys = [k for k in missing_keys if k.startswith('grid_ae.')]

        print(f"Successfully loaded {len(filtered_state_dict)} layers from pretrained checkpoint")
        if missing_grid_ae_keys:
            print(f"Missing grid_ae keys: {len(missing_grid_ae_keys)} keys")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")
    
    def _freeze_pretrained_weights(self):
        """
        Freeze all parameters that were loaded from the pretrained checkpoint.
        Also sets BatchNorm layers to eval mode to prevent running stats updates.
        """
        frozen_count = 0
        for name, param in self.named_parameters():
            if name in self.pretrained_keys:
                param.requires_grad = False
                frozen_count += 1

        # Set grid_ae BatchNorm layers to eval mode to prevent running stats drift
        bn_count = 0
        for module in self.grid_ae.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.eval()
                # Disable running stats updates during forward pass
                module.track_running_stats = False
                bn_count += 1

        print(f"Frozen {frozen_count} pretrained parameters in grid_ae")
        print(f"Set {bn_count} BatchNorm layers to eval mode (prevents running stats drift)")

    def training_step(self, batch, batch_idx):
        total_loss = self.train_val_step(batch, batch_idx, stage='train')
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        total_loss = self.train_val_step(batch, batch_idx, stage='val')
        return total_loss
    
    def train_val_step(self, batch, batch_idx, stage):
        self.grid_coords = self.grid_coords.view(-1, 3).to(self.device)
        obj_names = batch['objName']
        # simp_obj_mesh = getattr(self.trainer.datamodule, f'train_set').simp_obj_mesh
        # obj_templates = [trimesh.Trimesh(simp_obj_mesh[name]['verts'], simp_obj_mesh[name]['faces'])
        #                  for name in obj_names]
        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer)
        handobject.load_from_batch(batch, pool=self.pool)
        lg_contact = handobject.ml_contact
        batch_size, n_grids = lg_contact.shape[:2]
        obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, 1, self.msdf_k, self.msdf_k, self.msdf_k)
        # n_ho_dist = handobject.n_ho_dist

        obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:] # B x 3
        ## First process all grids separately using GRIDAE
        lg_contact = lg_contact.view(-1, self.msdf_k, self.msdf_k, self.msdf_k, lg_contact.shape[-1]).permute(0, 4, 1, 2, 3)
        posterior, obj_feat, multi_scale_obj_cond = self.grid_ae.encode(lg_contact, obj_msdf)
        obj_pc = torch.cat([obj_msdf_center, obj_feat.view(batch_size, n_grids, -1)], dim=-1)
        # z = torch.cat([n_ho_dist.unsqueeze(-1), posterior.sample().view(batch_size, n_grids, -1)], dim=-1) # n_dim + 1
        z = posterior.sample().view(batch_size, n_grids, -1) # n_dim

        input_data = {'x': z, 'obj_pc': obj_pc.permute(0, 2, 1)}

        # Check for NaN in input data
        for key, value in input_data.items():
            if torch.isnan(value).any():
                raise ValueError(f"NaN detected in input_data['{key}'] at batch_idx {batch_idx}")

        loss = self.diffusion.training_losses(self.model, input_data, grid_ae=self.grid_ae, ms_obj_cond=multi_scale_obj_cond,
                                              hand_cse=self.hand_cse)
        # grid_loss = F.mse_loss(err[:, :, 0], torch.zeros_like(err[:, :, 0]), reduction='mean')  # only compute loss on n_ho_dist dimension
        # diff_loss = F.mse_loss(err[:, :, 1:], torch.zeros_like(err[:, :, 1:]), reduction='none')
        # obj_pt_mask = input_data['x'][:, :, 0:1] < 0
        # diff_loss = torch.sum(diff_loss * obj_pt_mask) / (torch.sum(obj_pt_mask) + 1e-6) / diff_loss.shape[2]

        # loss_dict = {f'{stage}/grid_contact_loss': grid_loss,
        #              f'{stage}/latent_diff_loss': diff_loss,
        #              f'{stage}/total_loss': grid_loss + diff_loss}
        loss_dict = {f'{stage}/total_loss': loss}
        if stage == 'val':
            self.log_dict(loss_dict, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        else:
            self.log_dict(loss_dict, prog_bar=False, sync_dist=True)
        ## Also sample and reconstruct
        if batch_idx % self.cfg[stage].vis_every_n_batches == 0:
            samples = self.diffusion.p_sample_loop(self.model, input_data['x'].shape, input_data['obj_pc'])
            vis_idx = 0
            simp_obj_mesh = getattr(self.trainer.datamodule, f'{stage}_set').simp_obj_mesh
            obj_templates = [trimesh.Trimesh(simp_obj_mesh[name]['verts'], simp_obj_mesh[name]['faces'])
                            for name in batch['objName']]
            # sample_latent = samples[:, 0] ## 1 + latent
            sample_latent = samples  ## latent
            # grid_contact, sample_latent = sample_x[:, :, 0], sample_x[:, :, 1:]
            # grid_contact = grid_contact < 0
            pred_hand_verts, pred_verts_mask, pred_grid_contact = self.reconstruct_from_latent(sample_latent, multi_scale_obj_cond, obj_msdf_center)

            gt_latent = input_data['x']

            # recon_lg_contact, z_e, obj_feat = self.grid_ae(
            #     lg_contact, obj_msdf=obj_msdf, sample_posterior=False)
            # recon_lg_contact = recon_lg_contact.permute(0, 2, 3, 4, 1)  # B x N x K x K x K x (1 + cse_dim)
            # recon_lg_contact = recon_lg_contact.view(batch_size, 128, self.msdf_k, self.msdf_k, self.msdf_k, -1)
            # # recon_lg_contact = recon_lg_contact.view(batch_size, n_grids, -1, self.msdf_k ** 3).permute(0, 1, 3, 2)
            # recon_lg_contact = recon_lg_contact * handobject.obj_pt_mask[:, :, None, None, None, None]
            # sample_contact = recon_lg_contact[..., 0].reshape(batch_size, -1)
            # sample_cse = recon_lg_contact[..., 1:].reshape(batch_size, -1, self.cse_dim)
            # grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords.view(-1, 3)[None, None, :, :]  # B x N x K^3 x 3

            # pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
            #     self.hand_cse, None,
            #     sample_contact.reshape(batch_size, -1), sample_cse.reshape(batch_size, -1, self.cse_dim),
            #     grid_coords=grid_coords.reshape(batch_size, -1, 3),
            #     mask_th = 0.02
            # )

            gt_rec_hand_verts, gt_rec_verts_mask, gt_rec_grid_contact = self.reconstruct_from_latent(gt_latent, multi_scale_obj_cond, obj_msdf_center)

            pred_contact_img, _ = visualize_grid_contact(contact_pts = obj_msdf_center[vis_idx].detach().cpu().numpy(),
                                                    pt_contact = pred_grid_contact[vis_idx].detach().cpu().numpy(),
                                                    grid_scale = self.cfg.msdf.scale,
                                                    obj_mesh = obj_templates[vis_idx],
                                                    w=400, h=400)
            gt_rec_contact_img, _ = visualize_grid_contact(contact_pts = obj_msdf_center[vis_idx].detach().cpu().numpy(),
                                                    pt_contact = gt_rec_grid_contact[vis_idx].detach().cpu().numpy(),
                                                    grid_scale = self.cfg.msdf.scale,
                                                    obj_mesh = obj_templates[vis_idx],
                                                    w=400, h=400)

            gt_contact_img, _ = visualize_grid_contact(contact_pts = obj_msdf_center[vis_idx].detach().cpu().numpy(),
                                                    pt_contact = handobject.obj_pt_mask[vis_idx].detach().cpu().float().numpy(),
                                                    grid_scale = self.cfg.msdf.scale,
                                                    obj_mesh = obj_templates[vis_idx],
                                                    w=400, h=400)


            pred_img, pred_geoms = visualize_recon_hand_w_object(hand_verts=pred_hand_verts[vis_idx].detach().cpu().numpy(),
                                        hand_verts_mask=pred_verts_mask[vis_idx].detach().cpu().numpy(),
                                        hand_faces=self.mano_layer.th_faces.detach().cpu().numpy(),
                                        obj_mesh=obj_templates[vis_idx],
                                        msdf_center=obj_msdf_center[vis_idx].detach().cpu().numpy(),
                                        mesh_color=[0.8, 0.2, 0.2],
                                        grid_scale=self.cfg.msdf.scale,
                                        h=400, w=400)

            gt_rec_img, gt_rec_geoms = visualize_recon_hand_w_object(hand_verts=gt_rec_hand_verts[vis_idx].detach().cpu().numpy(),
                                        hand_verts_mask=gt_rec_verts_mask[vis_idx].detach().cpu().numpy(),
                                        hand_faces=self.mano_layer.th_faces.detach().cpu().numpy(),
                                        obj_mesh=obj_templates[vis_idx],
                                        mesh_color=[0.2, 0.4, 0.8],
                                        msdf_center=obj_msdf_center[vis_idx].detach().cpu().numpy(),
                                        grid_scale=self.cfg.msdf.scale,
                                        h=400, w=400)

            gt_img, gt_geoms = visualize_recon_hand_w_object(hand_verts=handobject.hand_verts[vis_idx].detach().cpu().numpy(),
                                        hand_verts_mask=handobject.hand_vert_mask[vis_idx].any(dim=0).detach().cpu().numpy(),
                                        hand_faces=self.mano_layer.th_faces.detach().cpu().numpy(),
                                        obj_mesh=obj_templates[vis_idx],
                                        mesh_color=[0.2, 0.8, 0.4],
                                        msdf_center=obj_msdf_center[vis_idx].detach().cpu().numpy(),
                                        grid_scale=self.cfg.msdf.scale,
                                        h=400, w=400)

            contact_img = np.concatenate([gt_contact_img, gt_rec_contact_img, pred_contact_img], axis=0)
            img = np.concatenate([gt_img, gt_rec_img, pred_img], axis=0)
            if hasattr(self.logger, 'experiment'):
                if hasattr(self.logger.experiment, 'add_image'):
                    # TensorBoardLogger
                    global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.{stage}_dataloader()')) + batch_idx
                    self.logger.experiment.add_image(f'{stage}/GT_vs_GTRec_vs_sampled_contact', contact_img, global_step, dataformats='HWC')
                    self.logger.experiment.add_image(f'{stage}/GT_vs_GTRec_vs_sampled_hand', img, global_step, dataformats='HWC')
                elif hasattr(self.logger.experiment, 'log'):
                    # WandbLogger
                    import wandb
                    self.logger.experiment.log({f'{stage}/GT_vs_GTRec_vs_sampled_contact': wandb.Image(contact_img)}, step=self.global_step)
                    self.logger.experiment.log({f'{stage}/GT_vs_GTRec_vs_sampled_hand': wandb.Image(img)}, step=self.global_step)

        return loss_dict[f'{stage}/total_loss']
                                  
        # recon_loss = F.mse_loss(recon_grid_contact, gt_grid_contact.permute(0, 4, 1, 2, 3))
        # loss_dict = {f'{stage}/embedding_loss': loss, f'{stage}/recon_loss': recon_loss, f'{stage}/perplexity': perplexity}
        # total_loss = sum(loss_dict.values())
        # total_loss = loss + self.recon_weight * recon_loss
        if batch_idx % self.cfg[stage].vis_every_n_batches == 0:
            # Log image - works for both WandbLogger and TensorBoardLogger
            vis_idx = 0
            pred_img, pred_geoms = self.visualize_recon_hand_w_object(hand_verts=pred_hand_verts[vis_idx].detach().cpu().numpy(),
                                        hand_verts_mask=pred_verts_mask[vis_idx].detach().cpu().numpy(),
                                        hand_faces=self.mano_layer.th_faces.detach().cpu().numpy(),
                                        obj_mesh=handobject.obj_models[vis_idx],
                                        msdf_center=obj_msdf_center[vis_idx].detach().cpu().numpy(),
                                        grid_scale=self.cfg.msdf.scale,
                                        h=400, w=400)
            gt_img, gt_geoms = self.visualize_recon_hand_w_object(hand_verts=handobject.hand_verts[vis_idx].detach().cpu().numpy(),
                                        # hand_verts_mask=handobject.hand_vert_mask[vis_idx].detach().cpu().numpy(),
                                        hand_verts_mask=pred_verts_mask[vis_idx].detach().cpu().numpy(),
                                        hand_faces=self.mano_layer.th_faces.detach().cpu().numpy(),
                                        obj_mesh=handobject.obj_models[vis_idx],
                                        mesh_color=[0.2, 0.4, 0.8],
                                        msdf_center=obj_msdf_center[vis_idx].detach().cpu().numpy(),
                                        grid_scale=self.cfg.msdf.scale,
                                        h=400, w=400)
            # vis_geoms = pred_geoms + [g.translate((0, 0.25, 0)) for g in gt_geoms]
            # o3d.visualization.draw_geometries(vis_geoms, window_name='GT Hand-Object')
            img = np.concatenate([gt_img, pred_img], axis=0)
            if hasattr(self.logger, 'experiment'):
                if hasattr(self.logger.experiment, 'add_image'):
                    # TensorBoardLogger
                    global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.{stage}_dataloader()')) + batch_idx
                    self.logger.experiment.add_image(f'{stage}/compare_hand', img, global_step, dataformats='HWC')
                elif hasattr(self.logger.experiment, 'log'):
                    # WandbLogger
                    import wandb
                    self.logger.experiment.log({f'{stage}/compare_hand': wandb.Image(img)},
                                              step=self.global_step)
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict['total_loss']
    
    def reconstruct_from_latent(self, latent, multi_scale_obj_cond, obj_msdf_center):
        batch_size, n_grids = latent.shape[:2]
        latent = latent.reshape(batch_size*n_grids, -1)
        recon_lg_contact = self.grid_ae.decode(latent, multi_scale_obj_cond)
        recon_lg_contact = recon_lg_contact.view(batch_size, n_grids, -1, self.msdf_k ** 3).permute(0, 1, 3, 2)
        sample_contact = recon_lg_contact[..., 0] # * grid_contact_mask[:, :, None].float()
        sample_cse = recon_lg_contact[..., 1:]
        grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords.view(-1, 3)[None, None, :, :]  # B x N x K^3 x 3
        sample_contact[sample_contact < 0.03] = 0  ## maskout low contact prob

        pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
            self.hand_cse, None,
            sample_contact.reshape(batch_size, -1), sample_cse.reshape(batch_size, -1, self.cse_dim),
            grid_coords=grid_coords.reshape(batch_size, -1, 3),
            mask_th = 2
        )

        grid_contact = sample_contact.max(dim=-1)[0]

        return pred_hand_verts, pred_verts_mask, grid_contact
    
    def on_test_epoch_start(self):
        self.pool = Pool(processes=min(self.cfg.test.batch_size, 16))
        self.all_results = []
        self.sample_joints = []

        ## Testing metrics
        self.runtime = 0
        if not self.debug:
            for metric in self.cfg.test.criteria:
                wandb.define_metric(metric, summary='mean')
            # Initialize W&B table for test images
            self.test_images_table = wandb.Table(columns=[
                "batch_idx", "obj_name", "surrounding_hands", "sampled_grasp",
                "sim_displacement", "penetration_depth", "intersection_volume"
            ])

    def on_train_epoch_start(self):
        """
        Called at the start of each training epoch.
        Handles unfreezing pretrained weights at the specified epoch.
        """
        # self.pool = ThreadPoolExecutor(max_workers=min(self.cfg.train.batch_size, 16))
        self.pool = None

    def on_validation_epoch_start(self):
        # self.pool = ThreadPoolExecutor(max_workers=min(self.cfg.val.batch_size, 16))
        self.pool = None

    def on_train_epoch_end(self):
        if self.pool is not None:
            # self.pool.shutdown(wait=True)
            self.pool.close()
            self.pool.join()

    def on_validation_epoch_end(self):
        if self.pool is not None:
            # self.pool.shutdown(wait=True)
            self.pool.close()
            self.pool.join()

    def test_step(self, batch, batch_idx):
        self.grid_coords = self.grid_coords.to(self.device)
        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer, normalize=True)
        obj_hulls = getattr(self.trainer.datamodule, 'test_set').obj_hulls
        obj_names = batch['objName']
        obj_hulls = [obj_hulls[name] for name in obj_names]
        obj_mesh_dict = getattr(self.trainer.datamodule, 'test_set').simp_obj_mesh
        obj_meshes = []
        batch_size, n_pts = batch['objMsdf'].shape[:2]
        for name in obj_names:
            obj_meshes.append(trimesh.Trimesh(obj_mesh_dict[name]['verts'], obj_mesh_dict[name]['faces']))

        if self.cfg.generator.model_type == 'gt':
            ## Test using gt contact grids.
            handobject.load_from_batch(batch)
            obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:]
            recon_lg_contact = handobject.ml_contact
        else:
            ## Test the reconstrucion 
            handobject.load_from_batch_obj_only(batch)
            # lg_contact = handobject.ml_contact
            # obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, self.msdf_k, self.msdf_k, self.msdf_k)
            # obj_msdf_center = handobject.obj_msdf[:, :, self.msdf_k**3:] # B x 3
            obj_msdf = batch['objMsdf'][:, :, :self.msdf_k**3].view(-1, 1, self.msdf_k, self.msdf_k, self.msdf_k)
            obj_msdf_center = batch['objMsdf'][:, :, self.msdf_k**3:] # B x N x 3
            batch_size, n_grids = obj_msdf_center.shape[:2]

            obj_feat, multi_scale_obj_cond = self.grid_ae.encode_object(obj_msdf)
            obj_pc = torch.cat([obj_msdf_center, obj_feat.view(batch_size, n_grids, -1)], dim=-1)

            ## 'x' only indicates the latent shape; latents are sampled inside the model
            input_data = {'x': torch.zeros(batch_size, n_grids, self.cfg.ae.feat_dim), 'obj_pc': obj_pc.permute(0, 2, 1)}
            samples = self.diffusion.sample(self.model, input_data, k=self.cfg.test.n_samples)
            sample_latent = samples[:, 0] ## B x latent
            latent = sample_latent.reshape(batch_size*n_grids, -1)
            recon_lg_contact = self.grid_ae.decode(latent, multi_scale_obj_cond)
            # recon_lg_contact, mu, logvar = self.model(
            #     lg_contact.permute(0, 1, 5, 2, 3, 4), obj_msdf=obj_msdf, msdf_center=obj_msdf_center)
            # recon_lg_contact, z_e, obj_feat = self.grid_ae(
            #     lg_contact.view(batch_size*n_pts, self.msdf_k, self.msdf_k, self.msdf_k, -1).permute(0, 4, 1, 2, 3),
            #     obj_msdf=obj_msdf.unsqueeze(1), sample_posterior=False)
            recon_lg_contact = recon_lg_contact.permute(0, 2, 3, 4, 1)  # B x K x K x K x (1 + cse_dim)
            recon_lg_contact = recon_lg_contact.view(batch_size, n_pts, self.msdf_k, self.msdf_k, self.msdf_k, -1)
            # recon_lg_contact = lg_contact

            ## If masked with GT pt mask
            # recon_lg_contact = recon_lg_contact * handobject.obj_pt_mask[:, :, None, None, None, None]
            recon_lg_contact[..., 0][recon_lg_contact[..., 0] < 0.03] = 0  ## maskout low contact prob

            # handobject.load_from_batch_object_only(batch)
            # obj_msdf = handobject.obj_msdf[:, :, :self.msdf_k**3].view(-1, self.msdf_k, self.msdf_k, self.msdf_k)
            # recon_lg_contact = self.model.sample(obj_msdf, obj_msdf_center).permute(0, 1, 3, 4, 5, 2)

        batch_size, n_pts = recon_lg_contact.shape[:2]
        pred_grid_contact = recon_lg_contact[..., 0].reshape(batch_size, -1)  # B x N x K^3
        grid_coords = obj_msdf_center[:, :, None, :] + self.grid_coords.view(-1, 3)[None, None, :, :]  # B x N x K^3 x 3
        pred_grid_cse = recon_lg_contact[..., 1:].reshape(batch_size, -1, self.cse_dim)
        pred_targetWverts = self.hand_cse.emb2Wvert(pred_grid_cse.view(batch_size, -1, self.cse_dim))

        with torch.enable_grad():
            global_pose, mano_pose, mano_shape, mano_trans = optimize_pose_wrt_local_grids(
                        self.mano_layer, target_pts=grid_coords.view(batch_size, -1, 3),
                        target_W_verts=pred_targetWverts, weights=pred_grid_contact,
                        n_iter=self.cfg.pose_optimizer.n_opt_iter, lr=self.cfg.pose_optimizer.opt_lr)
        
        handV, handJ, _ = self.mano_layer(torch.cat([global_pose, mano_pose], dim=1), th_betas=mano_shape, th_trans=mano_trans)

        handV, handJ = handV.detach().cpu().numpy(), handJ.detach().cpu().numpy()

        param_list = [{'dataset_name': 'grab', 'frame_name': f"{obj_names[i]}_{i}", 'hand_model': trimesh.Trimesh(handV[i], self.closed_mano_faces),
                       'obj_name': obj_names[i], 'hand_joints': handJ[i], 'obj_model': obj_meshes[i], 'obj_hulls': obj_hulls[i],
                       'idx': i} for i in range(handV.shape[0])]
            
        result = calculate_metrics(param_list, metrics=self.cfg.test.criteria, pool=self.pool, reduction='none')
        ## MPJPE:
        if self.cfg.test_gt:
            mpjpe = np.linalg.norm(handobject.hand_joints.cpu().numpy() - handJ, axis=-1).mean(axis=1) * 1000  # B,
            mpvpe = np.linalg.norm(handobject.hand_verts.cpu().numpy() - handV, axis=-1).mean(axis=1) * 1000  # B,
            result.update({'MPJPE': mpjpe, 'MPVPE': mpvpe})

        self.all_results.append(result)
        self.sample_joints.append(handJ)

        # Log raw per-sample metrics to wandb
        if not self.debug:
            # Option 1: Log each sample as individual rows (creates distributions in wandb)
            for i in range(len(next(iter(result.values())))):
                sample_metrics = {f"sample/{metric_name}": float(metric_values[i])
                                 for metric_name, metric_values in result.items()}
                wandb.log(sample_metrics, commit=False)

            # Option 2 (alternative): Use wandb Table for structured logging
            # table_data = [[obj_names[i]] + [float(result[m][i]) for m in result.keys()]
            #               for i in range(batch_size)]
            # table = wandb.Table(data=table_data, columns=["object"] + list(result.keys()))
            # wandb.log({f"test_metrics_batch_{batch_idx}": table})
        ## Visualization
        # for vis_idx in range(handV.shape[0]):
        vis_idx = 0
        # gt_geoms = handobject.get_vis_geoms(idx=vis_idx, obj_templates=obj_meshes)
        pred_ho = copy(handobject)
        pred_ho.hand_verts = torch.tensor(handV, dtype=torch.float32)
        pred_ho.hand_joints = torch.tensor(handJ, dtype=torch.float32)
        pred_img = pred_ho.vis_img(idx=vis_idx, h=400, w=400, obj_templates=obj_meshes)

        pred_hand_verts, pred_verts_mask = recover_hand_verts_from_contact(
            self.hand_cse, None,
            pred_grid_contact.reshape(batch_size, -1),
            pred_grid_cse.reshape(batch_size, -1, self.cse_dim),
            grid_coords=grid_coords.reshape(batch_size, -1, 3),
            mask_th = 2
        )
        recon_img, pred_geoms = visualize_recon_hand_w_object(hand_verts=pred_hand_verts[vis_idx].detach().cpu().numpy(),
                                    hand_verts_mask=pred_verts_mask[vis_idx].detach().cpu().numpy(),
                                    hand_faces=self.mano_layer.th_faces.detach().cpu().numpy(),
                                    obj_mesh=obj_meshes[vis_idx],
                                    msdf_center=obj_msdf_center[vis_idx].detach().cpu().numpy(),
                                    mesh_color=[0.8, 0.2, 0.2],
                                    grid_scale=self.cfg.msdf.scale,
                                    h=400, w=400)

        if hasattr(self.logger, 'experiment'):
            if hasattr(self.logger.experiment, 'add_image'):
                # TensorBoardLogger
                global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.test_dataloader()')) + batch_idx
                self.logger.experiment.add_image(f'test/Surrounding_hands', recon_img, global_step, dataformats='HWC')
                self.logger.experiment.add_image(f'test/Sampled_grasp', pred_img, global_step, dataformats='HWC')
            elif hasattr(self.logger.experiment, 'log') and not self.debug:
                # WandbLogger - add row to table
                self.test_images_table.add_data(
                    batch_idx,
                    obj_names[vis_idx],
                    wandb.Image(recon_img),
                    wandb.Image(pred_img),
                    float(np.mean(result.get("Simulation Displacement", [0]))),
                    float(np.mean(result.get("Penetration Depth", [0]))),
                    float(np.mean(result.get("Intersection Volume", [0])))
                )
            # o3d.visualization.draw(pred_geoms)
            # o3d.visualization.draw(gt_geoms + [g['geometry'].translate((0, 0.25, 0)) if 'geometry' in g else g for g in pred_geoms])

        return result

    def on_test_epoch_end(self):
        final_metrics = {}

        # Compute statistics for all metrics from test results
        for m in self.cfg.test.criteria:
            if "Entropy" not in m and "Cluster Size" not in m:
                all_metrics = np.concatenate([res[m] for res in self.all_results], axis=0)
                # Log comprehensive statistics for each metric
                final_metrics[f"{m}/mean"] = np.mean(all_metrics).item()
                final_metrics[f"{m}/std"] = np.std(all_metrics).item()
                final_metrics[f"{m}/min"] = np.min(all_metrics).item()
                final_metrics[f"{m}/max"] = np.max(all_metrics).item()
                # final_metrics[f"{m}/median"] = np.median(all_metrics).item()
                # final_metrics[f"{m}/p25"] = np.percentile(all_metrics, 25).item()
                # final_metrics[f"{m}/p75"] = np.percentile(all_metrics, 75).item()

        ## Calculate diversity
        sample_joints = np.concatenate(self.sample_joints, axis=0)
        # run_time = np.concatenate(run_time, axis=0)
        entropy, cluster_size, entropy_2, cluster_size_2 = calc_diversity(sample_joints)
        final_metrics.update({
            "Entropy": entropy.item(),
            "Cluster Size": cluster_size.item(),
            "Canonical Entropy": entropy_2.item(),
            "Canonical Cluster Size": cluster_size_2.item()
        })

        # Log final metrics and test images table to wandb
        if not self.debug:
            wandb.log(final_metrics, commit=False)
            wandb.log({"test/images": self.test_images_table})

        # return final_metrics
    
        ## GT visualization
        # vis_idx = 0
        # gt_geoms = visualize_local_grid_with_hand(
        #         batch['localGrid'][vis_idx].cpu().numpy(), hand_verts=batch['nHandVerts'][vis_idx].cpu().numpy(),
        #         hand_faces=self.mano_layer.th_faces.cpu().numpy(), hand_cse=self.hand_cse.embedding_tensor.detach().cpu().numpy(),
        #         kernel_size=self.cfg.msdf.kernel_size, grid_scale=self.cfg.msdf.scale
        #     )
        # o3d.visualization.draw_geometries(gt_geoms, window_name='GT Local Grid Visualization')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # StepLR scheduler: decay LR by gamma every step_size epochs
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.8
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }