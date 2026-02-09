"""
Adopted from VRCNet: https://github.com/paul007pl/VRCNet.git.
Changed the encoder from PointNet to MLP for positional awareness.
"""
import torch
import torch.nn as nn
from lightning import LightningModule
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle
import numpy as np

from common.model.handobject import HandObject
from common.manopth.manopth.manolayer import ManoLayer
from common.utils.vis import o3dmesh, geom_to_img, extract_masked_mesh_components


class Linear_ResBlock(nn.Module):
    def __init__(self, input_size=1024, output_size=256):
        super(Linear_ResBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.fc_res = nn.Linear(input_size, output_size)

        self.af = nn.ReLU(inplace=True)

    def forward(self, feature):
        return self.fc2(self.af(self.fc1(self.af(feature)))) + self.fc_res(feature)


class MLPPointEncoder(nn.Module):
    def __init__(self, in_channels=4, num_points=778, hidden_dim=256, feat_dim=1024):
        super(MLPPointEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, 1)
        self.fc2 = nn.Linear(16 * num_points, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, feat_dim)

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(feat_dim)

        self.af = nn.ReLU(inplace=True)

    def forward(self, points):
        x = self.af(self.bn1(self.conv1(points)))
        x = x.view(x.size(0), -1)
        x = self.af(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        return x


class HandImputationVAE(nn.Module):
    def __init__(self, input_channels=4, num_points=778, global_feature_size=1024, latent_dim=16, hidden_dim=256, output_size=61, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.posterior_infer1 = Linear_ResBlock(input_size=global_feature_size, output_size=global_feature_size)
        self.posterior_infer2 = Linear_ResBlock(input_size=global_feature_size, output_size=latent_dim * 2)
        self.prior_infer = Linear_ResBlock(input_size=global_feature_size, output_size=latent_dim * 2)
        self.encoder = MLPPointEncoder(in_channels=input_channels, num_points=num_points, hidden_dim=hidden_dim, feat_dim=global_feature_size)
        self.generator = Linear_ResBlock(input_size=latent_dim, output_size=global_feature_size)
        self.decoder = Linear_ResBlock(input_size=global_feature_size, output_size=output_size)

    def forward(self, x, mask, is_training=True):
        y = x.clone()
        y = torch.cat([y, torch.ones_like(mask, device=mask.device).float()], dim=1)

        x = x.clone()
        x.permute(0, 2, 1)[~mask.squeeze(1)] = 0.0
        x = torch.cat([x, mask.float()], dim=1)

        if is_training:
            points = torch.cat([x, y], dim=0) ## Concatenate as batches
        else:
            points = x

        feat = self.encoder(points)

        if is_training:
            feat_x, feat_y = feat.chunk(2)
            feat_x, feat_y = feat_x.clone(), feat_y.clone()
            o_x = self.posterior_infer2(self.posterior_infer1(feat_x))
            q_mu, q_logvar = torch.split(o_x, self.latent_dim, dim=1)
            o_y = self.prior_infer(feat_y)
            p_mu, p_logvar = torch.split(o_y, self.latent_dim, dim=1)
            q_std = torch.exp(0.5 * q_logvar)
            p_std = torch.exp(0.5 * p_logvar)
            q_distribution = torch.distributions.Normal(q_mu, q_std)
            p_distribution = torch.distributions.Normal(p_mu, p_std)
            p_distribution_fix = torch.distributions.Normal(p_mu.detach(), p_std.detach())
            m_distribution = torch.distributions.Normal(torch.zeros_like(p_mu), torch.ones_like(p_std))
            z_q = q_distribution.rsample()
            z_p = p_distribution.rsample()
            z = torch.cat([z_q, z_p], dim=0)
            feat = torch.cat([feat_x, feat_x], dim=0)

        else:
            o_x = self.posterior_infer2(self.posterior_infer1(feat))
            q_mu, q_logvar = torch.split(o_x, self.latent_dim, dim=1)
            q_std = torch.exp(0.5 * q_logvar)
            q_distribution = torch.distributions.Normal(q_mu, q_std)
            p_distribution = q_distribution
            p_distribution_fix = p_distribution
            m_distribution = p_distribution
            z = q_distribution.rsample()

        feat += self.generator(z)

        recon_params = self.decoder(feat)

        dl_rec = torch.distributions.kl_divergence(m_distribution, p_distribution)
        dl_g = torch.distributions.kl_divergence(p_distribution_fix, q_distribution)

        kl_loss = dl_rec.mean() + dl_g.mean()
        return recon_params, kl_loss
    

class HandImputationVAETrainer(LightningModule):
    def __init__(self, model, cfg):
        super(HandImputationVAETrainer, self).__init__()
        self.model = model
        self.mano_layer = ManoLayer(mano_root=cfg.data.mano_root, use_pca=False, flat_hand_mean=True)
        self.mano_layer.requires_grad_(False)
        self.criterion = nn.L1Loss()
        self.cfg = cfg
        self.lr = cfg.train.lr
        self.msdf_scale = cfg.msdf.scale

    def training_step(self, batch, batch_idx):
        return self.train_val_step(batch, batch_idx, stage='train')
    
    def validation_step(self, batch, batch_idx):
        return self.train_val_step(batch, batch_idx, stage='val')
    
    def train_val_step(self, batch, batch_idx, stage):
        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.mano_layer, normalize=False)
        handobject.load_from_batch(batch)
        # mask = handobject.hand_vert_mask
        handV_gt = handobject.hand_verts
        handJ_gt = handobject.hand_joints
        msdf_center = batch['objMsdf'][:, :, -3:]
        ho_dist = torch.abs(msdf_center[:, :, None, :] - handV_gt[:, None, :, :]).max(dim=-1)[0] # B x 778 x N
        mask = (ho_dist < self.msdf_scale).any(dim=1)

        # Random rotation augmentation (up to 45 degrees) during training
        if stage == 'train':
            max_angle = np.pi / 4  # 45 degrees
            random_aa = torch.randn(handV_gt.shape[0], 3, device=self.device)
            random_aa = random_aa / (random_aa.norm(dim=-1, keepdim=True) + 1e-8)
            random_aa = random_aa * (torch.rand(handV_gt.shape[0], 1, device=self.device) * max_angle)
            aug_R = axis_angle_to_matrix(random_aa)  # (B, 3, 3)
            handV_gt = handV_gt @ aug_R.transpose(-1, -2)
            handJ_gt = handJ_gt @ aug_R.transpose(-1, -2)

        recon_param, kld_loss = self.model(handV_gt.permute(0, 2, 1), mask=mask.unsqueeze(1), is_training=stage=='train')
        # param_loss = self.criterion(recon_param, hand_param) # Ignore the shape param here.
        nrecon_trans, recon_pose, recon_betas = torch.split(recon_param, [3, 48, 10], dim=1)
        recon_trans = nrecon_trans * 0.2
        gen_handV, gen_handJ, _ = self.mano_layer(recon_pose, th_betas=recon_betas, th_trans=recon_trans)

        # w_param = self.cfg.train.loss_weights.w_param 
            # print(f"Epoch {self.current_epoch}: Switching to reconstruction + KL loss.")
        if stage == 'train':
            recon_handV, gen_handV = torch.chunk(gen_handV, 2, dim=0)
            recon_handJ, gen_handJ = torch.chunk(gen_handJ, 2, dim=0)
            recon_loss = (self.criterion(recon_handV, handV_gt) + self.criterion(recon_handJ, handJ_gt)) / 2
        else:
            recon_loss = torch.tensor(0.0, device=self.device)
        w = torch.ones_like(mask).float()
        w[mask] = 3 # special attention on visible vertices
        w = w / torch.sum(w, dim=-1, keepdim=True)
        diff_genV = (gen_handV - handV_gt) * w.unsqueeze(-1)
        diff_genJ = (gen_handJ - handJ_gt) 
        gen_loss = (self.criterion(diff_genV, torch.zeros_like(diff_genV)) + self.criterion(diff_genJ, torch.zeros_like(diff_genJ))) / 2
        loss = self.cfg.train.loss_weights.w_gen * gen_loss + self.cfg.train.loss_weights.w_recon * recon_loss + self.cfg.train.loss_weights.w_kl * kld_loss

        loss_dict = {
            f'{stage}/total_loss': loss,
            f'{stage}/gen_loss': gen_loss,
            f'{stage}/recon_loss': recon_loss,
            f'{stage}/kld_loss': kld_loss
        }

        if stage == 'val':
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        else:
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True)

        if batch_idx % self.cfg[stage].vis_every_n_batches == 0:
            handV_gt = handV_gt.detach().cpu().numpy()
            gen_handV = gen_handV.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
            vis_idx = np.random.randint(0, handV_gt.shape[0]-1)
            hand_faces = self.mano_layer.th_faces.cpu().numpy()
            input_parts = extract_masked_mesh_components(handV_gt[vis_idx], hand_faces, mask[vis_idx], handobject.hand_part_ids)
            pred_mesh = o3dmesh(gen_handV[vis_idx], hand_faces, color=[0.9, 0.7, 0.6])
            pred_img = geom_to_img([pred_mesh] + input_parts, w=200, h=200, half_range=0.1)
            gt_mesh = o3dmesh(handV_gt[vis_idx], hand_faces, color=[0.9, 0.7, 0.6])
            gt_img = geom_to_img([gt_mesh] + input_parts, w=200, h=200, half_range=0.1)
            vis_imgs = [gt_img, pred_img]

            if stage == 'train':
                recon_handV = recon_handV.detach().cpu().numpy()
                recon_mesh = o3dmesh(recon_handV[vis_idx], hand_faces, color=[0.9, 0.7, 0.6])
                recon_img = geom_to_img([recon_mesh] + input_parts, w=200, h=200, half_range=0.1)
                vis_imgs.append(recon_img)

            vis_img = np.concatenate(vis_imgs, axis=0)
            # vis_img = geom_to_img([pred_mesh, gt_mesh], w=200, h=200, half_range=0.1)
            if hasattr(self.logger, 'experiment'):
                if hasattr(self.logger.experiment, 'add_image'):
                    # TensorBoardLogger expects (C, H, W), geom_to_img returns (H, W, C)
                    vis_img_chw = np.transpose(vis_img, (2, 0, 1))
                    global_step = self.current_epoch * len(eval(f'self.trainer.datamodule.{stage}_dataloader()')) + batch_idx
                    self.logger.experiment.add_image(f'{stage}/hand_recon', vis_img_chw, global_step)
                elif hasattr(self.logger.experiment, 'log'):
                    # WandbLogger expects (H, W, C)
                    import wandb
                    self.logger.experiment.log({f'{stage}/hand_recon': wandb.Image(vis_img)}, step=self.global_step)

        return loss
    
    def on_test_batch_start(self, batch, batch_idx):
        self.recon_err_list = []

    def test_step(self, batch, batch_idx):
        thetas = batch['theta']
        betas = batch['beta']
        ## Augment the betas by adding noise.
        betas = betas + torch.randn_like(betas)
        pose_6d = matrix_to_rotation_6d(axis_angle_to_matrix(thetas[:, 3:].view(-1, 15, 3))).view(-1, 15*6)
        nhandV_gt, nhandJ_gt, _ = self.model.mano_layer(
            torch.cat([torch.zeros(thetas.shape[0], 3, device=self.device), thetas[:, 3:]], dim=-1),
            th_betas=betas)
            
        # handobject = HandObject(self.cfg.data, self.device, mano_layer=self.model.mano_layer, normalize=False)
        # handobject.load_from_batch(batch)
        # handV_gt = handobject.hand_verts
        # handJ_gt = handobject.hand_joints
        # root_j = handobject.cano_joints[:, 0]
        # hand_root_R = axis_angle_to_matrix(handobject.hand_root_rot)
        # hand_trans = handobject.hand_trans
        # nhandV = (handV_gt - root_j.unsqueeze(1) - hand_trans.unsqueeze(1)) @ hand_root_R + root_j.unsqueeze(1)
        recon_param, handV_pred, handJ_pred, posterior = self.model(nhandV_gt.permute(0, 2, 1))
        recon_error = torch.norm(handV_pred - nhandV_gt, dim=-1).mean(dim=-1)  # B
        self.recon_err_list.append(recon_error.detach().cpu().numpy())
    
    def on_test_epoch_end(self):
        avg_recon_err = np.concatenate(self.recon_err_list, axis=0).mean() * 1000
        print(f"Average hand vertex reconstruction error: {avg_recon_err:.6f} mm")


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

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

