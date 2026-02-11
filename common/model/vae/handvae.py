import torch
import numpy as np
import torch.nn as nn
from common.model.pointnet_module import PointNetEncoder
from common.manopth.manopth.manolayer import ManoLayer
from common.model.handobject import HandObject
from common.model.hand_ipt_vae.hand_imputation import Linear_ResBlock, MLPPointEncoder
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, axis_angle_to_matrix, matrix_to_rotation_6d
from common.model.layers import DiagonalGaussianDistribution
import open3d as o3d
from common.utils.vis import o3dmesh, geom_to_img
from lightning import LightningModule
from .graphAE import Encoder

class HandVAE(nn.Module):
    def __init__(self, cfg):
        super(HandVAE, self).__init__()
        object.__setattr__(self, 'mano_layer', ManoLayer(mano_root=cfg.mano_root, use_pca=False, flat_hand_mean=True).eval().requires_grad_(False))
        self.hidden_dim = cfg.hidden_dim
        self.latent_dim = cfg.latent_dim
        # Define encoder layers (outputs 2x latent_dim for mean and logvar)
        # self.hand_encoder = PointNetEncoder(
        #     channel=cfg.input_channel, hidden_dim=cfg.hidden_dim)
        self.hand_encoder = MLPPointEncoder(in_channels=cfg.input_channel, hidden_dim=cfg.hidden_dim, feat_dim=cfg.feat_dim)
        
        self.encoder = nn.Sequential(
            Linear_ResBlock(input_size=cfg.feat_dim, output_size=cfg.feat_dim),
            Linear_ResBlock(input_size=cfg.feat_dim, output_size=cfg.latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            Linear_ResBlock(input_size=cfg.latent_dim, output_size=2 * cfg.latent_dim),
            Linear_ResBlock(input_size=2 * cfg.latent_dim, output_size=61)
        )


        # self.encoder = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.latent_dim * 2)
        # )

        # self.decoder = nn.Sequential(
        #     nn.Linear(self.latent_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.hidden_dim*2),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim*2, 61)
        # )

    def encode(self, x):
        feat = self.hand_encoder(x)
        h = self.encoder(feat)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z):
        self.mano_layer.to(z.device)
        recon_param = self.decoder(z)
        trans = self.denormalize_trans(recon_param[:, :3])
        pose = recon_param[:, 3:51]
        betas = recon_param[:, 51:]
        handV, handJ, _ = self.mano_layer(pose, th_betas=betas, th_trans=trans)
        return recon_param, handV, handJ

    def forward(self, x):
        posterior = self.encode(x)
        z = posterior.sample()
        recon_param, handV, handJ = self.decode(z)
        return recon_param, handV, handJ, posterior
    
    def normalize_trans(self, hand_trans):
        return hand_trans / 0.2

    def denormalize_trans(self, hand_trans):
        return hand_trans * 0.2


## Adopted from Mesh-VQVAE
class GraphHandVAE(nn.Module):
    def __init__(
        self,
        fully_conv_ae,
        num_embeddings=512,
        embedding_dim=9,
        commitment_cost=0.25,
        decay=0,
        num_quantizer=1,
        shared_codebook=False,
    ):
        """Initialize a Mesh-VQ-VAE

        Args:
            fully_conv_ae (FullyConvAE): A fully convolutional mesh autoencoder.
            num_embeddings (int, optional): The number of embeddings in the dictionary. Defaults to 512.
            embedding_dim (int, optional): The dimension of each embedding. Defaults to 9.
            commitment_cost (float, optional): The weight for the commitment loss in training the VQ-VAE. Defaults to 0.25.
            decay (int, optional): Decay for the moving averages. Defaults to 0.
            num_quantizer (int, optional): Allows to implement a RQ-VAE with multiple quantizers. Defaults to 1.
            shared_codebook (bool, optional): In the case of RQ-VAE, shares the codebook among quantizations. Defaults to False.
        """
        super(GraphHandVAE, self).__init__()

        self.encoder = Encoder(fully_conv_ae)
        self.num_quantizer = num_quantizer
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.quantize = False

    def forward(self, x, detailed=False):
        """Encoding and decoding the input meshes x.

        Args:
            x (torch.Tensor): Batch of input meshes in a tensor of shape (B, 6890, 3)
            detailed (bool, optional): If detailed is True, we output all the intermediate meshes in the RQ-VAE. Defaults to False.

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor: loss is the loss associated with the encoding-decoding process, x_recon are the reconstructed meshes,
            and perplexity gives information about the codebook usage.
        """
        z = self.encoder(x)
        if self.quantize:
            loss, quantized, perplexity, _ = self.vq_vae(z, detailed=detailed)
            if not detailed:
                x_recon = self.decoder(quantized)
            else:
                x_recon = []
                for quantized_int in quantized:
                    x_recon.append(self.decoder(quantized_int))
        else:
            loss = torch.zeros(1).to(z)
            perplexity = torch.zeros(1).to(z)
            x_recon = self.decoder(z)
        return loss, x_recon, perplexity

    def load(self, path_model: str):
        """Load the Mesh-VQ-VAE model

        Args:
            path_model (str): Path of the checkpoint.
        """
        checkpoint = torch.load(path_model)
        self.load_state_dict(checkpoint["model"], strict=False)
        loss = checkpoint["loss"]
        print(f"\t [Mesh-VQ-VAE is loaded successfully with loss = {loss}]")

    def encode(self, meshes):
        """Encode meshes

        Args:
            meshes (torch.Tensor): A batch of human meshes.

        Returns:
            torch.Tensor: The continuous latent representation if there is no quantization, the sequence of indices otherwise.
        """
        z = self.encoder(meshes)
        if self.quantize:
            z = self.vq_vae.get_codebook_indices(z)
        return z

    def decode(self, z):
        """Get the full mesh in 3D given the latent representation.

        Args:
            z (torch.Tensor): The latent representation in the form of indices if this is a VQ-VAE, and continuous otherwise.

        Returns:
            torch.Tensor: Meshes in a tensor of shape (B, 6890, 3).
        """
        if self.quantize:
            all_mesh_embeds = self.vq_vae.quantify(z)
            if self.num_quantizer != 1:
                all_mesh_embeds = torch.sum(all_mesh_embeds, dim=-2)
        else:
            all_mesh_embeds = z
        meshes = self.decoder(all_mesh_embeds)
        return meshes


class HandVAETrainer(LightningModule):
    def __init__(self, model, cfg):
        super(HandVAETrainer, self).__init__()
        self.model = model
        self.criterion = nn.L1Loss()
        self.cfg = cfg

    def training_step(self, batch, batch_idx):
        return self.train_val_step(batch, batch_idx, stage='train')
    
    def validation_step(self, batch, batch_idx):
        return self.train_val_step(batch, batch_idx, stage='val')
    
    def train_val_step(self, batch, batch_idx, stage):
        # thetas = batch['theta']
        # betas = batch['beta']
        # ## Augment the betas by adding noise.
        # betas = betas + torch.randn_like(betas) * 0.1
        # pose_6d = matrix_to_rotation_6d(axis_angle_to_matrix(thetas[:, 3:].view(-1, 15, 3))).view(-1, 15*6)
        # nhandV_gt, nhandJ_gt, _ = self.model.mano_layer(
        #     torch.cat([torch.zeros(thetas.shape[0], 3, device=self.device), thetas[:, 3:]], dim=-1),
        #     th_betas=betas)

        handobject = HandObject(self.cfg.data, self.device, mano_layer=self.model.mano_layer, normalize=True)
        handobject.load_from_batch(batch)
        handV_gt = handobject.hand_verts
        handJ_gt = handobject.hand_joints
        self.model.mano_layer.to(self.device)
        # root_j = handobject.cano_joints[:, 0]
        # hand_root_R = axis_angle_to_matrix(handobject.hand_root_rot)
        # hand_trans = handobject.hand_trans
        # nhandV_gt = (handV_gt - root_j.unsqueeze(1) - hand_trans.unsqueeze(1)) @ hand_root_R + root_j.unsqueeze(1)
        # nhandJ_gt = (handJ_gt - root_j.unsqueeze(1) - hand_trans.unsqueeze(1)) @ hand_root_R + root_j.unsqueeze(1)

        # Verify that hand_param and handV_gt match
        # if batch_idx == 0:
        #     with torch.no_grad():
        #         hand_trans = hand_param[:, :3]
        #         full_pose = matrix_to_axis_angle(rotation_6d_to_matrix(hand_param[:, 3:99].view(-1, 16, 6))).view(-1, 16*3)
        #         handV_from_param, handJ_from_param, _ = self.model.mano_layer(full_pose, th_trans=hand_trans)
        #         vert_diff = (handV_from_param - handV_gt).abs().max().item()
        #         joint_diff = (handJ_from_param - handJ_gt).abs().max().item()
        #         hand_from_param_mesh = o3dmesh(handV_from_param[0].cpu().numpy(), self.model.mano_layer.th_faces.cpu().numpy())
        #         hand_gt_mesh = o3dmesh(handV_gt[0].cpu().numpy(), self.model.mano_layer.th_faces.cpu().numpy())
        #         o3d.visualization.draw_geometries([hand_from_param_mesh, hand_gt_mesh])
        #         print(f"[Verification] Max vertex diff: {vert_diff:.6f}, Max joint diff: {joint_diff:.6f}")
        #         if vert_diff > 1e-4 or joint_diff > 1e-4:
        #             print(f"[Warning] Hand param and vertices mismatch!")

        recon_param, handV_pred, handJ_pred, posterior = self.model(handV_gt.permute(0, 2, 1))
        # param_loss = self.criterion(recon_param, hand_param) # Ignore the shape param here.

        # w_param = self.cfg.train.loss_weights.w_param 
            # print(f"Epoch {self.current_epoch}: Switching to reconstruction + KL loss.")

        recon_loss = (self.criterion(handV_pred, handV_gt) + self.criterion(handJ_pred, handJ_gt)) / 2
        kld_loss = posterior.kl().mean()
        loss = self.cfg.train.loss_weights.w_recon * recon_loss + self.cfg.train.loss_weights.w_kl * kld_loss

        loss_dict = {
            f'{stage}/total_loss': loss,
            # f'{stage}/param_loss': param_loss,
            f'{stage}/recon_loss': recon_loss,
            f'{stage}/kld_loss': kld_loss
        }

        if stage == 'val':
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        else:
            self.log_dict(loss_dict, prog_bar=True, sync_dist=True)

        if batch_idx % self.cfg[stage].vis_every_n_batches == 0:
            pred_mesh = o3dmesh(handV_pred[0].detach().cpu().numpy(), self.model.mano_layer.th_faces.cpu().numpy(), color=[1, 0, 0])
            gt_mesh = o3dmesh(handV_gt[0].detach().cpu().numpy(), self.model.mano_layer.th_faces.cpu().numpy(), color=[0, 0, 1])
            vis_img = geom_to_img([pred_mesh, gt_mesh], w=200, h=200, half_range=0.1)
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
        recon_param, handV_pred, handJ_pred, posterior = self.model(handV_gt.permute(0, 2, 1))
        recon_error = torch.norm(handV_pred - nhandV_gt, dim=-1).mean(dim=-1)  # B
        self.recon_err_list.append(recon_error.detach().cpu().numpy())
    
    def on_test_epoch_end(self):
        avg_recon_err = np.concatenate(self.recon_err_list, axis=0).mean() * 1000
        print(f"Average hand vertex reconstruction error: {avg_recon_err:.6f} mm")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.lr)
        return optimizer