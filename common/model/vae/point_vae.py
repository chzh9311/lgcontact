import torch
import torch.nn as nn
from .pointnet_module import PointNet2cls, LatentEncoder, PointNet2seg, Pointnet
# from ..gridae.encoder import GridEncoder3D
# from ..gridae.decoder import GridDecoder3D
from ..gridae.gridae import GRIDAE

class POINT_VAE(nn.Module):
    """
    MSDF-based 3D contact VQVAE of all local patches
    The input contact representation is expected to be kernel_size^3 x (1 + 16); 1 refers to contact likelihood, 16 refers to Hand CSE.
    Also, the object local patch is represented as kernel_size^3 SDF values.
    TODO 1: how to integrate scale? Fix the scale to a constant value like 0.01 (1cm);
    Thus the input is k x k x k x (1 + 1 + 16) = k x k x k x 18
    Will be processed by 3D convolutions & deconvolutions.
    TODO 2: how to enable conditional input, i.e., predict the contact given the object local geometry? 
    We can try add this as part of the input to the encoder & decoder.
    """
    def __init__(self, cfg):
        super(POINT_VAE, self).__init__()
        # encode image into continuous latent space
        # self.obj_encoder = Encoder(obj_in_dim, h_dims, obj_n_res_layers, obj_res_h_dim, condition=False)
        self.grid_ae = GRIDAE(cfg=cfg.ae, obj_1d_feat=True)
        self.cfg = cfg.generator
        self.latent_dim = cfg.generator.latent.dim
        self.out_dim = cfg.ae.out_dim
        self.msdf_k = cfg.msdf.kernel_size
        # self.objgridencoder = GridEncoder3D(obj_in_dim, h_dims, res_h_dim, n_res_layers, feat_dim, N=N, condition=False)
        # self.encoder = PointNet2cls(in_channel=cfg.ae.feat_dim + cfg.generator.glob_obj_feat_dim + 3,
        #                             in_point=cfg.msdf.num_grids,
        #                             hidden_dim=cfg.generator.encoder.hd,
        #                             out_channel=cfg.generator.glob_feat_dim)
        self.encoder = Pointnet(in_dim=cfg.ae.feat_dim + cfg.generator.glob_obj_feat_dim + 3,
                                hidden_dim=cfg.generator.encoder.hd,
                                out_dim=cfg.generator.glob_feat_dim)
        self.obj_encoder = PointNet2seg(in_dim=cfg.ae.obj_feat_dim+3,
                                        in_point=cfg.msdf.num_grids,
                                        hidden_dim=cfg.generator.obj_encoder.hd,
                                        out_dim=cfg.generator.glob_obj_feat_dim)
        self.latent_encoder = LatentEncoder(cfg.generator.glob_feat_dim, cfg.generator.latent.hd, self.latent_dim)
        # self.encoder = Encoder(in_dim, h_dims, n_res_layers, res_h_dim)
        # self.objgriddecoder = GridDecoder3D(latent_dim, h_dims[::-1], out_dim, n_res_layers, res_h_dim, condition=True)
        # self.decoder = PointNet2seg(in_dim=self.latent_dim + cfg.generator.glob_obj_feat_dim,
        #                             in_point=cfg.msdf.num_grids,
        #                             hidden_dim=cfg.generator.decoder.hd,
        #                             out_dim=cfg.ae.feat_dim)
        self.decoder = Pointnet(in_dim=self.latent_dim + cfg.generator.glob_obj_feat_dim,
                                hidden_dim=cfg.generator.decoder.hd,
                                out_dim=cfg.ae.feat_dim)

        # if save_img_embedding_map:
        #     self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        # else:
        #     self.img_to_embedding_map = None

    def forward(self, x, obj_msdf, msdf_center):
        ### x: B x N x (C x k x k x k)

        batch_size, num_grids = x.shape[:2]
        ## First process all grids separately using GRIDAE
        x = x.view(batch_size * num_grids, -1, self.msdf_k, self.msdf_k, self.msdf_k)
        obj_msdf = obj_msdf.view(batch_size * num_grids, 1, self.msdf_k, self.msdf_k, self.msdf_k)
        obj_feat, local_obj_feat = self.grid_ae.obj_encoder(obj_msdf)
        grid_feat, _ = self.grid_ae.encoder(x, cond=local_obj_feat)

        ## Reshape back to (B, N, ...)
        obj_feat = obj_feat.view(batch_size, num_grids, -1).permute(0, 2, 1)
        grid_feat = grid_feat.view(batch_size, num_grids, -1).permute(0, 2, 1)
        global_obj_feat, _ = self.obj_encoder(torch.cat([msdf_center.permute(0, 2, 1), obj_feat], dim=1))  # B x (3 + obj_feat_dim) x N
        x = torch.cat([msdf_center.permute(0, 2, 1), grid_feat, global_obj_feat], dim=1)  # B x (3 + grid_feat_dim + obj_feat_dim) x N

        z_e, _ = self.encoder(x)

        mean, logvar = self.latent_encoder(z_e)
        z_dist = torch.distributions.normal.Normal(mean, torch.exp(logvar))
        z_sample = z_dist.rsample()
        x = torch.cat([z_sample[:, :, None].repeat(1, 1, num_grids), global_obj_feat], dim=1)
        x, _ = self.decoder(x)

        x = x.view(batch_size * num_grids, -1).contiguous()
        c_hat, cse_hat, _ = self.grid_ae.decoder(x, cond=local_obj_feat[::-1])
        x_hat = torch.cat([c_hat, cse_hat], dim=1)
        x_hat = x_hat.view(batch_size, num_grids, self.msdf_k, self.msdf_k, self.msdf_k, self.out_dim).contiguous()    

        return x_hat, mean, logvar
    
    def sample(self, obj_msdf, msdf_center):
        ## Random noise
        z_dist = torch.distributions.normal.Normal(0, 1)
        z_sample = z_dist.rsample()

        batch_size, num_grids = obj_msdf.shape[:2]
        obj_feat, local_obj_feat = self.grid_ae.obj_encoder(obj_msdf)
        grid_feat, _ = self.grid_ae.encoder(x, cond=local_obj_feat)

        obj_feat = obj_feat.view(batch_size, num_grids, -1).permute(0, 2, 1)
        grid_feat = grid_feat.view(batch_size, num_grids, -1).permute(0, 2, 1)
        global_obj_feat = self.obj_encoder(torch.cat([msdf_center.permute(0, 2, 1), obj_feat], dim=1))  # B x (3 + obj_feat_dim) x N

        x = torch.cat([z_sample, global_obj_feat], dim=1)
        x = self.decoder(x)

        x = x.view(batch_size * num_grids, -1).contiguous()
        c_sample, cse_sample, _ = self.grid_ae.decoder(x, cond=local_obj_feat[::-1])
        x_sample = torch.cat([c_sample, cse_sample], dim=1)
        x_sample = x_sample.view(batch_size, num_grids, self.msdf_k, self.msdf_k, self.msdf_k, self.out_dim).contiguous()    

        return x_sample
