import torch
import torch.nn as nn
from ..pointnet_module import PointNet2cls, LatentEncoder, PointNet2seg
# from ..gridae.encoder import GridEncoder3D
# from ..gridae.decoder import GridDecoder3D
from ..gridae.gridae import GRIDAE

class GRID_VAE(nn.Module):
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
        super(GRID_VAE, self).__init__()
        # encode image into continuous latent space
        # self.obj_encoder = Encoder(obj_in_dim, h_dims, obj_n_res_layers, obj_res_h_dim, condition=False)
        self.grid_ae = GRIDAE(cfg=cfg.ae, obj_1d_feat=True)
        self.cfg = cfg.generator
        self.latent_dim = cfg.generator.latent.dim
        self.out_dim = cfg.ae.out_dim
        self.msdf_k = cfg.msdf.kernel_size
        # self.objgridencoder = GridEncoder3D(obj_in_dim, h_dims, res_h_dim, n_res_layers, feat_dim, N=N, condition=False)
        self.encoder = PointNet2cls(in_channel=cfg.ae.feat_dim + cfg.generator.glob_obj_feat_dim + 3,
                                    in_point=cfg.msdf.num_grids,
                                    hidden_dim=cfg.generator.encoder.hd,
                                    out_channel=cfg.generator.glob_feat_dim)
        # self.encoder = Pointnet(in_dim=cfg.ae.feat_dim + cfg.generator.glob_obj_feat_dim + 3,
        #                         hidden_dim=cfg.generator.encoder.hd,
        #                         out_dim=cfg.generator.glob_feat_dim)
        self.obj_encoder = PointNet2seg(in_dim=cfg.ae.obj_feat_dim+3,
                                        in_point=cfg.msdf.num_grids,
                                        hidden_dim=cfg.generator.obj_encoder.hd,
                                        out_dim=cfg.generator.glob_obj_feat_dim)
        self.latent_encoder = LatentEncoder(cfg.generator.glob_feat_dim, cfg.generator.latent.hd, self.latent_dim)
        # self.encoder = Encoder(in_dim, h_dims, n_res_layers, res_h_dim)
        # self.objgriddecoder = GridDecoder3D(latent_dim, h_dims[::-1], out_dim, n_res_layers, res_h_dim, condition=True)
        self.decoder = PointNet2seg(in_dim=self.latent_dim + cfg.generator.glob_obj_feat_dim,
                                    in_point=cfg.msdf.num_grids,
                                    hidden_dim=cfg.generator.decoder.hd,
                                    out_dim=cfg.ae.feat_dim)
        # self.decoder = Pointnet(in_dim=self.latent_dim + cfg.generator.glob_obj_feat_dim,
        #                         hidden_dim=cfg.generator.decoder.hd,
        #                         out_dim=cfg.ae.feat_dim)

        # Load pretrained autoencoder weights if provided (after all layers are initialized)
        self.pretrained_keys = []  # Track which parameters were loaded from checkpoint
        if hasattr(cfg.generator, 'ae_pretrained_weight') and cfg.generator.ae_pretrained_weight is not None:
            self._load_ae_pretrained_weights(cfg.generator.ae_pretrained_weight)
            # Freeze pretrained parameters initially
            if hasattr(cfg.generator, 'ae_freeze_until_epoch') and cfg.generator.ae_freeze_until_epoch > 0:
                self._freeze_pretrained_weights()

        # if save_img_embedding_map:
        #     self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        # else:
        #     self.img_to_embedding_map = None

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

    def unfreeze_pretrained_weights(self):
        """
        Unfreeze all parameters that were loaded from the pretrained checkpoint.
        Also restores BatchNorm layers to train mode for end-to-end training.
        """
        unfrozen_count = 0
        for name, param in self.named_parameters():
            if name in self.pretrained_keys:
                param.requires_grad = True
                unfrozen_count += 1

        # Restore grid_ae BatchNorm layers to train mode for end-to-end training
        bn_count = 0
        for module in self.grid_ae.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.train()
                # Re-enable running stats updates
                module.track_running_stats = True
                bn_count += 1

        print(f"Unfrozen {unfrozen_count} pretrained parameters in grid_ae for end-to-end training")
        print(f"Restored {bn_count} BatchNorm layers to train mode")

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
        _, global_obj_feat = self.obj_encoder(torch.cat([msdf_center.permute(0, 2, 1), obj_feat], dim=1))  # B x (3 + obj_feat_dim) x N
        x = torch.cat([msdf_center.permute(0, 2, 1), grid_feat, global_obj_feat], dim=1)  # B x (3 + grid_feat_dim + obj_feat_dim) x N

        z_e, _ = self.encoder(x)

        mean, logvar = self.latent_encoder(z_e)
        z_dist = torch.distributions.normal.Normal(mean, torch.exp(logvar))
        z_sample = z_dist.rsample()
        x = torch.cat([z_sample[:, :, None].repeat(1, 1, num_grids), global_obj_feat], dim=1)
        _, x = self.decoder(x)

        x = x.view(batch_size * num_grids, -1).contiguous()
        c_hat, cse_hat, _ = self.grid_ae.decoder(x, cond=local_obj_feat[::-1])
        x_hat = torch.cat([c_hat, cse_hat], dim=1)
        x_hat = x_hat.view(batch_size, num_grids, self.out_dim, self.msdf_k, self.msdf_k, self.msdf_k).contiguous()    

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
