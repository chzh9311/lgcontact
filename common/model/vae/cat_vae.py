"""
Modified from ContactGen (https://github.com/stevenlsw/contactgen.git)
"""
## Where the physics-aware encoder and decoder are defined.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet_module import Pointnet, PointNet2seg, LatentEncoder


class CAT_VAE(nn.Module):
    def __init__(self, cfg):
        super(CAT_VAE, self).__init__()
        self.num_parts = 16
        self.hc = cfg.generator.pointnet_hc
        self.embed_class = nn.Embedding(self.num_parts, self.hc)
        self.obj_encoder = PointNet2seg(in_dim=cfg.generator.obj_feature, hidden_dim=cfg.generator.pointnet_hc,
                                        in_point=cfg.data.n_obj_samples, out_dim=cfg.generator.pointnet_hc)
        self.encoder = CatEncoder(cfg)
        self.decoder = CatDecoder(cfg)

    def forward(self, verts_object, feat_object, contacts_object, partition_object):
        _, obj_cond = self.obj_encoder(torch.cat([verts_object, feat_object], -1).permute(0, 2, 1))
        z_contact, z_part, z_s_contact, z_s_part = self.encoder(self.embed_class, obj_cond, contacts_object,
                                                                                         partition_object)
        results = {'mean_contact': z_contact.mean, 'std_contact': z_contact.scale,
                   'mean_part': z_part.mean, 'std_part': z_part.scale}

        contacts_pred, partition_pred = self.decoder(self.embed_class, z_s_contact, z_s_part, obj_cond)
        results.update({'contacts_object': contacts_pred.permute(0, 2, 1),
                        'partition_object': partition_pred.permute(0, 2, 1)})
        return results

    def sample(self, verts_object, feat_object):
        bs = verts_object.shape[0]
        dtype = verts_object.dtype
        device = verts_object.device
        self.eval()
        with torch.no_grad():
            _, obj_cond = self.obj_encoder(torch.cat([verts_object, feat_object], -1).permute(0, 2, 1))
            z_gen_contact = np.random.normal(0., 1., size=(bs, self.latentD))
            z_gen_contact = torch.tensor(z_gen_contact, dtype=dtype).to(device)
            z_gen_part = np.random.normal(0., 1., size=(bs, self.latentD))
            z_gen_part = torch.tensor(z_gen_part, dtype=dtype).to(device)
            return self.decoder(self.embed_class, z_gen_contact, z_gen_part, obj_cond)

class CatEncoder(nn.Module):
    def __init__(self, cfg):
        super(CatEncoder, self).__init__()
        self.cfg = cfg
        self.n_neurons = cfg.generator.n_neurons
        self.latentD = cfg.generator.latentD
        self.hc = cfg.generator.pointnet_hc
        self.object_feature = cfg.generator.obj_feature

        self.num_parts = 16

        encode_dim = self.hc

        self.contact_encoder = Pointnet(in_dim=encode_dim + 1, hidden_dim=self.hc, out_dim=self.hc)
        self.part_encoder = Pointnet(in_dim=encode_dim + self.latentD + self.hc, hidden_dim=self.hc, out_dim=self.hc)
        # self.pressure_encoder = Pointnet(in_dim=encode_dim + self.hc + self.num_parts, hidden_dim=self.hc, out_dim=self.hc)

        self.contact_latent = LatentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)
        self.part_latent = LatentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)
        # self.pressure_latent = LatentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)

    def forward(self, embed_class, obj_cond, contacts_object, partition_object):
        contact_latent, _ = self.contact_encoder(torch.cat([obj_cond, contacts_object.unsqueeze(1)], dim=1))
        contact_mu, contact_std = self.contact_latent(contact_latent)
        z_contact = torch.distributions.normal.Normal(contact_mu, torch.exp(contact_std))
        z_s_contact = z_contact.rsample()

        partition_feat = embed_class(partition_object.argmax(dim=-1))
        part_latent, _ = self.part_encoder(
            torch.cat([obj_cond, z_s_contact.unsqueeze(dim=2).repeat(1, 1, obj_cond.shape[2]),
                       partition_feat.permute(0, 2, 1)], 1))
        part_mu, part_std = self.part_latent(part_latent)
        z_part = torch.distributions.normal.Normal(part_mu, torch.exp(part_std))
        # _, pressure_latent = self.pressure_encoder(torch.cat([obj_cond, partition_feat, pressure_object], -1))
        # pressure_mu, pressure_std = self.pressure_latent(pressure_latent)
        # z_pressure = torch.distributions.normal.Normal(pressure_mu, torch.exp(pressure_std))
        z_s_part = z_part.rsample()
        # z_s_pressure = z_pressure.rsample()

        return z_contact, z_part, z_s_contact, z_s_part


class CatDecoder(nn.Module):
    def __init__(self, cfg):
        super(CatDecoder, self).__init__()
        self.n_neurons = cfg.generator.n_neurons
        self.latentD = cfg.generator.latentD
        self.hc = cfg.generator.pointnet_hc
        self.object_feature = cfg.generator.obj_feature
        self.num_parts = 16
        encode_dim = self.hc

        self.contact_decoder = Pointnet(in_dim=encode_dim + self.latentD, hidden_dim=self.hc, out_dim=1)
        self.part_decoder = Pointnet(in_dim=encode_dim + self.latentD + self.latentD, hidden_dim=self.hc,
                                     out_dim=self.num_parts)
        # self.pressure_decoder = Pointnet(in_dim=self.hc + encode_dim + self.latentD, hidden_dim=self.hc, out_dim=self.num_parts)

    def forward(self, embed_class, z_contact, z_part, obj_cond):

        z_contact = z_contact.unsqueeze(dim=2).repeat(1, 1, obj_cond.shape[2])
        _, contacts_object = self.contact_decoder(torch.cat([z_contact, obj_cond], 1))
        contacts_object = torch.sigmoid(contacts_object)

        z_part = z_part.unsqueeze(dim=2).repeat(1, 1, obj_cond.shape[2])
        _, partition_object = self.part_decoder(torch.cat([z_part, obj_cond, z_contact], 1))
        # z_pressure = z_pressure.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)

        # if gt_partition_object is not None:
        #     partition_feat = embed_class(gt_partition_object.argmax(dim=-1))
        # else:
        #     partition_object_ = F.one_hot(partition_object.detach().argmax(dim=-1), num_classes=self.num_parts)
        #     partition_feat = embed_class(partition_object_.argmax(dim=-1))
        # pressure_object, _ = self.pressure_decoder(torch.cat([z_pressure, obj_cond, partition_feat], -1))
        return contacts_object, partition_object
    

